from __future__ import annotations

import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from sklearn.utils import resample
from tqdm import trange

from config import Config
from data import (
    add_edges,
    add_node_features,
    assert_supervision_edges_absent_from_graph,
    build_message_passing_graph,
    create_unique_nodes,
    extract_positive_link_prediction_data,
    grouped_positive_train_test_split,
    initialize_hetero_data,
    load_and_standardize_data,
    make_undirected_message_graph,
    map_nodes_to_indices,
    prepare_link_prediction_data,
    preprocess_data,
    split_disjoint_message_and_supervision_edges,
    validate_edges,
)
from model import HierarchicalRNAInteractionGNN
from reporting import (
    convert_to_networkx_with_type,
    generate_output,
    show_graph_info,
    visualize_graph,
    visualize_graph_3d,
)
from train import cross_validate, evaluate, train

def main(config: Config | None = None):
    """Run the RNA-cancer link-prediction pipeline."""
    config = config or Config()
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    print("Random seeds set for reproducibility.\n")

    min_threshold = 0.3
    max_threshold = 0.7
    print(
        f"Validation threshold optimization is constrained to "
        f"[{min_threshold}, {max_threshold}].\n"
    )

    dataframes = load_and_standardize_data(config)
    dataframes = preprocess_data(dataframes, config)
    nodes = create_unique_nodes(dataframes)

    mappings = map_nodes_to_indices(nodes)
    data = initialize_hetero_data(nodes)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    data = add_node_features(
        data, nodes, mappings, dataframes, config, device
    )

    print(f"'circRNA' features shape: {data['circRNA'].x.shape}")
    print(f"'miRNA' features shape: {data['miRNA'].x.shape}")
    print(f"'Cancer' embeddings shape: {data['Cancer'].x.shape}")
    print(f"'CSMI' embeddings shape: {data['CSMI'].x.shape}")
    print(
        f"'MET' embeddings shape: "
        f"{data['MET'].x.shape}\n"
    )

    data = add_edges(data, mappings, dataframes, config)
    validate_edges(data)

    G = convert_to_networkx_with_type(data, mappings)
    visualize_graph(
        G,
        node_size=300,
        font_size=8,
        save_path=os.path.join(config.output_dir, 'rna_cancer_graph.png')
    )
    visualize_graph_3d(
        G,
        save_path=os.path.join(config.output_dir, 'rna_cancer_graph_3d.png')
    )
    show_graph_info(
        data, mappings, config, num_samples=5, show_feature_stats=True
    )

    feature_dims = {
        node_type: {
            'feature_dim': (
                data[node_type].x.size(1)
                if hasattr(data[node_type], 'x') and data[node_type].x is not None
                else config.hidden_channels
            ),
            'num_nodes': data[node_type].num_nodes
        }
        for node_type in data.node_types
    }
    print("Node feature dimensions:")
    for node_type, dim_dict in feature_dims.items():
        print(f"{node_type}: {dim_dict['feature_dim']}")
    print()

    (
        all_positive_edges,
        all_positive_tasks,
        all_positive_groups,
        all_positive_sets
    ) = extract_positive_link_prediction_data(
        data,
        config.link_prediction_tasks
    )

    print("\nCreating group-aware development/test split on positive edges...")
    (
        development_positive_edges,
        test_positive_edges,
        development_positive_tasks,
        test_positive_tasks,
        development_groups,
        test_groups
    ) = grouped_positive_train_test_split(
        all_positive_edges,
        all_positive_tasks,
        all_positive_groups,
        test_size=0.2,
        random_state=42,
        split_name='Development/final-test split'
    )

    cross_validate(
        HierarchicalRNAInteractionGNN,
        data,
        config.link_prediction_tasks,
        mappings,
        config,
        feature_dims,
        all_positive_sets=all_positive_sets,
        num_cancer_nodes=data['Cancer'].num_nodes,
        num_csmi_nodes=data['CSMI'].num_nodes,
        num_met_nodes=data['MET'].num_nodes,
        train_positive_edges=development_positive_edges,
        train_positive_tasks=development_positive_tasks,
        train_groups=development_groups
    )

    print("\nCreating group-aware fitting/validation split on positives...")
    (
        fit_positive_edges,
        validation_positive_edges,
        fit_positive_tasks,
        validation_positive_tasks,
        fit_groups,
        validation_groups
    ) = grouped_positive_train_test_split(
        development_positive_edges,
        development_positive_tasks,
        development_groups,
        test_size=0.1,
        random_state=84,
        split_name='Final fitting/validation split'
    )


    (
        final_message_positive_edges,
        final_supervision_positive_edges,
        final_message_positive_tasks,
        final_supervision_positive_tasks
    ) = split_disjoint_message_and_supervision_edges(
        fit_positive_edges,
        fit_positive_tasks,
        supervision_ratio=config.disjoint_train_ratio,
        random_state=90,
        split_name='Final disjoint training-edge split'
    )


    directed_final_graph = build_message_passing_graph(
        data,
        config.link_prediction_tasks,
        final_message_positive_edges,
        final_message_positive_tasks
    )
    assert_supervision_edges_absent_from_graph(
        directed_final_graph,
        config.link_prediction_tasks,
        final_supervision_positive_edges,
        final_supervision_positive_tasks,
        check_name='Final training leakage check'
    )
    final_message_graph = make_undirected_message_graph(directed_final_graph)
    validate_edges(final_message_graph)


    reserved_negatives = {
        task_idx: set() for task_idx in range(config.out_channels)
    }
    fit_edges, fit_labels, fit_tasks = prepare_link_prediction_data(
        directed_final_graph,
        config.link_prediction_tasks,
        final_supervision_positive_edges,
        final_supervision_positive_tasks,
        all_positive_sets,
        min_common_neighbors=1,
        random_state=101,
        reserved_negatives=reserved_negatives
    )
    validation_edges, validation_labels, validation_tasks = (
        prepare_link_prediction_data(
            directed_final_graph,
            config.link_prediction_tasks,
            validation_positive_edges,
            validation_positive_tasks,
            all_positive_sets,
            min_common_neighbors=1,
            random_state=102,
            reserved_negatives=reserved_negatives
        )
    )
    test_edges, test_labels, test_tasks = prepare_link_prediction_data(
        directed_final_graph,
        config.link_prediction_tasks,
        test_positive_edges,
        test_positive_tasks,
        all_positive_sets,
        min_common_neighbors=1,
        random_state=103,
        reserved_negatives=reserved_negatives
    )

    print(
        f"Final fitting samples: {len(fit_edges)}, "
        f"validation samples: {len(validation_edges)}, "
        f"test samples: {len(test_edges)}\n"
    )

    model = HierarchicalRNAInteractionGNN(
        hidden_channels=config.hidden_channels,
        link_types=list(final_message_graph.edge_types),
        link_prediction_tasks=config.link_prediction_tasks,
        heads=config.heads,
        dropout=config.dropout,
        dropout_rates=config.dropout_rates,
        feature_dims=feature_dims,
        num_cancer_nodes=data['Cancer'].num_nodes,
        num_csmi_nodes=data['CSMI'].num_nodes,
        num_met_nodes=data['MET'].num_nodes,
        num_layers=config.num_layers,
        use_layer_norm=True
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )
    model.lr_scheduler = scheduler

    print("\nBalancing only the fitting supervision samples...")
    fit_edges_array = np.asarray(fit_edges)
    fit_labels_array = np.asarray(fit_labels, dtype=int)
    fit_tasks_array = np.asarray(fit_tasks, dtype=int)

    positive_indices = np.flatnonzero(fit_labels_array == 1)
    negative_indices = np.flatnonzero(fit_labels_array == 0)

    positive_edges = fit_edges_array[positive_indices]
    positive_tasks = fit_tasks_array[positive_indices]
    negative_edges = fit_edges_array[negative_indices]
    negative_tasks = fit_tasks_array[negative_indices]

    if len(positive_edges) == 0 or len(negative_edges) == 0:
        balanced_edges = fit_edges_array
        balanced_labels = fit_labels_array
        balanced_tasks = fit_tasks_array
        print("Skipping oversampling because one class is absent.")
    elif len(positive_edges) < len(negative_edges):
        additional_count = len(negative_edges) - len(positive_edges)
        extra_edges, extra_tasks = resample(
            positive_edges,
            positive_tasks,
            replace=True,
            n_samples=additional_count,
            random_state=42
        )
        balanced_edges = np.vstack(
            [positive_edges, extra_edges, negative_edges]
        )
        balanced_labels = np.hstack(
            [
                np.ones(len(positive_edges) + len(extra_edges), dtype=int),
                np.zeros(len(negative_edges), dtype=int)
            ]
        )
        balanced_tasks = np.hstack(
            [positive_tasks, extra_tasks, negative_tasks]
        )
        print(
            f"Balanced fitting set: "
            f"{len(positive_edges) + len(extra_edges)} positives and "
            f"{len(negative_edges)} negatives."
        )
    else:
        balanced_edges = np.vstack([positive_edges, negative_edges])
        balanced_labels = np.hstack(
            [
                np.ones(len(positive_edges), dtype=int),
                np.zeros(len(negative_edges), dtype=int)
            ]
        )
        balanced_tasks = np.hstack([positive_tasks, negative_tasks])
        print("No oversampling was needed.")

    shuffle_indices = np.random.permutation(len(balanced_edges))
    balanced_edges = balanced_edges[shuffle_indices].tolist()
    balanced_labels = balanced_labels[shuffle_indices].tolist()
    balanced_tasks = balanced_tasks[shuffle_indices].tolist()

    task_pos_weights = torch.ones(
        config.out_channels, dtype=torch.float, device=device
    )
    for task in range(config.out_channels):
        task_labels = [
            label for label, sample_task in zip(balanced_labels, balanced_tasks)
            if sample_task == task
        ]
        num_pos = sum(task_labels)
        num_neg = len(task_labels) - num_pos
        task_pos_weights[task] = num_neg / num_pos if num_pos > 0 else 1.0
        print(f"Task {task} pos_weight: {task_pos_weights[task].item():.4f}")

    best_val_auc = -np.inf
    patience_counter = 0
    best_epoch = 0
    checkpoint_path = os.path.join(config.output_dir, 'best_model.pt')
    train_losses = []
    validation_losses = []
    validation_aucs = []

    print("\nStarting final training with validation-based early stopping...")
    for epoch in trange(
        1,
        config.epochs + 1,
        desc="Final Model Training",
        total=config.epochs
    ):
        loss = train(
            model,
            final_message_graph,
            balanced_edges,
            balanced_labels,
            balanced_tasks,
            optimizer,
            task_pos_weights,
            device,
            batch_size=config.batch_size,
            max_norm=1.0
        )
        train_losses.append(loss)

        (
            val_acc,
            val_f1,
            val_auc,
            val_precision,
            val_recall,
            val_metrics
        ) = evaluate(
            model,
            final_message_graph,
            validation_edges,
            validation_labels,
            validation_tasks,
            device,
            config.out_channels,
            dynamic_threshold=False,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            fixed_threshold=0.5,
            task_pos_weights=task_pos_weights
        )
        val_loss = val_metrics['global']['loss']
        validation_losses.append(val_loss)
        validation_aucs.append(val_auc)

        scheduler_metric = -np.inf if np.isnan(val_auc) else val_auc
        scheduler.step(scheduler_metric)

        print(
            f"Epoch {epoch}, Training Loss: {loss:.4f}, "
            f"Validation Loss: {val_loss:.4f}, "
            f"Validation ROC AUC: {val_auc:.4f}, "
            f"Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}, "
            f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if epoch == 1 or scheduler_metric > best_val_auc:
            best_val_auc = scheduler_metric
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'validation_loss': val_loss,
                    'validation_roc_auc': val_auc
                },
                checkpoint_path
            )
            print(
                f"New best model saved with validation ROC AUC "
                f"{val_auc:.4f}."
            )
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                print(
                    f"Early stopping after {epoch} epochs; validation "
                    f"ROC AUC did not improve for {config.patience} epochs."
                )
                break

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(validation_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('BCE Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(validation_aucs)
    plt.title('Validation Global ROC AUC')
    plt.xlabel('Epoch')
    plt.ylabel('ROC AUC')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(config.output_dir, 'training_metrics.png'))
    plt.close()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(
        f"\nLoaded best model from epoch {checkpoint['epoch']} with "
        f"validation ROC AUC {checkpoint['validation_roc_auc']:.4f}."
    )


    (
        val_acc,
        val_f1,
        val_auc,
        val_precision,
        val_recall,
        validation_metrics
    ) = evaluate(
        model,
        final_message_graph,
        validation_edges,
        validation_labels,
        validation_tasks,
        device,
        config.out_channels,
        dynamic_threshold=True,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        task_pos_weights=task_pos_weights
    )
    val_loss = validation_metrics['global']['loss']
    best_threshold = validation_metrics['global']['threshold']

    checkpoint['global_threshold'] = best_threshold
    checkpoint['validation_loss'] = val_loss
    checkpoint['validation_accuracy'] = val_acc
    checkpoint['validation_f1'] = val_f1
    checkpoint['validation_precision'] = val_precision
    checkpoint['validation_recall'] = val_recall
    torch.save(checkpoint, checkpoint_path)


    print("\nEvaluating once on the untouched final test set...")
    (
        test_acc,
        test_f1,
        test_roc_auc,
        test_precision,
        test_recall,
        test_metrics
    ) = evaluate(
        model,
        final_message_graph,
        test_edges,
        test_labels,
        test_tasks,
        device,
        config.out_channels,
        dynamic_threshold=False,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        fixed_threshold=best_threshold,
        task_pos_weights=task_pos_weights
    )
    test_loss = test_metrics['global']['loss']

    final_threshold = best_threshold
    print(
        f"Test Global Metrics: Loss={test_loss:.4f}, "
        f"Accuracy={test_acc:.4f}, "
        f"F1={test_f1:.4f}, Precision={test_precision:.4f}, "
        f"Recall={test_recall:.4f}, ROC AUC={test_roc_auc:.4f}, "
        f"Fixed validation threshold={final_threshold:.4f}\n"
    )

    test_results = {
        'global': {
            'loss': test_loss,
            'accuracy': test_acc,
            'f1_score': test_f1,
            'roc_auc': test_roc_auc,
            'precision': test_precision,
            'recall': test_recall,
            'threshold': final_threshold,
            'threshold_source': 'validation'
        },
        'per_task_metrics': {
            'loss': test_metrics['loss'],
            'accuracy': test_metrics['accuracy'],
            'f1_score': test_metrics['f1_score'],
            'roc_auc': test_metrics['roc_auc'],
            'precision': test_metrics['precision'],
            'recall': test_metrics['recall'],
            'thresholds': test_metrics['thresholds']
        }
    }
    test_results_path = os.path.join(config.output_dir, 'test_results.json')
    with open(test_results_path, 'w', encoding='utf-8') as file:
        json.dump(test_results, file, indent=4)
    print(f"Test results saved to {test_results_path}")

    print("\nGenerating predictions for RNA-cancer interactions...")


    prediction_graph = final_message_graph
    pred_df = generate_output(
        model,
        prediction_graph,
        mappings,
        config,
        device,
        global_threshold=best_threshold,
        min_threshold=min_threshold,
        max_threshold=max_threshold,
        known_positive_sets=all_positive_sets
    )
    print(
        f"Generated {len(pred_df)} potential RNA-cancer interaction "
        f"predictions."
    )

    print("\n========== Analysis Complete ==========")
    print(f"Best epoch selected by validation ROC AUC: {best_epoch}")
    print(f"Best validation ROC AUC: {best_val_auc:.4f}")
    print(f"Validation loss at best checkpoint: {val_loss:.4f}")
    print(f"Validation-selected threshold: {best_threshold:.4f}")
    print(f"Test Global Loss: {test_loss:.4f}")
    print(f"Test Global ROC AUC: {test_roc_auc:.4f}")
    print(f"Test Global F1 Score: {test_f1:.4f}")
    print(f"All results and predictions saved to: {config.output_dir}")
    print("=======================================\n")


if __name__ == "__main__":
    main()
