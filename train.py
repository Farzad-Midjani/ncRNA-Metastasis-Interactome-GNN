from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch_geometric.data import HeteroData
from tqdm import tqdm, trange

from config import Config
from data import (
    assert_supervision_edges_absent_from_graph,
    build_message_passing_graph,
    grouped_positive_train_test_split,
    make_undirected_message_graph,
    prepare_link_prediction_data,
    split_disjoint_message_and_supervision_edges,
)

def make_stratification_labels(labels: List[int], tasks: List[int]) -> np.ndarray:
    labels_array = np.asarray(labels, dtype=int)
    tasks_array = np.asarray(tasks, dtype=int)
    return tasks_array * 2 + labels_array

def select_global_threshold(
    labels: np.ndarray,
    probabilities: np.ndarray,
    min_threshold: float,
    max_threshold: float
) -> float:
    default_threshold = (min_threshold + max_threshold) / 2

    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return default_threshold

    precisions, recalls, candidate_thresholds = precision_recall_curve(labels, probabilities)
    if len(candidate_thresholds) == 0:
        return default_threshold

    f1_values = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
    valid_mask = (candidate_thresholds >= min_threshold) & (candidate_thresholds <= max_threshold)

    if not np.any(valid_mask):
        return default_threshold

    valid_indices = np.flatnonzero(valid_mask)
    best_index = valid_indices[np.argmax(f1_values[valid_indices])]
    return float(candidate_thresholds[best_index])

def train(
    model: nn.Module,
    data: HeteroData,
    train_edges: List[Tuple[int, int]],
    train_labels: List[int],
    train_tasks: List[int],
    optimizer: torch.optim.Optimizer,
    task_pos_weights: torch.Tensor,
    device: torch.device,
    batch_size: int = 256,
    max_norm: float = 1.0
) -> float:

    model.train()
    total_loss = 0
    num_batches = int(np.ceil(len(train_edges) / batch_size))

    for batch_idx in tqdm(range(num_batches), total=num_batches, desc="Training Batches"):
        optimizer.zero_grad()
        batch_start = batch_idx * batch_size
        batch_end = min((batch_idx + 1) * batch_size, len(train_edges))
        batch_edges = train_edges[batch_start:batch_end]
        batch_labels = train_labels[batch_start:batch_end]
        batch_tasks = train_tasks[batch_start:batch_end]


        shared_embeddings = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)


        loss = 0.0


        task_to_samples = {}
        for i, task in enumerate(batch_tasks):
            task_to_samples.setdefault(task, []).append(i)

        for task, indices in task_to_samples.items():
            if task >= model.out_channels:
                continue


            src_type, relation, dst_type = model.link_prediction_tasks[task]
            src_indices = torch.tensor([batch_edges[i][0] for i in indices], dtype=torch.long, device=device)
            dst_indices = torch.tensor([batch_edges[i][1] for i in indices], dtype=torch.long, device=device)
            labels = torch.tensor([batch_labels[i] for i in indices], dtype=torch.float, device=device)


            relation_idx = torch.tensor(task, device=device)
            r = model.relation_embeddings(relation_idx)


            src_adapted, dst_adapted = model.get_task_adapted_embeddings(shared_embeddings, task)


            h = src_adapted[src_indices]
            t = dst_adapted[dst_indices]


            W_r = torch.einsum('hrd,r->hd', model.core_tensor, r)


            h_W_r = torch.matmul(h, W_r)


            scores = torch.bmm(h_W_r.unsqueeze(1), t.unsqueeze(2)).squeeze(2).squeeze(1)


            task_loss = nn.functional.binary_cross_entropy_with_logits(
                scores, labels, pos_weight=task_pos_weights[task]
            )
            loss += task_loss


        loss = loss / len(task_to_samples) if task_to_samples else torch.tensor(0.0, device=device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    print(f"Average Training Loss: {avg_loss:.4f}\n")
    return avg_loss

def evaluate(
    model: nn.Module,
    data_graph: HeteroData,
    eval_edges: List[Tuple[int, int]],
    eval_labels: List[int],
    eval_tasks: List[int],
    device: torch.device,
    num_tasks: int,
    dynamic_threshold: bool = False,
    min_threshold: float = 0.3,
    max_threshold: float = 0.7,
    fixed_threshold: Optional[float] = None,
    task_pos_weights: Optional[torch.Tensor] = None
) -> Tuple[float, float, float, float, float, Dict[str, List[Any]]]:

    model.eval()
    with torch.no_grad():
        shared_embeddings = model(
            data_graph.x_dict,
            data_graph.edge_index_dict,
            data_graph.edge_attr_dict
        )

        all_predictions = []
        all_true_labels = []
        task_predictions = {task: [] for task in range(num_tasks)}
        task_true = {task: [] for task in range(num_tasks)}
        task_losses = {task: np.nan for task in range(num_tasks)}

        task_to_samples = {}
        for sample_index, task in enumerate(eval_tasks):
            task_to_samples.setdefault(int(task), []).append(sample_index)

        for task, indices in task_to_samples.items():
            if task >= model.out_channels:
                continue

            src_indices = torch.tensor(
                [eval_edges[i][0] for i in indices], dtype=torch.long, device=device
            )
            dst_indices = torch.tensor(
                [eval_edges[i][1] for i in indices], dtype=torch.long, device=device
            )
            labels = torch.tensor(
                [eval_labels[i] for i in indices], dtype=torch.float, device=device
            )

            relation_embedding = model.relation_embeddings(
                torch.tensor(task, dtype=torch.long, device=device)
            )
            src_adapted, dst_adapted = model.get_task_adapted_embeddings(
                shared_embeddings, task
            )

            h = src_adapted[src_indices]
            t = dst_adapted[dst_indices]
            W_r = torch.einsum('hrd,r->hd', model.core_tensor, relation_embedding)
            h_W_r = torch.matmul(h, W_r)
            scores = torch.bmm(h_W_r.unsqueeze(1), t.unsqueeze(2)).squeeze(2).squeeze(1)
            pos_weight = (
                task_pos_weights[task] if task_pos_weights is not None else None
            )
            task_losses[task] = nn.functional.binary_cross_entropy_with_logits(
                scores, labels, pos_weight=pos_weight
            ).item()
            probabilities = torch.sigmoid(scores)

            probabilities_np = probabilities.cpu().numpy()
            labels_np = labels.cpu().numpy()
            task_predictions[task] = probabilities_np
            task_true[task] = labels_np
            all_predictions.extend(probabilities_np.tolist())
            all_true_labels.extend(labels_np.tolist())

        all_predictions = np.asarray(all_predictions, dtype=float)
        all_true_labels = np.asarray(all_true_labels, dtype=int)
        valid_task_losses = [
            loss_value for loss_value in task_losses.values()
            if not np.isnan(loss_value)
        ]
        global_loss = (
            float(np.mean(valid_task_losses)) if valid_task_losses else np.nan
        )

        if dynamic_threshold:
            global_threshold = select_global_threshold(
                all_true_labels,
                all_predictions,
                min_threshold,
                max_threshold
            )
            print(
                f"Validation-selected global threshold: {global_threshold:.4f} "
                f"within [{min_threshold}, {max_threshold}]"
            )
        else:
            if fixed_threshold is None:
                global_threshold = (min_threshold + max_threshold) / 2
            else:
                global_threshold = float(fixed_threshold)
            global_threshold = float(
                np.clip(global_threshold, min_threshold, max_threshold)
            )
            print(f"Using fixed global threshold: {global_threshold:.4f}")

        global_predictions = (all_predictions >= global_threshold).astype(int)

        if len(all_true_labels) > 0:
            global_acc = accuracy_score(all_true_labels, global_predictions)
            global_f1 = f1_score(all_true_labels, global_predictions, zero_division=0)
            global_precision = precision_score(
                all_true_labels, global_predictions, zero_division=0
            )
            global_recall = recall_score(
                all_true_labels, global_predictions, zero_division=0
            )
            if len(np.unique(all_true_labels)) > 1:
                global_roc_auc = roc_auc_score(all_true_labels, all_predictions)
            else:
                global_roc_auc = np.nan

            print("\nGLOBAL EVALUATION (all tasks together)")
            print(
                f"Loss: {global_loss:.4f}, Accuracy: {global_acc:.4f}, "
                f"F1: {global_f1:.4f}, Precision: {global_precision:.4f}, "
                f"Recall: {global_recall:.4f}, ROC AUC: {global_roc_auc:.4f}"
            )
        else:
            global_acc = global_f1 = global_precision = global_recall = 0.0
            global_roc_auc = np.nan
            print("Cannot calculate global metrics: no evaluation samples.")

        losses = []
        acc = []
        f1 = []
        roc_auc = []
        precision = []
        recall = []
        thresholds = []

        print("\nPER-TASK EVALUATION (using the fixed global threshold)")
        for task in range(num_tasks):
            y_true = np.asarray(task_true[task])
            y_prob = np.asarray(task_predictions[task])

            if len(y_true) == 0:
                acc_task = f1_task = precision_task = recall_task = np.nan
                roc_auc_task = np.nan
                print(f"Warning: Task {task} has no samples.")
            else:
                y_pred = (y_prob >= global_threshold).astype(int)
                acc_task = accuracy_score(y_true, y_pred)
                f1_task = f1_score(y_true, y_pred, zero_division=0)
                precision_task = precision_score(y_true, y_pred, zero_division=0)
                recall_task = recall_score(y_true, y_pred, zero_division=0)
                roc_auc_task = (
                    roc_auc_score(y_true, y_prob)
                    if len(np.unique(y_true)) > 1
                    else np.nan
                )

            loss_task = task_losses[task]
            losses.append(loss_task)
            acc.append(acc_task)
            f1.append(f1_task)
            roc_auc.append(roc_auc_task)
            precision.append(precision_task)
            recall.append(recall_task)
            thresholds.append(global_threshold)

            print(
                f"Task {task}: Loss={loss_task:.4f}, Accuracy={acc_task:.4f}, "
                f"F1={f1_task:.4f}, Precision={precision_task:.4f}, "
                f"Recall={recall_task:.4f}, ROC AUC={roc_auc_task:.4f}"
            )

        metrics = {
            'loss': losses,
            'accuracy': acc,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'precision': precision,
            'recall': recall,
            'thresholds': thresholds,
            'global': {
                'loss': global_loss,
                'accuracy': global_acc,
                'f1_score': global_f1,
                'roc_auc': global_roc_auc,
                'precision': global_precision,
                'recall': global_recall,
                'threshold': global_threshold
            }
        }

    return (
        global_acc,
        global_f1,
        global_roc_auc,
        global_precision,
        global_recall,
        metrics
    )

def cross_validate(
    model_class: Any,
    data: HeteroData,
    link_prediction_tasks: List[Tuple[str, str, str]],
    mappings: Dict[str, Dict[str, Any]],
    config: Config,
    feature_dims: Dict[str, Dict[str, int]],
    all_positive_sets: Dict[int, set],
    num_cancer_nodes: int = 0,
    num_csmi_nodes: int = 0,
    num_met_nodes: int = 0,
    train_positive_edges: Optional[List[Tuple[int, int]]] = None,
    train_positive_tasks: Optional[List[int]] = None,
    train_groups: Optional[List[str]] = None
) -> Dict[str, List[Any]]:
    print(
        f"\nStarting group-aware cross-validation with "
        f"{config.cross_val_folds} requested folds..."
    )

    if not train_positive_edges or train_positive_tasks is None or not train_groups:
        print("No positive development data provided to cross_validate.")
        return {}

    positive_edges = np.asarray(train_positive_edges, dtype=int)
    positive_tasks = np.asarray(train_positive_tasks, dtype=int)
    groups = np.asarray(train_groups, dtype=object)

    unique_groups_per_task = [
        len(set(groups[positive_tasks == task_idx].tolist()))
        for task_idx in sorted(set(positive_tasks.tolist()))
    ]
    effective_folds = min(
        config.cross_val_folds,
        len(set(groups.tolist())),
        min(unique_groups_per_task) if unique_groups_per_task else 0
    )
    if effective_folds < 2:
        print("Not enough biological RNA groups for grouped cross-validation.")
        return {}
    if effective_folds != config.cross_val_folds:
        print(
            f"Using {effective_folds} folds because the smallest task has only "
            f"{min(unique_groups_per_task)} distinct RNA groups."
        )

    splitter = StratifiedGroupKFold(
        n_splits=effective_folds,
        shuffle=True,
        random_state=42
    )

    fold_metrics = {
        'global_loss': [],
        'global_accuracy': [],
        'global_f1_score': [],
        'global_roc_auc': [],
        'global_precision': [],
        'global_recall': [],
        'global_threshold': [],
        'per_task': {
            'loss': [],
            'accuracy': [],
            'f1_score': [],
            'roc_auc': [],
            'precision': [],
            'recall': []
        }
    }

    split_iterator = splitter.split(
        positive_edges,
        positive_tasks,
        groups
    )
    for fold, (fold_train_idx, fold_test_idx) in enumerate(
        tqdm(
            split_iterator,
            total=effective_folds,
            desc="Group-Aware Cross-Validation Folds"
        )
    ):
        print(f"\n--- Fold {fold + 1}/{effective_folds} ---")

        fold_train_positive_edges = positive_edges[fold_train_idx].tolist()
        fold_train_positive_tasks = positive_tasks[fold_train_idx].tolist()
        fold_train_groups = groups[fold_train_idx].tolist()
        fold_test_positive_edges = positive_edges[fold_test_idx].tolist()
        fold_test_positive_tasks = positive_tasks[fold_test_idx].tolist()
        fold_test_groups = groups[fold_test_idx].tolist()

        (
            fold_fit_positive_edges,
            fold_val_positive_edges,
            fold_fit_positive_tasks,
            fold_val_positive_tasks,
            fold_fit_groups,
            fold_val_groups
        ) = grouped_positive_train_test_split(
            fold_train_positive_edges,
            fold_train_positive_tasks,
            fold_train_groups,
            test_size=0.1,
            random_state=1000 + fold,
            split_name=f"Fold {fold + 1} fit/validation split"
        )


        (
            fold_message_positive_edges,
            fold_supervision_positive_edges,
            fold_message_positive_tasks,
            fold_supervision_positive_tasks
        ) = split_disjoint_message_and_supervision_edges(
            fold_fit_positive_edges,
            fold_fit_positive_tasks,
            supervision_ratio=config.disjoint_train_ratio,
            random_state=1500 + fold,
            split_name=f"Fold {fold + 1} disjoint training-edge split"
        )

        directed_fold_graph = build_message_passing_graph(
            data,
            link_prediction_tasks,
            fold_message_positive_edges,
            fold_message_positive_tasks
        )
        assert_supervision_edges_absent_from_graph(
            directed_fold_graph,
            link_prediction_tasks,
            fold_supervision_positive_edges,
            fold_supervision_positive_tasks,
            check_name=f"Fold {fold + 1} leakage check"
        )
        fold_message_graph = make_undirected_message_graph(directed_fold_graph)

        reserved_negatives = {
            task_idx: set() for task_idx in range(len(link_prediction_tasks))
        }
        fold_fit_edges, fold_fit_labels, fold_fit_tasks = (
            prepare_link_prediction_data(
                directed_fold_graph,
                link_prediction_tasks,
                fold_supervision_positive_edges,
                fold_supervision_positive_tasks,
                all_positive_sets,
                min_common_neighbors=1,
                random_state=2000 + fold,
                reserved_negatives=reserved_negatives
            )
        )
        fold_val_edges, fold_val_labels, fold_val_tasks = (
            prepare_link_prediction_data(
                directed_fold_graph,
                link_prediction_tasks,
                fold_val_positive_edges,
                fold_val_positive_tasks,
                all_positive_sets,
                min_common_neighbors=1,
                random_state=3000 + fold,
                reserved_negatives=reserved_negatives
            )
        )
        fold_test_edges, fold_test_labels, fold_test_tasks = (
            prepare_link_prediction_data(
                directed_fold_graph,
                link_prediction_tasks,
                fold_test_positive_edges,
                fold_test_positive_tasks,
                all_positive_sets,
                min_common_neighbors=1,
                random_state=4000 + fold,
                reserved_negatives=reserved_negatives
            )
        )

        print(
            f"Fitting samples: {len(fold_fit_edges)}, "
            f"validation samples: {len(fold_val_edges)}, "
            f"testing samples: {len(fold_test_edges)}"
        )

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model_class(
            hidden_channels=config.hidden_channels,
            link_types=list(fold_message_graph.edge_types),
            link_prediction_tasks=config.link_prediction_tasks,
            heads=config.heads,
            dropout=config.dropout,
            dropout_rates=config.dropout_rates,
            feature_dims=feature_dims,
            num_cancer_nodes=num_cancer_nodes,
            num_csmi_nodes=num_csmi_nodes,
            num_met_nodes=num_met_nodes,
            num_layers=config.num_layers,
            use_layer_norm=True
        ).to(device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        task_pos_weights = torch.ones(
            config.out_channels, dtype=torch.float, device=device
        )
        for task_idx in range(config.out_channels):
            task_labels = [
                label
                for label, sample_task in zip(fold_fit_labels, fold_fit_tasks)
                if sample_task == task_idx
            ]
            num_pos = sum(task_labels)
            num_neg = len(task_labels) - num_pos
            task_pos_weights[task_idx] = (
                num_neg / num_pos if num_pos > 0 else 1.0
            )

        best_val_auc = -np.inf
        patience_counter = 0
        checkpoint_path = os.path.join(
            config.output_dir, f'best_model_fold_{fold + 1}.pt'
        )

        for epoch in trange(
            1,
            config.epochs + 1,
            desc=f"Fold {fold + 1}/{effective_folds} - Epochs",
            total=config.epochs,
            leave=False
        ):
            loss = train(
                model,
                fold_message_graph,
                fold_fit_edges,
                fold_fit_labels,
                fold_fit_tasks,
                optimizer,
                task_pos_weights,
                device,
                batch_size=config.batch_size,
                max_norm=1.0
            )

            _, _, val_auc, val_precision, val_recall, val_metrics = evaluate(
                model,
                fold_message_graph,
                fold_val_edges,
                fold_val_labels,
                fold_val_tasks,
                device,
                config.out_channels,
                dynamic_threshold=False,
                fixed_threshold=0.5,
                task_pos_weights=task_pos_weights
            )
            val_loss = val_metrics['global']['loss']

            print(
                f"Epoch {epoch}, Training Loss: {loss:.4f}, "
                f"Validation Loss: {val_loss:.4f}, "
                f"Validation Global ROC AUC: {val_auc:.4f}, "
                f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}"
            )

            score_for_selection = -np.inf if np.isnan(val_auc) else val_auc
            if epoch == 1 or score_for_selection > best_val_auc:
                best_val_auc = score_for_selection
                patience_counter = 0
                torch.save(model.state_dict(), checkpoint_path)
            else:
                patience_counter += 1
                if patience_counter >= config.patience:
                    print("Early stopping: validation ROC AUC did not improve.")
                    break

        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

        *_, validation_metrics = evaluate(
            model,
            fold_message_graph,
            fold_val_edges,
            fold_val_labels,
            fold_val_tasks,
            device,
            config.out_channels,
            dynamic_threshold=True,
            task_pos_weights=task_pos_weights
        )
        fold_threshold = validation_metrics['global']['threshold']

        (
            global_acc,
            global_f1,
            global_roc_auc,
            global_precision,
            global_recall,
            all_metrics
        ) = evaluate(
            model,
            fold_message_graph,
            fold_test_edges,
            fold_test_labels,
            fold_test_tasks,
            device,
            config.out_channels,
            dynamic_threshold=False,
            fixed_threshold=fold_threshold,
            task_pos_weights=task_pos_weights
        )
        global_loss = all_metrics['global']['loss']

        fold_metrics['global_loss'].append(global_loss)
        fold_metrics['global_accuracy'].append(global_acc)
        fold_metrics['global_f1_score'].append(global_f1)
        fold_metrics['global_roc_auc'].append(global_roc_auc)
        fold_metrics['global_precision'].append(global_precision)
        fold_metrics['global_recall'].append(global_recall)
        fold_metrics['global_threshold'].append(fold_threshold)
        for metric_name in ['loss', 'accuracy', 'f1_score', 'roc_auc', 'precision', 'recall']:
            fold_metrics['per_task'][metric_name].append(
                all_metrics[metric_name]
            )

        print(
            f"Fold {fold + 1}: Loss={global_loss:.4f}, "
            f"Accuracy={global_acc:.4f}, "
            f"F1={global_f1:.4f}, Precision={global_precision:.4f}, "
            f"Recall={global_recall:.4f}, ROC AUC={global_roc_auc:.4f}, "
            f"validation threshold={fold_threshold:.4f}"
        )

    print("\nCross-Validation Results:")
    for metric in [
        'global_loss',
        'global_accuracy',
        'global_f1_score',
        'global_roc_auc',
        'global_precision',
        'global_recall',
        'global_threshold'
    ]:
        values = np.asarray(fold_metrics[metric], dtype=float)
        print(
            f"{metric.replace('global_', '').replace('_', ' ').title()}: "
            f"{np.nanmean(values):.4f} ± {np.nanstd(values):.4f}"
        )

    print("Cross-validation complete.\n")
    return fold_metrics
