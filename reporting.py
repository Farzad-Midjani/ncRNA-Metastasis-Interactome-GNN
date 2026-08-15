from __future__ import annotations

import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import pyvista as pv
import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from config import Config

def convert_to_networkx_with_type(hetero_data: HeteroData, mappings: Dict[str, Dict[str, Any]]) -> nx.DiGraph:
    """
    Convert HeteroData to NetworkX directed graph with 'type' attributes.

    Parameters:
    - hetero_data (HeteroData): The heterogeneous graph data.
    - mappings (dict): Dictionary containing node_to_idx and idx_to_node mappings for each type.

    Returns:
    - G (networkx.DiGraph): The converted NetworkX directed graph.
    """
    print("Converting HeteroData to NetworkX graph for visualization...")
    G = nx.DiGraph()


    for node_type, mapping in mappings.items():
        node_names = list(mapping['node_to_idx'].keys())
        for name in node_names:
            G.add_node(name, type=node_type)
    print(f"Added {G.number_of_nodes()} nodes to NetworkX graph.")


    for edge_type in hetero_data.edge_types:
        src_type, relation, dst_type = edge_type
        src_nodes = list(mappings[src_type]['node_to_idx'].keys())
        dst_nodes = list(mappings[dst_type]['node_to_idx'].keys())
        edge_storage = hetero_data[edge_type]
        if not hasattr(edge_storage, 'edge_index'):
            print(f"Warning: Edge type {edge_type} has no 'edge_index'. Skipping edges for this type.")
            continue
        edge_index = edge_storage.edge_index.numpy()

        for src, dst in zip(edge_index[0], edge_index[1]):
            src_name = src_nodes[src]
            dst_name = dst_nodes[dst]
            G.add_edge(src_name, dst_name, relation=relation)

    print(f"Added {G.number_of_edges()} edges to NetworkX graph.\n")
    return G

def visualize_graph(
    G: nx.DiGraph,
    node_size: int = 400,
    font_size: int = 6,
    dpi: int = 400,
    save_path: str = None
):

    print("Visualizing the graph...")
    plt.figure(figsize=(15, 15), dpi=dpi)
    pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)


    type_colors = {
        'miRNA': '#1f78b4',
        'circRNA': '#e31a1c',
        'Cancer': '#33a02c',
        'CSMI': '#ff7f00',
        'MET': '#6a3d9a'
    }

    type_shapes = {
        'miRNA': 'o',
        'circRNA': 'o',
        'Cancer': 'o',
        'CSMI': 'o',
        'MET': 'o'
    }


    node_groups = {}
    for node, data_edge in G.nodes(data=True):
        node_type = data_edge.get('type', 'Unknown')
        node_groups.setdefault(node_type, []).append(node)


    for node_type, nodes in node_groups.items():
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=nodes,
            node_shape=type_shapes.get(node_type, 'o'),
            node_color=type_colors.get(node_type, 'grey'),
            node_size=node_size,
            label=node_type,
            alpha=0.9
        )


    relation_colors = {
        'regulates': '#a6cee3',
        'inhibits': '#fb9a99',
        'related_to': '#b2df8a',
        'represents': '#fdbf6f',
        'interacts_with': '#cab2d6'
    }


    edges_by_relation = {}
    for u, v, data_edge in G.edges(data=True):
        relation = data_edge.get('relation', 'other')
        edges_by_relation.setdefault(relation, []).append((u, v))

    for relation, edges in edges_by_relation.items():
        nx.draw_networkx_edges(
            G, pos,
            edgelist=edges,
            width=2,
            edge_color=relation_colors.get(relation, 'grey'),
            style='solid' if relation != 'inhibits' else 'dashed',
            alpha=0.7,
            label=relation
        )


    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    node_legend_elements = []
    for node_type, color in type_colors.items():
        node_legend_elements.append(Line2D([], [], marker=type_shapes.get(node_type, 'o'), color='w', label=node_type,
                                         markerfacecolor=color, markersize=10))

    edge_legend_elements = []
    for relation, color in relation_colors.items():
        style = 'solid' if relation != 'inhibits' else 'dashed'
        edge_legend_elements.append(Line2D([], [], color=color, linestyle=style, label=relation, linewidth=2))

    plt.legend(handles=node_legend_elements + edge_legend_elements, fontsize=12, loc='upper right')

    plt.title("RNA-Cancer Interaction Graph", fontsize=24)
    plt.axis('off')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, format='png', dpi=dpi, bbox_inches='tight')
        print(f"Graph saved to {save_path}")

    plt.show()
    print("Graph visualization complete.\n")

def visualize_graph_3d(
    G: nx.DiGraph,
    node_size: float = 0.03,
    edge_width: float = 1.0,
    node_colors: Optional[Dict[str, str]] = None,
    edge_colors: Optional[Dict[str, str]] = None,
    save_path: Optional[str] = None,
    seed: int = 42,
    background: str = 'white',
    show_grid: bool = True,
    camera_position: str = 'iso',
    force_iterations: int = 200,
    height: int = 1600,
    width: int = 2400
) -> None:
    """
    3D visualization of the NetworkX graph using PyVista.
    """

    if node_colors is None:
        node_colors = {
            'miRNA': '#7E57C2',
            'circRNA': '#42A5F5',
            'Cancer': '#26A69A',
            'CSMI': '#EF5350',
            'MET': '#AB47BC',
            'Unknown': '#9E9E9E'
        }

    if edge_colors is None:
        edge_colors = {
            'regulates': '#B39DDB',
            'inhibits': '#81D4FA',
            'related_to': '#80CBC4',
            'represents': '#FFAB91',
            'interacts_with': '#CE93D8',
            'other': '#FFCC80'
        }


    node_types = nx.get_node_attributes(G, 'type')
    unique_node_types = set(node_types.values())


    pos_2d = nx.spring_layout(G, k=0.15, iterations=force_iterations, seed=seed)


    pos_3d = {}
    for node, (x, y) in pos_2d.items():
        z = np.random.uniform(-0.5, 0.5)
        pos_3d[node] = (x, y, z)


    plotter = pv.Plotter(window_size=[width, height])
    plotter.set_background(background)


    for node, (x, y, z) in pos_3d.items():
        node_type = node_types.get(node, 'Unknown')
        color = node_colors.get(node_type, 'grey')

        sphere = pv.Sphere(radius=node_size, center=(x, y, z), theta_resolution=32, phi_resolution=32)
        plotter.add_mesh(
            sphere,
            color=color,
            name=str(node),
            smooth_shading=True,
            opacity=0.9,
            show_edges=False
        )


    for u, v, data_edge in G.edges(data=True):
        relation = data_edge.get('relation', 'other')
        color = edge_colors.get(relation, 'grey')
        start = pos_3d[u]
        end = pos_3d[v]


        midpoint = tuple((np.array(start) + np.array(end)) / 2 + np.random.uniform(-0.1, 0.1, 3))
        spline = pv.Spline([start, midpoint, end], n_points=50)
        plotter.add_mesh(
            spline,
            color=color,
            line_width=edge_width,
            opacity=0.7
        )


    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_entries = []
    for ntype in unique_node_types:
        c = node_colors.get(ntype, 'grey')
        legend_entries.append([ntype, c])


    unique_relations = set(nx.get_edge_attributes(G, 'relation').values())
    for rel in unique_relations:
        c = edge_colors.get(rel, 'grey')
        legend_entries.append([rel, c])


    plotter.add_legend(labels=legend_entries, bcolor='white', size=(0.15, 0.25))


    plotter.add_axes()
    if show_grid:
        plotter.show_grid(color='lightgrey')


    plotter.camera_position = camera_position


    plotter.show(auto_close=False)
    if save_path:
        plotter.screenshot(save_path)
        print(f"Graph saved to {save_path}")
    plotter.close()

    print("3D Graph visualization complete.\n")

def generate_output(
    model: nn.Module,
    data: HeteroData,
    mappings: Dict[str, Dict[str, Any]],
    config: Config,
    device: torch.device,
    global_threshold: Optional[float] = None,
    min_threshold: float = 0.3,
    max_threshold: float = 0.7,
    known_positive_sets: Optional[Dict[int, set]] = None
) -> pd.DataFrame:

    print(f"\nGenerating top {config.top_k} predicted interactions for specified link prediction tasks...")
    model.eval()
    with torch.no_grad():
        print("Starting forward pass for prediction...")
        shared_embeddings = model(data.x_dict, data.edge_index_dict, data.edge_attr_dict)


        if global_threshold is None:
            global_threshold = (min_threshold + max_threshold) / 2
            try:
                checkpoint = torch.load(
                    os.path.join(config.output_dir, 'best_model.pt'),
                    map_location=device
                )
                if 'global_threshold' in checkpoint:
                    global_threshold = float(checkpoint['global_threshold'])
            except (FileNotFoundError, KeyError, RuntimeError):
                print(
                    f"No validation threshold found; using default "
                    f"{global_threshold:.4f}."
                )

        global_threshold = float(
            np.clip(global_threshold, min_threshold, max_threshold)
        )
        print(
            f"Using validation-selected global threshold: {global_threshold:.4f} "
            f"for all predictions."
        )


        predictions = []


        csmi_to_cancer_idx = {}
        if ('CSMI', 'related_to', 'Cancer') in data.edge_types:
            edge_storage = data['CSMI', 'related_to', 'Cancer']
            if hasattr(edge_storage, 'edge_index'):
                csmi_src, cancer_dst = edge_storage.edge_index
                csmi_src = csmi_src.tolist()
                cancer_dst = cancer_dst.tolist()
                for s, d in zip(csmi_src, cancer_dst):
                    if s not in csmi_to_cancer_idx:
                        csmi_to_cancer_idx[s] = set()
                    csmi_to_cancer_idx[s].add(d)
            else:
                print("Warning: 'CSMI' -> 'Cancer' edges have no 'edge_index'.")


        for task_idx, link_type in enumerate(config.link_prediction_tasks):
            print(f"\nGenerating top {config.top_k} predictions for link type {link_type}...")
            src_type, relation, dst_type = link_type


            if link_type not in data.edge_types:
                print(f"Warning: Link type {link_type} not found in data. Skipping prediction for this link type.")
                continue


            src_adapted, dst_adapted = model.get_task_adapted_embeddings(shared_embeddings, task_idx)

            num_src = src_adapted.size(0)
            num_dst = dst_adapted.size(0)


            if known_positive_sets is not None:
                existing_links = {
                    (int(src_idx), int(dst_idx))
                    for src_idx, dst_idx in known_positive_sets.get(task_idx, set())
                }
            else:
                existing_links = set(zip(
                    data[link_type].edge_index[0].tolist(),
                    data[link_type].edge_index[1].tolist()
                ))
            print(
                f"Number of known links excluded for link type {link_type}: "
                f"{len(existing_links)}"
            )


            r = model.relation_embeddings(torch.tensor(task_idx, device=device)).squeeze(0)


            batch_size = 10000
            scores = []
            indices = []

            for start in range(0, num_src * num_dst, batch_size):
                end = min(start + batch_size, num_src * num_dst)
                batch_indices = torch.arange(start, end, device=device)
                src_indices = batch_indices // num_dst
                dst_indices = batch_indices % num_dst
                src_batch = src_adapted[src_indices]
                dst_batch = dst_adapted[dst_indices]


                W_r = torch.einsum('hrd,r->hd', model.core_tensor, r)
                intermediate = torch.matmul(src_batch, W_r)
                score = torch.sum(intermediate * dst_batch, dim=1)

                probs = torch.sigmoid(score)
                scores.append(probs.cpu())
                indices.extend(batch_indices.cpu().tolist())

            scores = torch.cat(scores).numpy()
            indices = np.array(indices)


            existing_indices = [s * num_dst + d for (s, d) in existing_links]
            if existing_indices:
                scores[existing_indices] = -np.inf
                print(f"Excluded {len(existing_indices)} existing links from predictions.")


            above_threshold_indices = np.where(scores > global_threshold)[0]
            above_threshold_scores = scores[above_threshold_indices]


            sorted_indices = np.argsort(above_threshold_scores)[::-1]


            if len(sorted_indices) < config.top_k:
                top_k = len(sorted_indices)
                print(f"Found {top_k} predictions above threshold (fewer than top_k)")
            else:
                top_k = config.top_k
                print(f"Found {len(sorted_indices)} predictions above threshold, keeping top {top_k}")

            sorted_top_k = [(above_threshold_indices[sorted_indices[i]], above_threshold_scores[sorted_indices[i]])
                             for i in range(top_k)]


            for idx, score in sorted_top_k:
                src_idx = idx // num_dst
                dst_idx = idx % num_dst

                src_name = mappings[src_type]['idx_to_node'].get(src_idx, f"{src_type}_{src_idx}")
                dst_name = mappings[dst_type]['idx_to_node'].get(dst_idx, f"{dst_type}_{dst_idx}")


                associated_cancers = 'N/A'
                if dst_type == 'CSMI':
                    cancer_indices = csmi_to_cancer_idx.get(dst_idx, None)
                    if cancer_indices:
                        associated_cancers = [mappings['Cancer']['idx_to_node'].get(c_idx, 'Unknown Cancer') for c_idx in cancer_indices]
                        for cancer in associated_cancers:
                            predictions.append({
                                'Link Type': f"{link_type}",
                                'miRNA': src_name if src_type == 'miRNA' else 'N/A',
                                'circRNA': src_name if src_type == 'circRNA' else 'N/A',
                                'Cancer': cancer,
                                'CSMI': dst_name if dst_type == 'CSMI' else 'N/A',
                                'MET': dst_name if dst_type == 'MET' else 'N/A',
                                'Score': score,
                                'Above Threshold': 'Yes'
                            })
                            print(f"Predicted interaction: {src_name} -> {dst_name} with score {score:.4f} (Cancer: {cancer})")
                elif src_type == 'CSMI':
                    cancer_indices = csmi_to_cancer_idx.get(src_idx, None)
                    if cancer_indices:
                        associated_cancers = [mappings['Cancer']['idx_to_node'].get(c_idx, 'Unknown Cancer') for c_idx in cancer_indices]
                        for cancer in associated_cancers:
                            predictions.append({
                                'Link Type': f"{link_type}",
                                'miRNA': dst_name if dst_type == 'miRNA' else 'N/A',
                                'circRNA': dst_name if dst_type == 'circRNA' else 'N/A',
                                'Cancer': cancer,
                                'CSMI': src_name if src_type == 'CSMI' else 'N/A',
                                'MET': src_name if src_type == 'MET' else 'N/A',
                                'Score': score,
                                'Above Threshold': 'Yes'
                            })
                            print(f"Predicted interaction: {dst_name} -> {src_name} with score {score:.4f} (Cancer: {cancer})")
                else:

                    predictions.append({
                        'Link Type': f"{link_type}",
                        'miRNA': src_name if src_type == 'miRNA' else 'N/A',
                        'circRNA': src_name if src_type == 'circRNA' else 'N/A',
                        'Cancer': dst_name if dst_type == 'Cancer' else 'N/A',
                        'CSMI': 'N/A',
                        'MET': (
                            dst_name if dst_type == 'MET'
                            else src_name if src_type == 'MET'
                            else 'N/A'
                        ),
                        'Score': score,
                        'Above Threshold': 'Yes'
                    })
                    print(
                        f"Predicted interaction: {src_name} -> {dst_name} "
                        f"with score {score:.4f} "
                        f"(Cancer: {dst_name if dst_type == 'Cancer' else 'N/A'})"
                    )


        pred_df = pd.DataFrame(predictions)


        if not pred_df.empty:
            pred_df['Global Threshold'] = global_threshold


    pred_df.to_excel(os.path.join(config.output_dir, 'top_rna_cancer_interactions.xlsx'), index=False)
    print(f"\nTop predictions for specified link prediction tasks exported to 'top_rna_cancer_interactions.xlsx'.")
    print(f"Used global threshold: {global_threshold} for all predictions.\n")
    return pred_df

def show_graph_info(
    data: HeteroData,
    mappings: Dict[str, Dict[str, Any]],
    config: Config,
    num_samples: int = 5,
    show_feature_stats: bool = False
):

    print("\n===== Graph Information =====\n")


    print(">> Node Information:")
    node_info = []
    for node_type in data.node_types:
        num_nodes = data[node_type].num_nodes
        has_features = hasattr(data[node_type], 'x') and data[node_type].x is not None
        feature_dim = data[node_type].x.size(1) if has_features else "No features"
        node_info.append({
            'Node Type': node_type,
            'Number of Nodes': num_nodes,
            'Feature Dimension': feature_dim
        })

    node_info_df = pd.DataFrame(node_info)
    print(node_info_df.to_string(index=False))


    print("\n>> Edge Information:")
    edge_info = []
    for edge_type in data.edge_types:
        src_type, relation, dst_type = edge_type
        num_edges = data[edge_type].edge_index.size(1) if hasattr(data[edge_type], 'edge_index') else 0
        has_edge_attr = hasattr(data[edge_type], 'edge_attr') and data[edge_type].edge_attr is not None
        edge_attr_info = f"Yes, dim={data[edge_type].edge_attr.size(1)}" if has_edge_attr else "No"
        edge_info.append({
            'Edge Type': f"{src_type} -[{relation}]-> {dst_type}",
            'Number of Edges': num_edges,
            'Has Edge Attributes': edge_attr_info
        })

    edge_info_df = pd.DataFrame(edge_info)
    print(edge_info_df.to_string(index=False))


    print("\n>> Sample Nodes:")
    for node_type in data.node_types:
        print(f"\n-- {node_type} Nodes --")
        node_indices = list(range(min(num_samples, data[node_type].num_nodes)))
        if data[node_type].num_nodes == 0:
            print("No nodes available.")
            continue
        node_samples = []
        for idx in node_indices:
            node_name = mappings[node_type]['idx_to_node'].get(idx, f"{node_type}_{idx}")
            node_samples.append(node_name)
        print(node_samples)


    print("\n>> Sample Edges:")
    for edge_type in data.edge_types:
        src_type, relation, dst_type = edge_type
        print(f"\n-- {src_type} -[{relation}]-> {dst_type} Edges --")
        num_edges = data[edge_type].edge_index.size(1) if hasattr(data[edge_type], 'edge_index') else 0
        num_samples_edge = min(num_samples, num_edges)
        if num_edges == 0:
            print("No edges available.")
            continue
        edge_samples = []
        for i in range(num_samples_edge):
            src_idx = data[edge_type].edge_index[0, i].item()
            dst_idx = data[edge_type].edge_index[1, i].item()
            src_name = mappings[src_type]['idx_to_node'].get(src_idx, f"{src_type}_{src_idx}")
            dst_name = mappings[dst_type]['idx_to_node'].get(dst_idx, f"{dst_type}_{dst_idx}")
            edge_samples.append(f"{src_name} -> {dst_name}")
        print(edge_samples)


    if show_feature_stats:
        print("\n>> Feature Statistics:")
        for node_type in data.node_types:
            if hasattr(data[node_type], 'x') and data[node_type].x is not None:
                print(f"\n-- {node_type} Node Features --")

                features_np = data[node_type].x.cpu().numpy()
                stats_df = pd.DataFrame(features_np).describe()
                print(stats_df)
            else:
                print(f"\n-- {node_type} Nodes have no features to display statistics.")


    print("\n===== End of Graph Information =====\n")
