from __future__ import annotations

import os
import random
import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

from config import Config

def standardize_node_names(df: pd.DataFrame, column: str, node_type: str) -> pd.DataFrame:
    """
    Standardize node names for miRNA and circRNA to ensure consistency across datasets.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the node names.
    - column (str): The column name in the DataFrame to be standardized.
    - node_type (str): Type of node ('miRNA' or 'circRNA').

    Returns:
    - pd.DataFrame: DataFrame with standardized node names.
    """
    print(f"Standardizing node names for '{node_type}' in column '{column}'...")
    if node_type == 'miRNA':
        def clean_miRNA(x):
            if pd.isnull(x):
                return x

            x = unicodedata.normalize('NFC', str(x).strip())

            x = re.sub(r'[–—‑‒−_\s]+', '-', x)

            x = re.sub(r'^-+', '', x)

            x = re.sub(r'\.\d+$', '', x)

            x = re.sub(r'^hsa-?', '', x, flags=re.IGNORECASE)
            x = re.sub(r'^miRNA-?', 'miR-', x, flags=re.IGNORECASE)

            x = re.sub(r'[^A-Za-z0-9*-]', '', x)
            x = re.sub(r'-+', '-', x).strip('-')


            if re.match(r'^let-?\d', x, flags=re.IGNORECASE):
                body = re.sub(r'^let-?', '', x, count=1, flags=re.IGNORECASE)
                standardized = f'let-{body.lower()}'
            else:

                body = x
                while re.match(r'^miR-?', body, flags=re.IGNORECASE):
                    body = re.sub(r'^miR-?', '', body, count=1, flags=re.IGNORECASE)
                standardized = f'miR-{body.lower()}'

            return standardized
        df[column] = df[column].apply(clean_miRNA)
    elif node_type == 'circRNA':
        def clean_circRNA(x):
            if pd.isnull(x):
                return x

            x = unicodedata.normalize('NFC', x)

            x = re.sub(r'[–—‑‒−]', '-', x)

            x = re.sub(r'[_\- ]', '', x)
            standardized = x.upper()
            return standardized
        df[column] = df[column].apply(clean_circRNA)
    print(f"Completed standardizing '{node_type}' names.\n")
    return df

def identify_missing_nodes(all_nodes: List[str], available_nodes: List[str], node_type: str, output_file: str) -> List[str]:

    print(f"Identifying missing '{node_type}' nodes...")
    missing_nodes = sorted(set(all_nodes) - set(available_nodes))
    with open(output_file, 'w', encoding='utf-8') as f:
        if missing_nodes:
            f.write(f"Missing {node_type} Nodes ({len(missing_nodes)}):\n")
            for node in missing_nodes:
                f.write(f"{node}\n")
            print(f"Missing {node_type} nodes have been written to '{output_file}'.")
        else:
            f.write(f"No missing {node_type} nodes.\n")
            print(f"No missing {node_type} nodes. Log file '{output_file}' created.")
    print(f"Total missing '{node_type}' nodes: {len(missing_nodes)}\n")
    return missing_nodes

def log_feature_missingness(
    feature_df: pd.DataFrame,
    node_type: str,
    fully_missing_output_file: str,
    partially_missing_output_file: str
) -> Dict[str, Any]:

    if feature_df.shape[1] == 0:
        raise ValueError(f"No feature columns were found for '{node_type}'.")

    missing_cells = feature_df.isna()
    fully_missing_rows = missing_cells.all(axis=1)
    partially_missing_rows = missing_cells.any(axis=1) & ~fully_missing_rows

    fully_missing_nodes = feature_df.index[fully_missing_rows].tolist()
    partially_missing_nodes = feature_df.index[partially_missing_rows].tolist()

    with open(fully_missing_output_file, 'w', encoding='utf-8') as f:
        f.write(
            f"Fully missing {node_type} feature rows "
            f"({len(fully_missing_nodes)}):\n"
        )
        for node in fully_missing_nodes:
            f.write(f"{node}\n")

    with open(partially_missing_output_file, 'w', encoding='utf-8') as f:
        f.write(
            f"Partially missing {node_type} feature rows "
            f"({len(partially_missing_nodes)}):\n"
        )
        for node in partially_missing_nodes:
            missing_count = int(missing_cells.loc[node].sum())
            f.write(
                f"{node}\tmissing={missing_count}/{feature_df.shape[1]}\n"
            )

    total_missing_cells = int(missing_cells.to_numpy().sum())
    print(f"Fully missing '{node_type}' feature rows: {len(fully_missing_nodes)}")
    print(
        f"Partially missing '{node_type}' feature rows: "
        f"{len(partially_missing_nodes)}"
    )
    print(
        f"Trainable missing '{node_type}' feature cells: "
        f"{total_missing_cells}"
    )
    print(
        f"Missingness logs written to '{fully_missing_output_file}' and "
        f"'{partially_missing_output_file}'.\n"
    )

    return {
        'missing_cells': missing_cells,
        'fully_missing_rows': fully_missing_rows,
        'partially_missing_rows': partially_missing_rows,
        'fully_missing_nodes': fully_missing_nodes,
        'partially_missing_nodes': partially_missing_nodes,
        'total_missing_cells': total_missing_cells
    }

def load_and_standardize_data(config: Config) -> Dict[str, pd.DataFrame]:
    """
    Load all datasets and standardize node names.

    Parameters:
    - config (Config): Configuration object with file paths.

    Returns:
    - dataframes (dict): Dictionary containing all loaded DataFrames.
    """
    print("Loading datasets...")
    try:

        circRNA_metastasis_df = pd.read_excel(os.path.join(config.data_dir, config.circRNA_metastasis_file))
        print(f"Loaded 'circRNA_metastasis' with shape {circRNA_metastasis_df.shape}")

        cleaned_circRNA_df = pd.read_csv(os.path.join(config.data_dir, config.cleaned_circRNA_file))
        print(f"Loaded 'cleaned_circRNA' with shape {cleaned_circRNA_df.shape}")

        circbase_interactions_df = pd.read_csv(os.path.join(config.data_dir, config.circbase_interactions_file))
        print(f"Loaded 'circbase_interactions' with shape {circbase_interactions_df.shape}")

        mirna_expression_df = pd.read_csv(os.path.join(config.data_dir, config.mirna_expression_file))
        print(f"Loaded 'mirna_expression' with shape {mirna_expression_df.shape}")

        mirna_metastasis_df = pd.read_excel(os.path.join(config.data_dir, config.mirna_metastasis_file))
        print(f"Loaded 'mirna_metastasis' with shape {mirna_metastasis_df.shape}\n")
    except Exception as e:
        print(f"Error loading datasets: {e}")
        raise


    print("Standardizing node names...")
    circRNA_metastasis_df = standardize_node_names(circRNA_metastasis_df, 'CircBase ID', 'circRNA')
    circbase_interactions_df = standardize_node_names(circbase_interactions_df, 'miR_ID', 'miRNA')
    circbase_interactions_df = standardize_node_names(circbase_interactions_df, 'circbase_ID', 'circRNA')
    mirna_metastasis_df = standardize_node_names(mirna_metastasis_df, 'miRNA name', 'miRNA')
    mirna_expression_df = standardize_node_names(mirna_expression_df, 'mirna', 'miRNA')
    cleaned_circRNA_df = standardize_node_names(cleaned_circRNA_df, 'circ', 'circRNA')


    if 'mirna' in mirna_expression_df.columns:
        mirna_expression_df['mirna'] = mirna_expression_df['mirna'].str.strip()
        print("Stripped leading/trailing spaces from 'mirna' column in 'mirna_expression_df'.")
    if 'circ' in cleaned_circRNA_df.columns:
        cleaned_circRNA_df['circ'] = cleaned_circRNA_df['circ'].str.strip()
        print("Stripped leading/trailing spaces from 'circ' column in 'cleaned_circRNA_df'.")


    for df in [circRNA_metastasis_df, mirna_metastasis_df]:
        df['CSMI name'] = pd.NA
        valid_csmi = (
            df['metastatic event'].notna()
            & df['Cancer type'].notna()
        )
        df.loc[valid_csmi, 'CSMI name'] = (
            df.loc[valid_csmi, 'metastatic event'].astype(str)
            + '_'
            + df.loc[valid_csmi, 'Cancer type'].astype(str)
        )

    print(
        "Created cancer-specific CSMI names only; MET-only records without "
        "a cancer context do not create CSMI nodes.\n"
    )


    print("Data standardization complete.\n")

    dataframes = {
        'circRNA_metastasis': circRNA_metastasis_df,
        'cleaned_circRNA': cleaned_circRNA_df,
        'circbase_interactions': circbase_interactions_df,
        'mirna_expression': mirna_expression_df,
        'mirna_metastasis': mirna_metastasis_df
    }

    return dataframes

def create_unique_nodes(dataframes: Dict[str, pd.DataFrame]) -> Dict[str, List[str]]:

    print("Creating unique node lists...")

    circRNA_nodes_cirR2metasta = dataframes['circRNA_metastasis']['CircBase ID'].dropna().unique().tolist()
    circRNA_nodes_cleaned = dataframes['cleaned_circRNA']['circ'].dropna().unique().tolist()
    circRNA_nodes = list(set(circRNA_nodes_cirR2metasta).union(set(circRNA_nodes_cleaned)))
    print(f"Total unique 'circRNA' nodes: {len(circRNA_nodes)}")


    miRNA_nodes_metasta = dataframes['mirna_metastasis']['miRNA name'].dropna().unique().tolist()
    miRNA_nodes_circbase = dataframes['circbase_interactions']['miR_ID'].dropna().unique().tolist()
    miRNA_nodes_expression = dataframes['mirna_expression']['mirna'].dropna().unique().tolist()
    miRNA_nodes = list(set(miRNA_nodes_metasta).union(set(miRNA_nodes_circbase)).union(set(miRNA_nodes_expression)))
    print(f"Total unique 'miRNA' nodes: {len(miRNA_nodes)}")


    cancer_nodes_cirR2metasta = dataframes['circRNA_metastasis']['Cancer type'].dropna().unique().tolist()
    cancer_nodes_metasta = dataframes['mirna_metastasis']['Cancer type'].dropna().unique().tolist()
    cancer_nodes = list(set(cancer_nodes_cirR2metasta).union(set(cancer_nodes_metasta)))
    print(f"Total unique 'Cancer' nodes: {len(cancer_nodes)}")


    met_cirR2metasta = dataframes['circRNA_metastasis']['metastatic event'].dropna().unique().tolist()
    met_mirna_source = dataframes['mirna_metastasis']['metastatic event'].dropna().unique().tolist()
    met_nodes = list(set(met_cirR2metasta).union(set(met_mirna_source)))
    print(f"Total unique 'MET' nodes: {len(met_nodes)}")


    circ_metasta_unique = dataframes['circRNA_metastasis'].dropna(subset=['CSMI name'])
    mirna_metasta_unique = dataframes['mirna_metastasis'].dropna(subset=['CSMI name'])
    unique_csmi_cirR2metasta = circ_metasta_unique['CSMI name'].unique().tolist()
    unique_csmi_mirna_metasta = mirna_metasta_unique['CSMI name'].unique().tolist()
    unique_csmi_nodes = list(set(unique_csmi_cirR2metasta).union(set(unique_csmi_mirna_metasta)))
    print(f"Total unique 'CSMI' nodes: {len(unique_csmi_nodes)}\n")

    nodes = {
        'circRNA': circRNA_nodes,
        'miRNA': miRNA_nodes,
        'Cancer': cancer_nodes,
        'MET': met_nodes,
        'CSMI': unique_csmi_nodes
    }

    return nodes

def map_nodes_to_indices(nodes: Dict[str, List[str]]) -> Dict[str, Dict[str, Any]]:
    """
    Create mappings from node names to indices and vice versa.

    Parameters:
    - nodes (dict): Dictionary containing unique node lists for each type.

    Returns:
    - mappings (dict): Dictionary containing node_to_idx and idx_to_node mappings for each type.
    """
    print("Mapping node names to indices...")
    mappings = {}
    for node_type, node_list in nodes.items():
        node_to_idx = {name: idx for idx, name in enumerate(node_list)}
        idx_to_node = {idx: name for name, idx in node_to_idx.items()}
        mappings[node_type] = {
            'node_to_idx': node_to_idx,
            'idx_to_node': idx_to_node
        }
        print(f"Mapped {len(node_to_idx)} '{node_type}' nodes.")
    print("Node mapping complete.\n")
    return mappings

def initialize_hetero_data(nodes: Dict[str, List[str]]) -> HeteroData:
    """
    Initialize a HeteroData object with node types.

    Parameters:
    - nodes (dict): Dictionary containing unique node lists for each type.

    Returns:
    - data (HeteroData): Initialized HeteroData object.
    """
    print("Initializing HeteroData object...")
    data = HeteroData()
    for node_type, node_list in nodes.items():
        data[node_type].num_nodes = len(node_list)
        print(f"Added node type '{node_type}' with {len(node_list)} nodes.")
    print("HeteroData initialization complete.\n")
    return data

def add_node_features(
    data: HeteroData,
    nodes: Dict[str, List[str]],
    mappings: Dict[str, Dict[str, Any]],
    dataframes: Dict[str, pd.DataFrame],
    config: Config,
    device: torch.device
) -> HeteroData:
    """
    Add node features to the HeteroData object for all node types in the graph.

    This function processes each node type differently:
    - For circRNA and miRNA: Extracts real-valued features from the provided dataframes
    - For Cancer, CSMI, and MET: Initializes zero tensors of appropriate dimensions
    - Establishes relationships between CSMI and MET nodes
    - Adds edges between miRNA/circRNA and MET nodes
    - Sets up MET to Cancer relationships

    Parameters:
    ----------
    data : HeteroData
        The heterogeneous graph data structure to be populated with node features
    nodes : Dict[str, List[str]]
        Dictionary containing unique node lists for each node type
    mappings : Dict[str, Dict[str, Any]]
        Dictionary containing node_to_idx and idx_to_node mappings for each node type
    dataframes : Dict[str, pd.DataFrame]
        Dictionary containing all loaded DataFrames with source data
    config : Config
        Configuration object with model parameters and directory paths
    device : torch.device
        Device to allocate tensors on (CPU or GPU)

    Returns:
    -------
    HeteroData
        Updated HeteroData object with node features and basic relationship edges
    """
    print("Adding node features...")


    all_circRNA_nodes = nodes['circRNA']
    data['circRNA'].num_nodes = len(all_circRNA_nodes)
    print(f"Adding features for 'circRNA' nodes.")


    if 'cleaned_circRNA' in dataframes:

        circ_feature_columns = [
            col for col in dataframes['cleaned_circRNA'].columns
            if col not in ['circ', 'number']
        ]
        circ_features_df = (
            dataframes['cleaned_circRNA']
            .set_index('circ')
            .reindex(all_circRNA_nodes)[circ_feature_columns]
            .apply(pd.to_numeric, errors='coerce')
        )

        circ_missingness = log_feature_missingness(
            circ_features_df,
            'circRNA',
            os.path.join(config.missing_nodes_log, 'missing_circRNAs.txt'),
            os.path.join(
                config.missing_nodes_log, 'partially_missing_circRNAs.txt'
            )
        )

        circ_features = torch.tensor(
            circ_features_df.to_numpy(dtype=np.float32),
            dtype=torch.float,
            device=device
        )
        data['circRNA'].x = circ_features
        data['circRNA'].feature_observed_mask = torch.tensor(
            (~circ_missingness['missing_cells']).to_numpy(),
            dtype=torch.bool,
            device=device
        )
        data['circRNA'].fully_missing_feature_mask = torch.tensor(
            circ_missingness['fully_missing_rows'].to_numpy(),
            dtype=torch.bool,
            device=device
        )
        print(f"'circRNA' features tensor shape: {circ_features.shape}")
    else:
        raise KeyError(
            "The 'cleaned_circRNA' DataFrame is required to determine the "
            "13-dimensional feature schema and its missing-value mask."
        )


    all_miRNA_nodes = nodes['miRNA']
    data['miRNA'].num_nodes = len(all_miRNA_nodes)
    print(f"Adding features for 'miRNA' nodes.")


    mirna_expression_df = dataframes['mirna_expression']
    if mirna_expression_df.duplicated(subset='mirna').any():
        print("Duplicate miRNA entries found in 'mirna_expression_df'. Aggregating by mean.")
        mirna_expression_df = mirna_expression_df.groupby('mirna').mean().reset_index()


    miRNA_feature_columns = [
        col for col in mirna_expression_df.columns if col != 'mirna'
    ]
    miRNA_features_df = (
        mirna_expression_df
        .set_index('mirna')
        .reindex(all_miRNA_nodes)[miRNA_feature_columns]
        .apply(pd.to_numeric, errors='coerce')
    )

    miRNA_missingness = log_feature_missingness(
        miRNA_features_df,
        'miRNA',
        os.path.join(config.missing_nodes_log, 'missing_miRNAs.txt'),
        os.path.join(config.missing_nodes_log, 'partially_missing_miRNAs.txt')
    )

    miRNA_features = torch.tensor(
        miRNA_features_df.to_numpy(dtype=np.float32),
        dtype=torch.float,
        device=device
    )
    data['miRNA'].x = miRNA_features
    data['miRNA'].feature_observed_mask = torch.tensor(
        (~miRNA_missingness['missing_cells']).to_numpy(),
        dtype=torch.bool,
        device=device
    )
    data['miRNA'].fully_missing_feature_mask = torch.tensor(
        miRNA_missingness['fully_missing_rows'].to_numpy(),
        dtype=torch.bool,
        device=device
    )
    print(f"'miRNA' features tensor shape: {miRNA_features.shape}")


    print(f"Initializing embeddings for 'Cancer' and 'CSMI' nodes with dimension {config.hidden_channels}.")
    data['Cancer'].x = torch.zeros((data['Cancer'].num_nodes, config.hidden_channels), dtype=torch.float, device=device)
    data['CSMI'].x = torch.zeros((data['CSMI'].num_nodes, config.hidden_channels), dtype=torch.float, device=device)
    print(f"'Cancer' and 'CSMI' node embeddings initialized with zeros.")


    print(f"Initializing embeddings for 'MET' nodes with dimension {config.hidden_channels}.")
    data['MET'].x = torch.zeros((data['MET'].num_nodes, config.hidden_channels), dtype=torch.float, device=device)
    print(f"'MET' node embeddings initialized with zeros.\n")


    print("Contextualizing CSMI nodes with MET nodes...")

    csmi_names = nodes['CSMI']
    met_mapping = mappings['MET']['node_to_idx']

    csmi_to_met_src = []
    csmi_to_met_dst = []

    for csmi_idx, csmi_name in enumerate(csmi_names):

        if '_' in csmi_name:
            met_name = '_'.join(csmi_name.split('_')[:-1])
        else:
            met_name = csmi_name
        if met_name in met_mapping:
            met_idx = met_mapping[met_name]
            csmi_to_met_src.append(csmi_idx)
            csmi_to_met_dst.append(met_idx)
        else:
            print(f"Warning: MET '{met_name}' not found in mappings.")

    if csmi_to_met_src and csmi_to_met_dst:
        edge_index = torch.tensor([csmi_to_met_src, csmi_to_met_dst], dtype=torch.long)
        edge_attr = torch.ones((edge_index.size(1), 1), dtype=torch.float, device=device)
        data['CSMI', 'represents', 'MET'].edge_index = edge_index
        data['CSMI', 'represents', 'MET'].edge_attr = edge_attr
        print(f"Added {len(csmi_to_met_src)} 'CSMI' -> 'MET' edges.")
    else:
        print("No 'CSMI' -> 'MET' edges added.")

    print("CSMI nodes contextualization complete.\n")


    print("Processing 'miRNA' -> 'MET' and 'circRNA' -> 'MET' edges...")


    mirna_metasta_df = dataframes['mirna_metastasis']
    mirna_met_edges_src = []
    mirna_met_edges_dst = []

    mirna_met_pairs = (
        mirna_metasta_df[['miRNA name', 'metastatic event']]
        .dropna(subset=['miRNA name', 'metastatic event'])
        .drop_duplicates(subset=['miRNA name', 'metastatic event'])
        .reset_index(drop=True)
    )

    for _, row in mirna_met_pairs.iterrows():
        met_name = row['metastatic event']
        mirna_name = row['miRNA name']

        if mirna_name in mappings['miRNA']['node_to_idx'] and met_name in mappings['MET']['node_to_idx']:
            mirna_idx = mappings['miRNA']['node_to_idx'][mirna_name]
            met_idx = mappings['MET']['node_to_idx'][met_name]
            mirna_met_edges_src.append(mirna_idx)
            mirna_met_edges_dst.append(met_idx)
        else:
            print(f"Warning: Cannot map 'miRNA' '{mirna_name}' or 'MET' '{met_name}' to indices.")

    if mirna_met_edges_src and mirna_met_edges_dst:
        edge_index = torch.tensor([mirna_met_edges_src, mirna_met_edges_dst], dtype=torch.long)
        edge_attr = torch.ones((edge_index.size(1), 1), dtype=torch.float, device=device)
        data['miRNA', 'regulates', 'MET'].edge_index = edge_index
        data['miRNA', 'regulates', 'MET'].edge_attr = edge_attr
        print(f"Added {len(mirna_met_edges_src)} 'miRNA' -> 'MET' edges.")
    else:
        print("No 'miRNA' -> 'MET' edges added.")


    circRNA_metasta_df = dataframes['circRNA_metastasis']
    circRNA_met_edges_src = []
    circRNA_met_edges_dst = []

    circRNA_met_pairs = (
        circRNA_metasta_df[['CircBase ID', 'metastatic event']]
        .dropna(subset=['CircBase ID', 'metastatic event'])
        .drop_duplicates(subset=['CircBase ID', 'metastatic event'])
        .reset_index(drop=True)
    )

    for _, row in circRNA_met_pairs.iterrows():
        met_name = row['metastatic event']
        circRNA_name = row['CircBase ID']

        if circRNA_name in mappings['circRNA']['node_to_idx'] and met_name in mappings['MET']['node_to_idx']:
            circRNA_idx = mappings['circRNA']['node_to_idx'][circRNA_name]
            met_idx = mappings['MET']['node_to_idx'][met_name]
            circRNA_met_edges_src.append(circRNA_idx)
            circRNA_met_edges_dst.append(met_idx)
        else:
            print(f"Warning: Cannot map 'circRNA' '{circRNA_name}' or 'MET' '{met_name}' to indices.")

    if circRNA_met_edges_src and circRNA_met_edges_dst:
        edge_index = torch.tensor([circRNA_met_edges_src, circRNA_met_edges_dst], dtype=torch.long)
        edge_attr = torch.ones((edge_index.size(1), 1), dtype=torch.float, device=device)
        data['circRNA', 'regulates', 'MET'].edge_index = edge_index
        data['circRNA', 'regulates', 'MET'].edge_attr = edge_attr
        print(f"Added {len(circRNA_met_edges_src)} 'circRNA' -> 'MET' edges.")
    else:
        print("No 'circRNA' -> 'MET' edges added.")

    print("Added 'miRNA' -> 'MET' and 'circRNA' -> 'MET' edges successfully.\n")


    print("Processing 'MET' -> 'Cancer' edges...")


    met_related_to_cancer_df = (
        pd.concat(
            [
                dataframes["mirna_metastasis"][["metastatic event", "Cancer type"]],
                dataframes["circRNA_metastasis"][["metastatic event", "Cancer type"]],
            ],
            ignore_index=True,
        )
        .dropna(subset=["metastatic event", "Cancer type"])
        .drop_duplicates()
    )

    met_related_to_cancer_src = []
    met_related_to_cancer_dst = []

    for _, row in met_related_to_cancer_df.iterrows():
        met_name = row['metastatic event']
        cancer_type = row['Cancer type']

        if met_name in mappings['MET']['node_to_idx'] and cancer_type in mappings['Cancer']['node_to_idx']:
            met_idx = mappings['MET']['node_to_idx'][met_name]
            cancer_idx = mappings['Cancer']['node_to_idx'][cancer_type]
            met_related_to_cancer_src.append(met_idx)
            met_related_to_cancer_dst.append(cancer_idx)
        else:
            print(f"Warning: Cannot map 'MET' '{met_name}' or 'Cancer' '{cancer_type}' to indices.")

    if met_related_to_cancer_src and met_related_to_cancer_dst:
        edge_index = torch.tensor([met_related_to_cancer_src, met_related_to_cancer_dst], dtype=torch.long)
        edge_attr = torch.ones((edge_index.size(1), 1), dtype=torch.float, device=device)
        data['MET', 'related_to', 'Cancer'].edge_index = edge_index
        data['MET', 'related_to', 'Cancer'].edge_attr = edge_attr
        print(f"Added {len(met_related_to_cancer_src)} 'MET' -> 'Cancer' edges.")
    else:
        print("No 'MET' -> 'Cancer' edges added.")

    print("Added 'MET' -> 'Cancer' edges successfully.\n")

    print("Node features added successfully.\n")
    return data

def filter_edges(
    df: pd.DataFrame,
    src_column: str,
    dst_column: str,
    src_mapping: Dict[str, int],
    dst_mapping: Dict[str, int],
    edge_attr_column: str = None,
    edge_attr_mapping: Dict[str, float] = None
) -> Tuple[List[int], List[int], List[float]]:
    """
    Filter edges to include only those where both source and destination nodes are present in the mappings.

    Parameters:
    - df (pd.DataFrame): DataFrame containing edge information.
    - src_column (str): Column name for source nodes.
    - dst_column (str): Column name for destination nodes.
    - src_mapping (dict): Mapping from source node names to indices.
    - dst_mapping (dict): Mapping from destination node names to indices.
    - edge_attr_column (str): Column name for edge attributes (optional).
    - edge_attr_mapping (dict): Mapping for edge attributes (optional).

    Returns:
    - src_indices (list): List of source node indices.
    - dst_indices (list): List of destination node indices.
    - edge_attrs (list): List of edge attributes (if any).
    """

    df_filtered = (
        df[
            df[src_column].isin(src_mapping)
            & df[dst_column].isin(dst_mapping)
        ]
        .copy()
        .drop_duplicates(subset=[src_column, dst_column])
        .reset_index(drop=True)
    )

    src_indices = [
        src_mapping[name] for name in df_filtered[src_column]
    ]
    dst_indices = [
        dst_mapping[name] for name in df_filtered[dst_column]
    ]

    if edge_attr_column and edge_attr_mapping:
        edge_attrs = df_filtered[edge_attr_column].map(edge_attr_mapping).fillna(0.0).tolist()
    else:
        edge_attrs = [0.0] * len(df_filtered)
    print(f"Filtered edges from '{src_column}' to '{dst_column}': {len(src_indices)} edges found.")
    return src_indices, dst_indices, edge_attrs

def add_edges(
    data: HeteroData,
    mappings: Dict[str, Dict[str, Any]],
    dataframes: Dict[str, pd.DataFrame],
    config: Config
) -> HeteroData:
    """
    Add all edge types to the HeteroData object with appropriate attributes.

    Parameters:
    - data (HeteroData): The heterogeneous graph data.
    - mappings (dict): Dictionary containing node_to_idx and idx_to_node mappings for each type.
    - dataframes (dict): Dictionary containing all loaded DataFrames.
    - config (Config): Configuration object with parameters.

    Returns:
    - data (HeteroData): Updated HeteroData object with edges.
    """
    print("Adding edges to HeteroData...")

    edge_types = config.all_link_types


    regulation_mapping = {'Up-regulated': 1.0, 'Down-regulated': -1.0}


    csmi_to_cancer = {}


    print("Processing 'circRNA' -> 'Cancer' edges...")
    circ_cancer_src, circ_cancer_dst, circ_cancer_attr = filter_edges(
        df=dataframes['circRNA_metastasis'],
        src_column='CircBase ID',
        dst_column='Cancer type',
        src_mapping=mappings['circRNA']['node_to_idx'],
        dst_mapping=mappings['Cancer']['node_to_idx'],
        edge_attr_column='expression pattern',
        edge_attr_mapping=regulation_mapping
    )

    if circ_cancer_src:
        edge_index = torch.tensor([circ_cancer_src, circ_cancer_dst], dtype=torch.long)
        edge_attr = torch.tensor(circ_cancer_attr, dtype=torch.float).unsqueeze(1)
        data['circRNA', 'regulates', 'Cancer'].edge_index = edge_index
        data['circRNA', 'regulates', 'Cancer'].edge_attr = edge_attr
        print(f"Added {len(circ_cancer_src)} 'circRNA' -> 'Cancer' edges.")
    else:
        print("No 'circRNA' -> 'Cancer' edges found.")


    print("Processing 'circRNA' -> 'CSMI' edges...")
    circ_csmi_src, circ_csmi_dst, circ_csmi_attr = filter_edges(
        df=dataframes['circRNA_metastasis'],
        src_column='CircBase ID',
        dst_column='CSMI name',
        src_mapping=mappings['circRNA']['node_to_idx'],
        dst_mapping=mappings['CSMI']['node_to_idx'],
        edge_attr_column='expression pattern',
        edge_attr_mapping=regulation_mapping
    )

    if circ_csmi_src:
        edge_index = torch.tensor([circ_csmi_src, circ_csmi_dst], dtype=torch.long)
        edge_attr = torch.tensor(circ_csmi_attr, dtype=torch.float).unsqueeze(1)
        data['circRNA', 'regulates', 'CSMI'].edge_index = edge_index
        data['circRNA', 'regulates', 'CSMI'].edge_attr = edge_attr
        print(f"Added {len(circ_csmi_src)} 'circRNA' -> 'CSMI' edges.")
    else:
        print("No 'circRNA' -> 'CSMI' edges found.")


    print("Processing 'miRNA' -> 'Cancer' edges...")
    mirna_cancer_src, mirna_cancer_dst, mirna_cancer_attr = filter_edges(
        df=dataframes['mirna_metastasis'],
        src_column='miRNA name',
        dst_column='Cancer type',
        src_mapping=mappings['miRNA']['node_to_idx'],
        dst_mapping=mappings['Cancer']['node_to_idx'],
        edge_attr_column='expression pattern',
        edge_attr_mapping=regulation_mapping
    )

    if mirna_cancer_src:
        edge_index = torch.tensor([mirna_cancer_src, mirna_cancer_dst], dtype=torch.long)
        edge_attr = torch.tensor(mirna_cancer_attr, dtype=torch.float).unsqueeze(1)
        data['miRNA', 'regulates', 'Cancer'].edge_index = edge_index
        data['miRNA', 'regulates', 'Cancer'].edge_attr = edge_attr
        print(f"Added {len(mirna_cancer_src)} 'miRNA' -> 'Cancer' edges.")
    else:
        print("No 'miRNA' -> 'Cancer' edges found.")


    print("Processing 'miRNA' -> 'CSMI' edges...")
    mirna_csmi_src, mirna_csmi_dst, mirna_csmi_attr = filter_edges(
        df=dataframes['mirna_metastasis'],
        src_column='miRNA name',
        dst_column='CSMI name',
        src_mapping=mappings['miRNA']['node_to_idx'],
        dst_mapping=mappings['CSMI']['node_to_idx'],
        edge_attr_column='expression pattern',
        edge_attr_mapping=regulation_mapping
    )

    if mirna_csmi_src:
        edge_index = torch.tensor([mirna_csmi_src, mirna_csmi_dst], dtype=torch.long)
        edge_attr = torch.tensor(mirna_csmi_attr, dtype=torch.float).unsqueeze(1)
        data['miRNA', 'regulates', 'CSMI'].edge_index = edge_index
        data['miRNA', 'regulates', 'CSMI'].edge_attr = edge_attr
        print(f"Added {len(mirna_csmi_src)} 'miRNA' -> 'CSMI' edges.")
    else:
        print("No 'miRNA' -> 'CSMI' edges found.")


    print("Processing 'circRNA' -> 'miRNA' regulatory edges...")
    circ_mirna_df = dataframes['circbase_interactions']
    circ_mirna_src, circ_mirna_dst, _ = filter_edges(
        df=circ_mirna_df,
        src_column='circbase_ID',
        dst_column='miR_ID',
        src_mapping=mappings['circRNA']['node_to_idx'],
        dst_mapping=mappings['miRNA']['node_to_idx']
    )

    if circ_mirna_src:
        edge_index = torch.tensor([circ_mirna_src, circ_mirna_dst], dtype=torch.long)

        edge_attr = torch.ones(
            (edge_index.size(1), 1),
            dtype=torch.float
        )
        data['circRNA', 'regulates', 'miRNA'].edge_index = edge_index
        data['circRNA', 'regulates', 'miRNA'].edge_attr = edge_attr
        print(f"Added {len(circ_mirna_src)} 'circRNA' -> 'miRNA' regulatory edges.")
    else:
        print("No 'circRNA' -> 'miRNA' regulatory edges found.")


    print("Processing 'CSMI' -> 'Cancer' edges...")

    csmi_cancer_df = (
        pd.concat(
            [
            dataframes['circRNA_metastasis'][
                ['metastatic event', 'Cancer type', 'CSMI name']
                ],
                dataframes['mirna_metastasis'][
                    ['metastatic event', 'Cancer type', 'CSMI name']
                ],
            ],
            ignore_index=True,
        )
        .dropna(subset=['CSMI name', 'Cancer type'])
        .drop_duplicates(subset=['CSMI name', 'Cancer type'])
        .reset_index(drop=True)
    )


    if 'CSMI name' not in csmi_cancer_df.columns:
        csmi_cancer_df['CSMI name'] = csmi_cancer_df['metastatic event'].astype(str) + '_' + csmi_cancer_df['Cancer type'].astype(str)

    csmi_cancer_src, csmi_cancer_dst, _ = filter_edges(
        df=csmi_cancer_df,
        src_column='CSMI name',
        dst_column='Cancer type',
        src_mapping=mappings['CSMI']['node_to_idx'],
        dst_mapping=mappings['Cancer']['node_to_idx']
    )

    if csmi_cancer_src:
        edge_index = torch.tensor([csmi_cancer_src, csmi_cancer_dst], dtype=torch.long)

        edge_attr = torch.ones(
            (edge_index.size(1), 1),
            dtype=torch.float
        )
        data['CSMI', 'related_to', 'Cancer'].edge_index = edge_index
        data['CSMI', 'related_to', 'Cancer'].edge_attr = edge_attr
        print(f"Added {len(csmi_cancer_src)} 'CSMI' -> 'Cancer' edges.")


        for s, d in zip(csmi_cancer_src, csmi_cancer_dst):
            if s not in csmi_to_cancer:
                csmi_to_cancer[s] = set()
            csmi_to_cancer[s].add(d)
    else:
        print("No 'CSMI' -> 'Cancer' edges found.")


    print("Processing 'CSMI' -> 'MET' edges...")

    if ('CSMI', 'represents', 'MET') in data.edge_types:
        edge_storage = data['CSMI', 'represents', 'MET']
        if hasattr(edge_storage, 'edge_index'):
            num_represent_edges = edge_storage.edge_index.size(1)
            print(f"Added {num_represent_edges} 'CSMI' -> 'MET' edges.")
        else:
            print(f"No 'edge_index' found for edge type ('CSMI', 'represents', 'MET').")
    else:
        print("No 'CSMI' -> 'MET' edges found.")


    print("Processing 'MET' -> 'Cancer' edges...")

    if ('MET', 'related_to', 'Cancer') in data.edge_types:
        edge_storage = data['MET', 'related_to', 'Cancer']
        if hasattr(edge_storage, 'edge_index'):
            num_related_edges = edge_storage.edge_index.size(1)
            print(f"Added {num_related_edges} 'MET' -> 'Cancer' edges.")
        else:
            print(f"No 'edge_index' found for edge type ('MET', 'related_to', 'Cancer').")
    else:
        print("No 'MET' -> 'Cancer' edges found.")


    print("MET -> miRNA and circRNA edges have been handled in earlier steps.\n")

    print("All edges added successfully.\n")

    data.csmi_to_cancer = csmi_to_cancer
    return data

def validate_edges(data: HeteroData):
    """
    Validate that all edge indices are within the range of node counts.

    Parameters:
    - data (HeteroData): The heterogeneous graph data.
    """
    print("Validating edge indices...")
    for edge_type in data.edge_types:
        src_type, relation, dst_type = edge_type
        num_src = data[src_type].num_nodes
        num_dst = data[dst_type].num_nodes
        edge_storage = data[edge_type]
        if not hasattr(edge_storage, 'edge_index'):
            print(f"Warning: Edge type {edge_type} has no 'edge_index'. Skipping validation.")
            continue
        edge_index = edge_storage.edge_index
        src_ids = edge_index[0].tolist()
        dst_ids = edge_index[1].tolist()
        for i, (s, d) in enumerate(zip(src_ids, dst_ids)):
            if s >= num_src or d >= num_dst:
                print(f"Error: In edge type {edge_type}, source node ID {s} exceeds number of nodes {num_src} or destination node ID {d} exceeds number of nodes {num_dst}")
                raise ValueError(f"Edge index out of range in edge type {edge_type} at position {i}")
        print(f"Edge type {edge_type} validated successfully with {len(src_ids)} edges.")
    print("Edge validation complete.\n")

def extract_positive_link_prediction_data(
    data: HeteroData,
    link_prediction_tasks: List[Tuple[str, str, str]]
) -> Tuple[List[Tuple[int, int]], List[int], List[str], Dict[int, set]]:
    """
    Extract only observed positive target edges. Splitting is performed on these
    positives before any negative examples are generated.
    """
    positive_edges: List[Tuple[int, int]] = []
    positive_tasks: List[int] = []
    groups: List[str] = []
    all_positive_sets: Dict[int, set] = {}

    for task_idx, link_type in enumerate(link_prediction_tasks):
        if link_type not in data.edge_types or not hasattr(data[link_type], 'edge_index'):
            print(f"Warning: positive edge type {link_type} is unavailable.")
            all_positive_sets[task_idx] = set()
            continue

        src_type = link_type[0]
        edge_index = data[link_type].edge_index
        task_pairs = sorted(set(zip(
            edge_index[0].detach().cpu().tolist(),
            edge_index[1].detach().cpu().tolist()
        )))
        all_positive_sets[task_idx] = {
            (int(src), int(dst)) for src, dst in task_pairs
        }

        for src_idx, dst_idx in task_pairs:
            positive_edges.append((int(src_idx), int(dst_idx)))
            positive_tasks.append(task_idx)


            groups.append(f"{src_type}:{int(src_idx)}")

        print(f"Positive samples for {link_type}: {len(task_pairs)}")

    return positive_edges, positive_tasks, groups, all_positive_sets

def grouped_positive_train_test_split(
    edges: List[Tuple[int, int]],
    tasks: List[int],
    groups: List[str],
    test_size: float,
    random_state: int,
    split_name: str
) -> Tuple[
    List[Tuple[int, int]], List[Tuple[int, int]],
    List[int], List[int], List[str], List[str]
]:
    """
    RNA-group-aware holdout split. Candidate GroupShuffleSplit partitions are
    scored for task balance while guaranteeing that each biological RNA occurs on
    one side only across all prediction tasks.
    """
    edges_array = np.asarray(edges, dtype=int)
    tasks_array = np.asarray(tasks, dtype=int)
    groups_array = np.asarray(groups, dtype=object)

    if len(edges_array) == 0:
        raise ValueError(f"Cannot create {split_name}: no positive edges were supplied.")
    if len(np.unique(groups_array)) < 2:
        raise ValueError(f"Cannot create {split_name}: fewer than two biological RNA groups exist.")

    splitter = GroupShuffleSplit(
        n_splits=64,
        test_size=test_size,
        random_state=random_state
    )
    num_tasks = int(tasks_array.max()) + 1
    overall_distribution = np.bincount(
        tasks_array, minlength=num_tasks
    ) / len(tasks_array)

    best_indices = None
    best_score = np.inf
    dummy_x = np.zeros((len(edges_array), 1), dtype=float)

    for train_idx, test_idx in splitter.split(dummy_x, tasks_array, groups_array):
        train_task_set = set(tasks_array[train_idx].tolist())
        test_task_set = set(tasks_array[test_idx].tolist())
        missing_tasks = (
            len(set(range(num_tasks)) - train_task_set)
            + len(set(range(num_tasks)) - test_task_set)
        )

        test_distribution = np.bincount(
            tasks_array[test_idx], minlength=num_tasks
        ) / max(len(test_idx), 1)
        size_error = abs((len(test_idx) / len(edges_array)) - test_size)
        distribution_error = np.abs(
            test_distribution - overall_distribution
        ).mean()
        score = 10.0 * missing_tasks + size_error + distribution_error

        if score < best_score:
            best_score = score
            best_indices = (train_idx, test_idx)

    if best_indices is None:
        raise RuntimeError(f"Unable to create group-aware split: {split_name}.")

    train_idx, test_idx = best_indices
    train_groups_set = set(groups_array[train_idx].tolist())
    test_groups_set = set(groups_array[test_idx].tolist())
    overlap = train_groups_set.intersection(test_groups_set)
    if overlap:
        raise RuntimeError(
            f"Cross-task RNA leakage detected in {split_name}: "
            f"{len(overlap)} overlapping biological RNA groups."
        )

    print(
        f"{split_name}: {len(train_idx)} train positives, "
        f"{len(test_idx)} holdout positives, "
        f"{len(train_groups_set)} train RNA groups, "
        f"{len(test_groups_set)} holdout RNA groups."
    )

    return (
        edges_array[train_idx].tolist(),
        edges_array[test_idx].tolist(),
        tasks_array[train_idx].tolist(),
        tasks_array[test_idx].tolist(),
        groups_array[train_idx].tolist(),
        groups_array[test_idx].tolist()
    )

def split_disjoint_message_and_supervision_edges(
    positive_edges: List[Tuple[int, int]],
    positive_tasks: List[int],
    supervision_ratio: float,
    random_state: int,
    split_name: str
) -> Tuple[
    List[Tuple[int, int]], List[Tuple[int, int]],
    List[int], List[int]
]:

    if not 0.0 < supervision_ratio < 1.0:
        raise ValueError(
            f"{split_name}: supervision_ratio must be between 0 and 1, "
            f"received {supervision_ratio}."
        )

    edges_array = np.asarray(positive_edges, dtype=int)
    tasks_array = np.asarray(positive_tasks, dtype=int)

    if len(edges_array) != len(tasks_array):
        raise ValueError(
            f"{split_name}: positive_edges and positive_tasks have "
            "different lengths."
        )
    if len(edges_array) == 0:
        raise ValueError(f"{split_name}: no fitting positives were supplied.")

    task_counts = np.bincount(tasks_array)
    present_task_counts = task_counts[task_counts > 0]
    if len(present_task_counts) == 0 or int(present_task_counts.min()) < 2:
        raise ValueError(
            f"{split_name}: every represented task needs at least two "
            "positive edges for a disjoint split."
        )

    (
        message_edges,
        supervision_edges,
        message_tasks,
        supervision_tasks
    ) = train_test_split(
        edges_array,
        tasks_array,
        test_size=supervision_ratio,
        random_state=random_state,
        stratify=tasks_array,
        shuffle=True
    )

    message_keys = {
        (int(task), int(edge[0]), int(edge[1]))
        for edge, task in zip(message_edges, message_tasks)
    }
    supervision_keys = {
        (int(task), int(edge[0]), int(edge[1]))
        for edge, task in zip(supervision_edges, supervision_tasks)
    }
    overlap = message_keys.intersection(supervision_keys)
    if overlap:
        raise RuntimeError(
            f"{split_name}: {len(overlap)} fitting positives were assigned "
            "to both message passing and supervision."
        )

    print(
        f"{split_name}: {len(message_edges)} message-passing positives and "
        f"{len(supervision_edges)} disjoint supervision positives "
        f"({supervision_ratio:.0%} supervision)."
    )
    for task_idx in sorted(set(tasks_array.tolist())):
        message_count = int(np.sum(message_tasks == task_idx))
        supervision_count = int(np.sum(supervision_tasks == task_idx))
        print(
            f"  Task {task_idx}: {message_count} graph positives, "
            f"{supervision_count} supervised positives."
        )

    return (
        message_edges.tolist(),
        supervision_edges.tolist(),
        message_tasks.tolist(),
        supervision_tasks.tolist()
    )

def assert_supervision_edges_absent_from_graph(
    directed_graph: HeteroData,
    link_prediction_tasks: List[Tuple[str, str, str]],
    supervision_edges: List[Tuple[int, int]],
    supervision_tasks: List[int],
    check_name: str
) -> None:
    """Fail immediately if a positive loss label is present in encoder edges."""
    graph_pairs_by_task: Dict[int, set] = {}
    for task_idx, edge_type in enumerate(link_prediction_tasks):
        edge_store = directed_graph[edge_type]
        if not hasattr(edge_store, 'edge_index'):
            graph_pairs_by_task[task_idx] = set()
            continue
        graph_pairs_by_task[task_idx] = {
            (int(src), int(dst))
            for src, dst in zip(
                edge_store.edge_index[0].detach().cpu().tolist(),
                edge_store.edge_index[1].detach().cpu().tolist()
            )
        }

    leaked = []
    for edge, task_idx in zip(supervision_edges, supervision_tasks):
        pair = (int(edge[0]), int(edge[1]))
        if pair in graph_pairs_by_task.get(int(task_idx), set()):
            leaked.append((int(task_idx), pair))

    if leaked:
        raise RuntimeError(
            f"{check_name}: {len(leaked)} supervised positive edges are "
            "still present in the directed message-passing graph."
        )

    print(
        f"{check_name}: verified that all "
        f"{len(supervision_edges)} supervised positives are absent from "
        "the encoder graph."
    )

def build_message_passing_graph(
    data: HeteroData,
    link_prediction_tasks: List[Tuple[str, str, str]],
    allowed_positive_edges: List[Tuple[int, int]],
    allowed_positive_tasks: List[int]
) -> HeteroData:

    graph = HeteroData()

    for node_type in data.node_types:
        graph[node_type].num_nodes = data[node_type].num_nodes
        if hasattr(data[node_type], 'x') and data[node_type].x is not None:
            graph[node_type].x = data[node_type].x.clone()

    allowed_by_task = {
        task_idx: set() for task_idx in range(len(link_prediction_tasks))
    }
    for edge, task_idx in zip(allowed_positive_edges, allowed_positive_tasks):
        allowed_by_task[int(task_idx)].add((int(edge[0]), int(edge[1])))

    task_lookup = {
        edge_type: task_idx
        for task_idx, edge_type in enumerate(link_prediction_tasks)
    }

    for edge_type in data.edge_types:
        edge_store = data[edge_type]
        if not hasattr(edge_store, 'edge_index'):
            continue

        original_edge_index = edge_store.edge_index
        if edge_type in task_lookup:
            allowed_pairs = allowed_by_task[task_lookup[edge_type]]
            original_pairs = zip(
                original_edge_index[0].detach().cpu().tolist(),
                original_edge_index[1].detach().cpu().tolist()
            )
            mask_values = [
                (int(src), int(dst)) in allowed_pairs
                for src, dst in original_pairs
            ]
            mask = torch.tensor(
                mask_values,
                dtype=torch.bool,
                device=original_edge_index.device
            )
            graph[edge_type].edge_index = original_edge_index[:, mask].clone()
            if hasattr(edge_store, 'edge_attr') and edge_store.edge_attr is not None:
                graph[edge_type].edge_attr = edge_store.edge_attr[mask].clone()
        else:
            graph[edge_type].edge_index = original_edge_index.clone()
            if hasattr(edge_store, 'edge_attr') and edge_store.edge_attr is not None:
                graph[edge_type].edge_attr = edge_store.edge_attr.clone()

    if hasattr(data, 'csmi_to_cancer'):
        graph.csmi_to_cancer = data.csmi_to_cancer

    return graph

def make_undirected_message_graph(directed_graph: HeteroData) -> HeteroData:
    """
    Add automatic reverse edge types after target-edge filtering. Applying
    ToUndirected here prevents held-out positives from re-entering through a
    reverse relation.
    """
    graph = ToUndirected()(directed_graph.clone())


    for edge_type in graph.edge_types:
        edge_store = graph[edge_type]
        if not hasattr(edge_store, 'edge_index'):
            continue
        if not hasattr(edge_store, 'edge_attr') or edge_store.edge_attr is None:
            edge_store.edge_attr = torch.ones(
                (edge_store.edge_index.size(1), 1),
                dtype=torch.float,
                device=edge_store.edge_index.device
            )

    return graph

def prepare_link_prediction_data(
    topology_graph: HeteroData,
    link_prediction_tasks: List[Tuple[str, str, str]],
    positive_edges: List[Tuple[int, int]],
    positive_tasks: List[int],
    all_positive_sets: Dict[int, set],
    min_common_neighbors: int = 1,
    negative_ratio: float = 1.0,
    random_state: int = 42,
    reserved_negatives: Optional[Dict[int, set]] = None
) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """
    Generate post-split negatives using fixed 30/30/30/10 targets:

    - 30% exact 5-hop negatives
    - 30% exact 4-hop negatives
    - 30% exact 3-hop negatives
    - 10% random absent-pair negatives

    The random bucket contains every otherwise eligible absent pair that is not
    assigned to an exact 5-, 4-, or 3-hop bucket.

    Integer quotas are constructed so they sum exactly to the requested number
    of negatives. If a bucket cannot meet its quota, its shortfall is carried
    forward to the next bucket in this order:

        5-hop -> 4-hop -> 3-hop -> random

    Thus, a 5-hop shortage is added to the 4-hop target, a 4-hop shortage is
    added to the 3-hop target, and a 3-hop shortage is added to the random
    target. If the random bucket is still short, a final recovery pass uses any
    remaining eligible candidates so the requested negative count is preserved
    whenever enough absent pairs exist.
    """
    print("\nPreparing post-split 30/30/30/10 negative samples...")
    rng = random.Random(random_state)

    topology_hop_order = (5, 4, 3)
    sampling_order: Tuple[Any, ...] = topology_hop_order + ("random",)
    max_topology_hops = max(topology_hop_order)

    if reserved_negatives is None:
        reserved_negatives = {
            task_idx: set() for task_idx in range(len(link_prediction_tasks))
        }


    typed_adjacency: Dict[Tuple[str, int], set] = {}
    for edge_type in topology_graph.edge_types:
        edge_store = topology_graph[edge_type]
        if not hasattr(edge_store, 'edge_index'):
            continue

        edge_src_type, _, edge_dst_type = edge_type
        edge_src = edge_store.edge_index[0].detach().cpu().tolist()
        edge_dst = edge_store.edge_index[1].detach().cpu().tolist()

        for src_idx, dst_idx in zip(edge_src, edge_dst):
            src_node = (edge_src_type, int(src_idx))
            dst_node = (edge_dst_type, int(dst_idx))
            typed_adjacency.setdefault(src_node, set()).add(dst_node)
            typed_adjacency.setdefault(dst_node, set()).add(src_node)

    topology_destination_cache: Dict[
        Tuple[str, int, str], Dict[int, int]
    ] = {}

    def get_topology_destinations(
        src_type: str,
        src_idx: int,
        dst_type: str
    ) -> Dict[int, int]:
        cache_key = (src_type, int(src_idx), dst_type)
        if cache_key in topology_destination_cache:
            return topology_destination_cache[cache_key]

        start_node = (src_type, int(src_idx))
        distances = {start_node: 0}
        queue = [start_node]
        queue_position = 0

        while queue_position < len(queue):
            current = queue[queue_position]
            queue_position += 1
            current_distance = distances[current]

            if current_distance >= max_topology_hops:
                continue

            for neighbor in typed_adjacency.get(current, set()):
                if neighbor not in distances:
                    distances[neighbor] = current_distance + 1
                    queue.append(neighbor)

        destinations = {
            int(node_idx): int(distance)
            for (node_type, node_idx), distance in distances.items()
            if node_type == dst_type
        }
        topology_destination_cache[cache_key] = destinations
        return destinations

    combined_edges: List[Tuple[int, int]] = []
    combined_labels: List[int] = []
    combined_tasks: List[int] = []

    for task_idx, link_type in enumerate(link_prediction_tasks):
        task_positive_edges = sorted({
            (int(edge[0]), int(edge[1]))
            for edge, sample_task in zip(positive_edges, positive_tasks)
            if int(sample_task) == task_idx
        })
        if not task_positive_edges:
            continue

        src_type, _, dst_type = link_type
        split_sources = sorted({src for src, _ in task_positive_edges})
        num_dst_nodes = int(topology_graph[dst_type].num_nodes)
        requested_negatives = int(round(
            len(task_positive_edges) * negative_ratio
        ))

        forbidden = set(all_positive_sets.get(task_idx, set()))
        forbidden.update(reserved_negatives.setdefault(task_idx, set()))


        candidates = [
            (src_idx, dst_idx)
            for src_idx in split_sources
            for dst_idx in range(num_dst_nodes)
            if (src_idx, dst_idx) not in forbidden
        ]
        rng.shuffle(candidates)

        bucket_candidates: Dict[Any, List[Tuple[int, int]]] = {
            bucket: [] for bucket in sampling_order
        }

        for src_idx, dst_idx in candidates:
            distance = get_topology_destinations(
                src_type, src_idx, dst_type
            ).get(dst_idx)

            if distance in topology_hop_order:
                bucket_candidates[distance].append((src_idx, dst_idx))
            else:
                bucket_candidates["random"].append((src_idx, dst_idx))


        quota_percentages = {5: 30, 4: 30, 3: 30, "random": 10}
        weighted_quotas = {
            bucket: requested_negatives * quota_percentages[bucket]
            for bucket in sampling_order
        }
        bucket_quotas = {
            bucket: weighted_quotas[bucket] // 100
            for bucket in sampling_order
        }
        quota_remainder = requested_negatives - sum(bucket_quotas.values())
        remainder_priority = sorted(
            sampling_order,
            key=lambda bucket: (
                -(weighted_quotas[bucket] % 100),
                sampling_order.index(bucket)
            )
        )
        for bucket in remainder_priority[:quota_remainder]:
            bucket_quotas[bucket] += 1

        selected_by_bucket: Dict[Any, List[Tuple[int, int]]] = {
            bucket: [] for bucket in sampling_order
        }
        selected_set = set()
        carried_shortfall = 0
        effective_targets: Dict[Any, int] = {}


        for bucket in sampling_order:
            effective_target = bucket_quotas[bucket] + carried_shortfall
            effective_targets[bucket] = effective_target
            available = [
                pair for pair in bucket_candidates[bucket]
                if pair not in selected_set
            ]
            chosen = available[:effective_target]

            selected_by_bucket[bucket].extend(chosen)
            selected_set.update(chosen)
            carried_shortfall = effective_target - len(chosen)

        selected_negatives = [
            pair
            for bucket in sampling_order
            for pair in selected_by_bucket[bucket]
        ]


        remaining_needed = requested_negatives - len(selected_negatives)
        if remaining_needed > 0:
            recovery_order: Tuple[Any, ...] = (3, 4, 5, "random")
            for bucket in recovery_order:
                if remaining_needed <= 0:
                    break

                available_extras = [
                    pair for pair in bucket_candidates[bucket]
                    if pair not in selected_set
                ]
                extras = available_extras[:remaining_needed]
                selected_by_bucket[bucket].extend(extras)
                selected_set.update(extras)
                selected_negatives.extend(extras)
                remaining_needed -= len(extras)

        if len(selected_negatives) < requested_negatives:
            print(
                f"Warning: task {link_type} requested {requested_negatives} "
                f"negatives but only {len(selected_negatives)} eligible "
                f"absent pairs were available."
            )

        reserved_negatives[task_idx].update(selected_negatives)

        target_summary = ", ".join(
            (
                f"{bucket}-hop={bucket_quotas[bucket]}"
                if bucket != "random"
                else f"random={bucket_quotas[bucket]}"
            )
            for bucket in sampling_order
        )
        effective_summary = ", ".join(
            (
                f"{bucket}-hop={effective_targets[bucket]}"
                if bucket != "random"
                else f"random={effective_targets[bucket]}"
            )
            for bucket in sampling_order
        )
        selected_summary = ", ".join(
            (
                f"{bucket}-hop={len(selected_by_bucket[bucket])}"
                if bucket != "random"
                else f"random={len(selected_by_bucket[bucket])}"
            )
            for bucket in sampling_order
        )
        available_summary = ", ".join(
            (
                f"{bucket}-hop={len(bucket_candidates[bucket])}"
                if bucket != "random"
                else f"random={len(bucket_candidates[bucket])}"
            )
            for bucket in sampling_order
        )

        print(
            f"{link_type}: {len(task_positive_edges)} positives, "
            f"{len(selected_negatives)} negatives. "
            f"Base targets [{target_summary}]; "
            f"cascade targets [{effective_summary}]; "
            f"available [{available_summary}]; "
            f"selected [{selected_summary}]."
        )

        task_edges = task_positive_edges + selected_negatives
        task_labels = (
            [1] * len(task_positive_edges)
            + [0] * len(selected_negatives)
        )
        task_ids = [task_idx] * len(task_edges)

        task_samples = list(zip(task_edges, task_labels, task_ids))
        rng.shuffle(task_samples)
        for edge, label, sample_task in task_samples:
            combined_edges.append(edge)
            combined_labels.append(label)
            combined_tasks.append(sample_task)

    all_samples = list(zip(combined_edges, combined_labels, combined_tasks))
    rng.shuffle(all_samples)
    if not all_samples:
        return [], [], []

    edges_out, labels_out, tasks_out = zip(*all_samples)
    return list(edges_out), list(labels_out), list(tasks_out)

def preprocess_data(
    dataframes: Dict[str, pd.DataFrame],
    config: Config
) -> Dict[str, pd.DataFrame]:

    print("Starting data preprocessing...")


    cancer_type_corrections = {}

    for df_key in ['circRNA_metastasis', 'mirna_metastasis']:
        if 'Cancer type' in dataframes[df_key].columns and cancer_type_corrections:
            dataframes[df_key]['Cancer type'] = (
                dataframes[df_key]['Cancer type'].replace(cancer_type_corrections)
            )


    print("Data preprocessing complete.\n")
    return dataframes
