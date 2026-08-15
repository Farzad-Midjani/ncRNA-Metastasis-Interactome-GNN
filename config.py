from __future__ import annotations

import os


class Config:
    def __init__(self, data_dir: str = "data", output_dir: str = "outputs"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.graph_info_dir = os.path.join(self.output_dir, "graph_info")
        self.missing_nodes_log = os.path.join(self.output_dir, "missing_nodes_logs")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.graph_info_dir, exist_ok=True)
        os.makedirs(self.missing_nodes_log, exist_ok=True)

        self.circRNA_metastasis_file = "circrna_metastasis_regulation.xlsx"
        self.cleaned_circRNA_file = "circrna_expression_features.csv"
        self.circbase_interactions_file = "circrna_mirna_interactions.csv"
        self.mirna_expression_file = "mirna_expression_features.csv"
        self.mirna_metastasis_file = "mirna_metastasis_regulation.xlsx"

        self.hidden_channels = 12
        self.heads = 2
        self.dropout = 0.4
        self.learning_rate = 0.001
        self.weight_decay = 1e-3
        self.epochs = 150
        self.batch_size = 64
        self.cross_val_folds = 4
        self.patience = 10
        self.num_layers = 2
        self.disjoint_train_ratio = 0.20

        self.dropout_rates = {
            "primary": [0.3 for _ in range(self.num_layers)],
            "secondary": [0.3 for _ in range(self.num_layers)],
        }

        self.primary_link_types = [
            ("CSMI", "represents", "MET"),
            ("CSMI", "related_to", "Cancer"),
            ("MET", "related_to", "Cancer"),
        ]

        self.secondary_link_types = [
            ("miRNA", "regulates", "Cancer"),
            ("miRNA", "regulates", "CSMI"),
            ("circRNA", "regulates", "Cancer"),
            ("circRNA", "regulates", "CSMI"),
            ("circRNA", "regulates", "miRNA"),
            ("miRNA", "regulates", "MET"),
            ("circRNA", "regulates", "MET"),
        ]

        self.link_types = self.primary_link_types + self.secondary_link_types
        self.all_link_types = self.link_types.copy()

        self.link_prediction_tasks = [
            ("miRNA", "regulates", "CSMI"),
            ("circRNA", "regulates", "CSMI"),
            ("miRNA", "regulates", "Cancer"),
            ("circRNA", "regulates", "Cancer"),
            ("miRNA", "regulates", "MET"),
            ("circRNA", "regulates", "MET"),
        ]

        self.out_channels = len(self.link_prediction_tasks)
        self.top_k = 1000
