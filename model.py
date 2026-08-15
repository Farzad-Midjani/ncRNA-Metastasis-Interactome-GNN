from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv

class HierarchicalRNAInteractionGNN(nn.Module):
    """
    Hierarchical Graph Neural Network for RNA-Cancer interaction analysis.

    This model combines heterogeneous GNN layers with TuckER-based link prediction. It features:
    - Hierarchical message passing with separate handling of primary and secondary edge types
    - Residual connections to mitigate over-smoothing in deep GNNs
    - Edge type attention to learn relative importance of different relationship types
    - Shared representation layer with task-specific adaptation layers
    - TuckER scoring for link prediction with relation-specific embeddings
    - Support for learning rate scheduling and gradient clipping

    The model incorporates a TuckER scoring function for knowledge graph embedding:
    score(h,r,t) = h^T (W ×₃ r) t
    Where W is a 3D tensor, ×₃ denotes tensor contraction along the 3rd mode,
    and h, r, t are the source, relation, and target embeddings respectively.

    Attributes:
    ----------
    hidden_channels : int
        Dimension of node embeddings and hidden features
    heads : int
        Number of attention heads in transformer convolution layers
    num_layers : int
        Number of GNN layers
    use_layer_norm : bool
        Whether to use layer normalization
    link_types : list
        List of all edge types in the graph
    link_prediction_tasks : list
        List of edge types for link prediction tasks
    out_channels : int
        Number of link prediction tasks
    primary_edge_types : list
        List of primary edge types (meta-relationships)
    secondary_edge_types : list
        List of secondary edge types (ncRNA-related relationships)
    use_residual : bool
        Whether to use residual connections
    core_tensor : nn.Parameter
        3D tensor for TuckER scoring function
    relation_embeddings : nn.Embedding
        Embeddings for different relation types
    learnable_missing_features : nn.ParameterDict
    """

    def __init__(
        self,
        hidden_channels: int,
        link_types: List[Tuple[str, str, str]],
        link_prediction_tasks: List[Tuple[str, str, str]],
        heads: int = 2,
        dropout: float = 0.4,
        dropout_rates: Dict[str, List[float]] = None,
        feature_dims: Dict[str, Dict[str, int]] = None,
        num_cancer_nodes: int = 0,
        num_csmi_nodes: int = 0,
        num_met_nodes: int = 0,
        num_layers: int = 2,
        use_layer_norm: bool = True
    ):

        super(HierarchicalRNAInteractionGNN, self).__init__()

        print("Initializing HierarchicalRNAInteractionGNN model with shared representation and task-specific adaptation...")
        self.hidden_channels = hidden_channels
        self.heads = heads

        if self.hidden_channels % self.heads != 0:
            raise ValueError(
                f"hidden_channels ({self.hidden_channels}) must be divisible "
                f"by heads ({self.heads})."
            )

        self.head_channels = self.hidden_channels // self.heads


        self.attention_output_channels = self.hidden_channels * self.heads

        self.num_layers = num_layers
        self.use_layer_norm = use_layer_norm
        self.link_types = link_types
        self.link_prediction_tasks = link_prediction_tasks
        self.out_channels = len(self.link_prediction_tasks)


        self.cancer_embeddings = nn.Embedding(num_cancer_nodes, self.hidden_channels) if num_cancer_nodes > 0 else None
        self.csmi_embeddings = nn.Embedding(num_csmi_nodes, self.hidden_channels) if num_csmi_nodes > 0 else None
        self.met_embeddings = nn.Embedding(
            num_met_nodes, self.hidden_channels
        ) if num_met_nodes > 0 else None


        if self.cancer_embeddings:
            nn.init.xavier_uniform_(self.cancer_embeddings.weight)
        if self.csmi_embeddings:
            nn.init.xavier_uniform_(self.csmi_embeddings.weight)
        if self.met_embeddings:
            nn.init.xavier_uniform_(self.met_embeddings.weight)


        self.learnable_missing_features = nn.ParameterDict()
        for node_type in ('miRNA', 'circRNA'):
            if node_type not in feature_dims:
                continue
            input_dim = feature_dims[node_type]['feature_dim']
            num_nodes = feature_dims[node_type].get('num_nodes', 0)
            if num_nodes <= 0:
                raise ValueError(
                    f"num_nodes must be positive for trainable '{node_type}' "
                    "missing features."
                )
            missing_values = torch.empty(num_nodes, input_dim)
            nn.init.xavier_uniform_(missing_values)
            self.learnable_missing_features[node_type] = nn.Parameter(
                missing_values
            )


        self.preprocess = nn.ModuleDict()
        for node_type in feature_dims.keys():
            input_dim = feature_dims[node_type]['feature_dim']
            if input_dim != self.hidden_channels:
                self.preprocess[node_type] = nn.Linear(
                    input_dim, self.hidden_channels
                )
            else:
                self.preprocess[node_type] = nn.Identity()


        primary_forward_edge_types = {
            ('CSMI', 'represents', 'MET'),
            ('CSMI', 'related_to', 'Cancer'),
            ('MET', 'related_to', 'Cancer')
        }
        secondary_forward_edge_types = {
            ('miRNA', 'regulates', 'Cancer'),
            ('miRNA', 'regulates', 'CSMI'),
            ('circRNA', 'regulates', 'Cancer'),
            ('circRNA', 'regulates', 'CSMI'),
            ('circRNA', 'regulates', 'miRNA'),
            ('miRNA', 'regulates', 'MET'),
            ('circRNA', 'regulates', 'MET')
        }

        def belongs_to_family(edge_type, forward_family):
            if edge_type in forward_family:
                return True
            src_type, relation, dst_type = edge_type
            if relation.startswith('rev_'):
                original_type = (dst_type, relation[4:], src_type)
                return original_type in forward_family
            return False


        self.primary_edge_types = [
            edge_type for edge_type in link_types
            if belongs_to_family(edge_type, primary_forward_edge_types)
        ]
        self.secondary_edge_types = [
            edge_type for edge_type in link_types
            if belongs_to_family(edge_type, secondary_forward_edge_types)
        ]

        if not self.primary_edge_types:
            raise ValueError('No primary message-passing edge types were found.')
        if not self.secondary_edge_types:
            raise ValueError('No secondary message-passing edge types were found.')

        self.primary_edge_keys = {
            edge_type: self._edge_type_key(edge_type)
            for edge_type in self.primary_edge_types
        }
        self.secondary_edge_keys = {
            edge_type: self._edge_type_key(edge_type)
            for edge_type in self.secondary_edge_types
        }


        self.conv_primary_layers = nn.ModuleList()
        self.conv_secondary_layers = nn.ModuleList()
        for layer_idx in range(num_layers):
            primary_dropout = (
                dropout_rates['primary'][layer_idx]
                if dropout_rates and 'primary' in dropout_rates else dropout
            )
            secondary_dropout = (
                dropout_rates['secondary'][layer_idx]
                if dropout_rates and 'secondary' in dropout_rates else dropout
            )

            self.conv_primary_layers.append(nn.ModuleDict({
                self.primary_edge_keys[edge_type]: TransformerConv(
                    in_channels=(self.hidden_channels, self.hidden_channels),
                    out_channels=self.hidden_channels,
                    heads=heads,
                    concat=True,
                    dropout=primary_dropout,
                    edge_dim=1
                )
                for edge_type in self.primary_edge_types
            }))
            self.conv_secondary_layers.append(nn.ModuleDict({
                self.secondary_edge_keys[edge_type]: TransformerConv(
                    in_channels=(self.hidden_channels, self.hidden_channels),
                    out_channels=self.hidden_channels,
                    heads=heads,
                    concat=True,
                    dropout=secondary_dropout,
                    edge_dim=1
                )
                for edge_type in self.secondary_edge_types
            }))

        if self.use_layer_norm:
            self.layer_norm_primary = nn.ModuleList([
                nn.LayerNorm(self.hidden_channels) for _ in range(num_layers)
            ])
            self.layer_norm_secondary = nn.ModuleList([
                nn.LayerNorm(self.hidden_channels) for _ in range(num_layers)
            ])


        all_node_types = list(feature_dims.keys())
        primary_node_types = sorted({
            node_type
            for edge_type in self.primary_edge_types
            for node_type in (edge_type[0], edge_type[2])
        })


        self.lin_primary = nn.ModuleDict({
            node_type: nn.Linear(
                self.attention_output_channels, self.hidden_channels
            )
            for node_type in primary_node_types
        })
        self.lin_secondary = nn.ModuleDict({
            node_type: nn.Linear(
                self.attention_output_channels, self.hidden_channels
            )
            for node_type in all_node_types
        })

        self.use_residual = True
        self.activation = nn.LeakyReLU(0.1)
        self.dropout_layer = nn.Dropout(p=dropout)


        self.edge_attention = nn.ParameterDict({
            'primary': nn.Parameter(torch.zeros(len(self.primary_edge_types))),
            'secondary': nn.Parameter(torch.zeros(len(self.secondary_edge_types)))
        })


        self.shared_representation = nn.ModuleDict()
        for node_type in ['miRNA', 'circRNA', 'Cancer', 'CSMI', 'MET']:

            self.shared_representation[node_type] = nn.Sequential(
                nn.Linear(self.hidden_channels, self.hidden_channels * 2),
                nn.LayerNorm(self.hidden_channels * 2),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout),
                nn.Linear(self.hidden_channels * 2, self.hidden_channels),
                nn.LayerNorm(self.hidden_channels)
            )


        self.task_adaptation = nn.ModuleDict()
        for task_idx, (src_type, relation, dst_type) in enumerate(self.link_prediction_tasks):
            task_name = f"{src_type}_{relation}_{dst_type}"


            self.task_adaptation[f"{task_name}_src"] = nn.Sequential(
                nn.Linear(self.hidden_channels, self.hidden_channels),
                nn.LayerNorm(self.hidden_channels),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout/2)
            )


            self.task_adaptation[f"{task_name}_dst"] = nn.Sequential(
                nn.Linear(self.hidden_channels, self.hidden_channels),
                nn.LayerNorm(self.hidden_channels),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout/2)
            )


        self.relation_dim = 12
        self.num_relations = len(link_prediction_tasks)
        self.relation_embeddings = nn.Embedding(self.num_relations, self.relation_dim)


        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        print(f"Initialized relation embeddings with shape ({self.num_relations}, {self.relation_dim}).")


        self.core_tensor = nn.Parameter(torch.empty(
            self.hidden_channels, self.relation_dim, self.hidden_channels
        ))

        nn.init.xavier_uniform_(self.core_tensor, gain=0.5)
        print(f"Initialized TuckER core tensor with shape {self.core_tensor.shape}.")


        self.lr_scheduler = None

        self.dropout = dropout
        print("Model initialization complete with shared representation and task-specific adaptations.")

    @staticmethod
    def _edge_type_key(edge_type: Tuple[str, str, str]) -> str:
        return '__'.join(edge_type)

    def _relation_weighted_message_pass(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor],
        convs: nn.ModuleDict,
        edge_types: List[Tuple[str, str, str]],
        edge_keys: Dict[Tuple[str, str, str], str],
        attention_logits: torch.Tensor
    ) -> Dict[str, torch.Tensor]:

        outputs_by_destination: Dict[str, List[torch.Tensor]] = {}
        logits_by_destination: Dict[str, List[torch.Tensor]] = {}

        for relation_idx, edge_type in enumerate(edge_types):
            if edge_type not in edge_index_dict:
                continue

            edge_index = edge_index_dict[edge_type]
            if edge_index.numel() == 0 or edge_index.size(1) == 0:
                continue

            src_type, _, dst_type = edge_type
            edge_attr = edge_attr_dict.get(edge_type)
            relation_output = convs[edge_keys[edge_type]](
                (x_dict[src_type], x_dict[dst_type]),
                edge_index,
                edge_attr
            )
            outputs_by_destination.setdefault(dst_type, []).append(relation_output)
            logits_by_destination.setdefault(dst_type, []).append(
                attention_logits[relation_idx]
            )

        aggregated: Dict[str, torch.Tensor] = {}
        for dst_type, relation_outputs in outputs_by_destination.items():
            destination_logits = torch.stack(logits_by_destination[dst_type])
            destination_weights = torch.softmax(destination_logits, dim=0)


            destination_weights = destination_weights * len(relation_outputs)
            stacked_outputs = torch.stack(relation_outputs, dim=0)
            weight_shape = [destination_weights.size(0)] + [1] * (
                stacked_outputs.dim() - 1
            )
            aggregated[dst_type] = (
                stacked_outputs * destination_weights.view(*weight_shape)
            ).sum(dim=0)

        return aggregated

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        edge_attr_dict: Dict[Tuple[str, str, str], torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        device = next(self.parameters()).device

        x_dict = {
            node_type: features.to(device)
            for node_type, features in x_dict.items()
        }
        edge_index_dict = {
            edge_type: edge_index.to(device)
            for edge_type, edge_index in edge_index_dict.items()
        }
        edge_attr_dict = {
            edge_type: (
                edge_attr.to(device) if edge_attr is not None else None
            )
            for edge_type, edge_attr in edge_attr_dict.items()
        }

        if self.cancer_embeddings is not None:
            indices = torch.arange(
                self.cancer_embeddings.num_embeddings, device=device
            )
            x_dict['Cancer'] = self.cancer_embeddings(indices)
        if self.csmi_embeddings is not None:
            indices = torch.arange(
                self.csmi_embeddings.num_embeddings, device=device
            )
            x_dict['CSMI'] = self.csmi_embeddings(indices)
        if self.met_embeddings is not None:
            indices = torch.arange(
                self.met_embeddings.num_embeddings, device=device
            )
            x_dict['MET'] = self.met_embeddings(indices)

        for node_type, features in list(x_dict.items()):
            if node_type in self.learnable_missing_features:
                if torch.isinf(features).any():
                    raise ValueError(
                        f"Infinite values found in '{node_type}' features."
                    )
                missing_mask = torch.isnan(features)
                if missing_mask.any():
                    learned_values = self.learnable_missing_features[node_type]
                    if learned_values.shape != features.shape:
                        raise ValueError(
                            f"Trainable missing-feature shape mismatch for "
                            f"'{node_type}': expected {tuple(features.shape)}, "
                            f"got {tuple(learned_values.shape)}."
                        )


                    features = torch.where(
                        missing_mask, learned_values, features
                    )
            x_dict[node_type] = self.preprocess[node_type](features)

        original_x_dict = {key: value.clone() for key, value in x_dict.items()}

        for layer_idx, convs in enumerate(self.conv_primary_layers):
            previous = {key: value.clone() for key, value in x_dict.items()}
            primary_out = self._relation_weighted_message_pass(
                x_dict=x_dict,
                edge_index_dict=edge_index_dict,
                edge_attr_dict=edge_attr_dict,
                convs=convs,
                edge_types=self.primary_edge_types,
                edge_keys=self.primary_edge_keys,
                attention_logits=self.edge_attention['primary']
            )

            for node_type, node_output in list(primary_out.items()):
                if node_type not in self.lin_primary:
                    continue
                node_output = self.lin_primary[node_type](node_output)
                if self.use_layer_norm:
                    node_output = self.layer_norm_primary[layer_idx](node_output)
                node_output = self.activation(node_output)
                node_output = self.dropout_layer(node_output)
                if self.use_residual and node_type in previous:
                    node_output = node_output + previous[node_type]
                primary_out[node_type] = node_output

            x_dict.update(primary_out)

        for layer_idx, convs in enumerate(self.conv_secondary_layers):
            previous = {key: value.clone() for key, value in x_dict.items()}
            secondary_out = self._relation_weighted_message_pass(
                x_dict=x_dict,
                edge_index_dict=edge_index_dict,
                edge_attr_dict=edge_attr_dict,
                convs=convs,
                edge_types=self.secondary_edge_types,
                edge_keys=self.secondary_edge_keys,
                attention_logits=self.edge_attention['secondary']
            )

            for node_type, node_output in list(secondary_out.items()):
                if node_type not in self.lin_secondary:
                    continue
                node_output = self.lin_secondary[node_type](node_output)
                if self.use_layer_norm:
                    node_output = self.layer_norm_secondary[layer_idx](node_output)
                node_output = self.activation(node_output)
                node_output = self.dropout_layer(node_output)
                if self.use_residual and node_type in previous:
                    node_output = node_output + previous[node_type]
                secondary_out[node_type] = node_output

            x_dict.update(secondary_out)

        if self.use_residual:
            for node_type, original_features in original_x_dict.items():
                if node_type in x_dict:
                    x_dict[node_type] = x_dict[node_type] + original_features

        shared_x_dict = {}
        for node_type, features in x_dict.items():
            if node_type in self.shared_representation:
                shared_output = self.shared_representation[node_type](features)
                shared_x_dict[node_type] = features + shared_output
            else:
                shared_x_dict[node_type] = features

        return shared_x_dict

    def get_task_adapted_embeddings(
        self,
        x_dict: Dict[str, torch.Tensor],
        task_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        src_type, relation, dst_type = self.link_prediction_tasks[task_idx]
        task_name = f"{src_type}_{relation}_{dst_type}"


        src_adapted = self.task_adaptation[f"{task_name}_src"](x_dict[src_type])
        dst_adapted = self.task_adaptation[f"{task_name}_dst"](x_dict[dst_type])

        return src_adapted, dst_adapted

    def predict_links(
        self,
        x_dict: Dict[str, torch.Tensor],
        src_indices: List[int],
        dst_indices: List[int],
        link_types: List[Tuple[str, str, str]]
    ) -> torch.Tensor:

        device = next(self.parameters()).device
        predictions = torch.zeros(len(link_types), device=device)

        for i, lt in enumerate(link_types):
            if lt not in self.link_prediction_tasks:
                continue


            relation_idx = self.link_prediction_tasks.index(lt)
            relation_idx_tensor = torch.tensor(relation_idx, device=device)
            r = self.relation_embeddings(relation_idx_tensor)


            src_type, _, dst_type = lt


            src_adapted, dst_adapted = self.get_task_adapted_embeddings(x_dict, relation_idx)


            h = src_adapted[src_indices[i]].unsqueeze(0)
            t = dst_adapted[dst_indices[i]].unsqueeze(0)


            W_r = torch.einsum('hrd,r->hd', self.core_tensor, r)


            h_W_r = torch.matmul(h, W_r)


            score = torch.bmm(
                h_W_r.unsqueeze(1), t.unsqueeze(2)
            ).squeeze(2).squeeze(1).squeeze(0)


            predictions[i] = torch.sigmoid(score)

        return predictions
