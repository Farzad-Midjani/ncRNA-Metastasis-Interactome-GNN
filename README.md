A context-aware multi-task heterogeneous graph neural network for predicting ncRNA associations with cancer types, metastatic events, and cancer-specific metastatic instances (CSMIs). The model uses hierarchical Transformer-based message passing and TuckER link scoring across six prediction tasks.

## Project structure

```text
config.py      Configuration and file paths
data.py        Data loading, preprocessing, graph construction, splitting, and negative sampling
model.py       Hierarchical heterogeneous GNN and TuckER decoder
train.py       Training, evaluation, threshold selection, and cross-validation
reporting.py   Graph visualization, graph summaries, and prediction export
main.py        End-to-end pipeline
```