"""
gnn_detector.py: Graph Neural Network architecture for Hardware Trojan Detection & Localization.

Implements a unified dual-task architecture:
1. Graph-level Head: Classifies circuit netlists as Golden (0) or Trojan-Infected (1).
2. Node-level Head: Identifies and localizes compromised Trojan gates/triggers.
3. Latent Representation: Extracts graph embeddings for unseen/zero-day Trojan anomaly detection.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv, global_add_pool, global_max_pool, global_mean_pool


class HardwareTrojanGNN(nn.Module):
    """
    Dual-Task Graph Neural Network for Hardware Trojan Detection and Localization.

    Supports GCN, GAT, and GraphSAGE backbones with residual connections, layer normalization,
    and joint graph-classification / node-classification heads.
    """

    def __init__(
        self,
        in_channels: int = 20,
        hidden_channels: int = 64,
        num_layers: int = 3,
        backbone: str = 'gcn',
        dropout: float = 0.2,
        heads: int = 4  # for GAT
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.backbone_name = backbone.lower()
        self.dropout_rate = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        # Build GNN Backbone
        current_dim = in_channels
        for i in range(num_layers):
            if self.backbone_name == 'gat':
                # GAT with multi-head attention
                if i < num_layers - 1:
                    conv = GATConv(current_dim, hidden_channels // heads, heads=heads, dropout=dropout)
                    current_dim = hidden_channels
                else:
                    conv = GATConv(current_dim, hidden_channels, heads=1, concat=False, dropout=dropout)
                    current_dim = hidden_channels
            elif self.backbone_name == 'sage':
                conv = SAGEConv(current_dim, hidden_channels)
                current_dim = hidden_channels
            else:  # default 'gcn'
                conv = GCNConv(current_dim, hidden_channels)
                current_dim = hidden_channels

            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_channels))

        # Graph Pooling Combination (Mean + Max + Add pooling = 3 * hidden_channels)
        self.graph_embed_dim = hidden_channels * 3

        # Task 1: Graph-Level Trojan Detection Head (Binary Classification)
        self.graph_classifier = nn.Sequential(
            nn.Linear(self.graph_embed_dim, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 2)  # [Logits: Clean, Trojan]
        )

        # Task 2: Node-Level Trojan Localization Head (Gate-level classification)
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 2)  # [Logits: Clean Gate, Trojan Gate]
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Args:
            x: Node feature matrix [N, in_channels]
            edge_index: Graph edge indices [2, E]
            batch: Batch vector mapping each node to graph index in batch [N]
        Returns:
            Dictionary with:
                - 'graph_logits': [B, 2]
                - 'node_logits': [N, 2]
                - 'graph_embedding': [B, embed_dim]
                - 'node_embeddings': [N, hidden_channels]
        """
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)

        h = x
        for i in range(self.num_layers):
            h_new = self.convs[i](h, edge_index)
            h_new = self.norms[i](h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout_rate, training=self.training)
            # Residual connection if dims match
            if h.size(1) == h_new.size(1):
                h = h + h_new
            else:
                h = h_new

        node_embeddings = h

        # 1. Node-Level Localization Logits
        node_logits = self.node_classifier(node_embeddings)

        # 2. Graph Pooling (Mean + Max + Sum for expressive topological representation)
        pool_mean = global_mean_pool(node_embeddings, batch)
        pool_max = global_max_pool(node_embeddings, batch)
        pool_add = global_add_pool(node_embeddings, batch)
        graph_embedding = torch.cat([pool_mean, pool_max, pool_add], dim=-1)

        # 3. Graph-Level Detection Logits
        graph_logits = self.graph_classifier(graph_embedding)

        return {
            'graph_logits': graph_logits,
            'node_logits': node_logits,
            'graph_embedding': graph_embedding,
            'node_embeddings': node_embeddings
        }

    def predict_probabilities(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes calibrated Trojan probabilities.
        Returns:
            graph_prob: Probability that the circuit contains a Trojan [B]
            node_prob: Probability that each gate is a Trojan gate [N]
        """
        self.eval()
        with torch.no_grad():
            out = self.forward(x, edge_index, batch)
            graph_prob = F.softmax(out['graph_logits'], dim=-1)[:, 1]
            node_prob = F.softmax(out['node_logits'], dim=-1)[:, 1]
        return graph_prob, node_prob
