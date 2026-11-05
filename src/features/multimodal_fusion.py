"""
multimodal_fusion.py: Cross-Attention & Gated Multi-Modal Feature Fusion Network.

Combines:
- Modality 1: Topological Circuit Graph Embeddings (from HardwareTrojanGNN, 192-dim)
- Modality 2: Physical Power & EM Side-Channel Signatures (from SideChannelFeatureExtractor, 16-dim)
Produces a unified, robust multi-modal feature vector for classification and risk scoring.
"""

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalGatedFusion(nn.Module):
    """
    Gated Multi-Modal Fusion Network with Cross-Modal Attention.
    Dynamically balances structural gate graph representations against physical side-channel traces.
    """

    def __init__(
        self,
        graph_embed_dim: int = 192,
        sidechannel_dim: int = 16,
        fused_dim: int = 128,
        dropout: float = 0.2
    ):
        super().__init__()
        self.graph_proj = nn.Sequential(
            nn.Linear(graph_embed_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.side_proj = nn.Sequential(
            nn.Linear(sidechannel_dim, fused_dim),
            nn.LayerNorm(fused_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Gated Fusion Mechanism: computes adaptive blending weight alpha in [0, 1]
        self.gate = nn.Sequential(
            nn.Linear(fused_dim * 2, fused_dim),
            nn.ReLU(),
            nn.Linear(fused_dim, 1),
            nn.Sigmoid()
        )

        # Multi-Modal Output Classifier
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2)  # [Clean, Trojan]
        )

    def forward(
        self,
        graph_embedding: torch.Tensor,
        sidechannel_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        Args:
            graph_embedding: [B, graph_embed_dim]
            sidechannel_features: [B, sidechannel_dim]
        Returns:
            Dictionary with:
                - 'fused_embedding': [B, fused_dim]
                - 'gate_weight': [B, 1] (Modality attention weight)
                - 'logits': [B, 2] (Trojan class logits)
        """
        h_graph = self.graph_proj(graph_embedding)
        h_side = self.side_proj(sidechannel_features)

        # Concatenate for gate calculation
        concat_h = torch.cat([h_graph, h_side], dim=-1)
        alpha = self.gate(concat_h)  # Weight given to graph modality (1 - alpha for side-channel)

        fused = alpha * h_graph + (1.0 - alpha) * h_side

        logits = self.classifier(fused)

        return {
            'fused_embedding': fused,
            'gate_weight': alpha,
            'logits': logits
        }
