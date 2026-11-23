"""
Enhanced Graph Neural Network Backbone for Brain Tumor Tissue Topology.
Implements Multi-Head Graph Attention Networks (GATv2) with residual skip-connections
and topological node importance for explainability.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, global_mean_pool, global_max_pool
from typing import Tuple, Optional


class BrainTumorGNN(nn.Module):
    """
    Advanced Graph Attention Network (GATv2) capturing fine-grained microenvironment
    heterogeneity between necrotic center, active rim, peritumoral edema, and brain parenchyma.
    """
    def __init__(
        self,
        in_channels: int = 64,
        hidden_dim: int = 128,
        out_dim: int = 128,
        heads: int = 4
    ):
        super().__init__()
        self.heads = heads
        head_dim = hidden_dim // heads

        # Multi-Head Graph Attention Layers
        self.conv1 = GATv2Conv(in_channels, head_dim, heads=heads, concat=True)  # -> hidden_dim
        self.conv2 = GATv2Conv(hidden_dim, head_dim, heads=heads, concat=True)   # -> hidden_dim
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Skip projection if in_channels != hidden_dim
        self.skip_proj = nn.Linear(in_channels, hidden_dim) if in_channels != hidden_dim else nn.Identity()

        # Node importance scoring for spatial interpretability
        self.node_attention = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Multi-Scale Readout projection
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
          x: Node features [Total_Nodes, in_channels]
          edge_index: Connectivity [2, Total_Edges]
          batch: Batch vector [Total_Nodes]
        Returns:
          - graph_embedding: [Batch_Size, out_dim]
          - node_weights: [Total_Nodes, 1]
        """
        residual = self.skip_proj(x)

        # Layer 1
        h1 = self.conv1(x, edge_index)
        h1 = self.bn1(h1)
        h1 = F.leaky_relu(h1 + residual, 0.1)

        # Layer 2 with residual
        h2 = self.conv2(h1, edge_index)
        h2 = self.bn2(h2)
        h2 = F.leaky_relu(h2 + h1, 0.1)

        # Layer 3
        h3 = self.conv3(h2, edge_index)
        h3 = self.bn3(h3)
        h3 = F.leaky_relu(h3 + h2, 0.1)

        # Learned node importance
        node_weights = torch.sigmoid(self.node_attention(h3))  # [Total_Nodes, 1]

        # Multi-scale pooling
        mean_pooled = global_mean_pool(h3 * node_weights, batch)
        max_pooled = global_max_pool(h3, batch)
        graph_repr = torch.cat([mean_pooled, max_pooled], dim=1)  # [B, hidden_dim * 2]

        graph_embedding = self.proj(graph_repr)  # [B, out_dim]

        return graph_embedding, node_weights
