"""
Graph Neural Network Backbone for Brain Tumor Tissue Topology and Heterogeneity.
Processes Superpixel Region Adjacency Graphs using Graph Convolutional Networks (GCN/GAT).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from typing import Tuple, Optional


class BrainTumorGNN(nn.Module):
    """
    Graph Neural Network capturing topological connectivity between tumor subregions,
    edema boundaries, and surrounding healthy brain parenchyma.
    """
    def __init__(
        self,
        in_channels: int = 64,
        hidden_dim: int = 128,
        out_dim: int = 128,
        use_attention: bool = True
    ):
        super().__init__()
        self.use_attention = use_attention

        # Graph Convolution / Attention Layers
        if use_attention:
            self.conv1 = GATConv(in_channels, hidden_dim // 2, heads=2, concat=True)  # -> hidden_dim
            self.conv2 = GATConv(hidden_dim, hidden_dim // 2, heads=2, concat=True)   # -> hidden_dim
            self.conv3 = GCNConv(hidden_dim, hidden_dim)
        else:
            self.conv1 = GCNConv(in_channels, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)

        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Node importance scoring for explainability
        self.node_attention = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Graph Readout & Projection
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
          edge_index: Graph connectivity [2, Total_Edges]
          batch: Batch assignment vector [Total_Nodes]
          edge_weight: Edge attribute weights [Total_Edges, 1]
        Returns:
          - graph_embedding: [Batch_Size, out_dim]
          - node_weights: [Total_Nodes, 1] (attention weights for explainability)
        """
        # Layer 1
        h1 = self.conv1(x, edge_index)
        h1 = self.bn1(h1)
        h1 = F.leaky_relu(h1, 0.1)

        # Layer 2
        h2 = self.conv2(h1, edge_index)
        h2 = self.bn2(h2)
        h2 = F.leaky_relu(h2 + h1, 0.1)  # Residual connection

        # Layer 3
        h3 = self.conv3(h2, edge_index)
        h3 = self.bn3(h3)
        h3 = F.leaky_relu(h3 + h2, 0.1)

        # Node importance / attention weights
        node_weights = torch.sigmoid(self.node_attention(h3))  # [Total_Nodes, 1]

        # Multi-pooling readout: mean and max across nodes per graph
        mean_pooled = global_mean_pool(h3 * node_weights, batch)
        max_pooled = global_max_pool(h3, batch)
        graph_repr = torch.cat([mean_pooled, max_pooled], dim=1)  # [B, hidden_dim * 2]

        graph_embedding = self.proj(graph_repr)  # [B, out_dim]

        return graph_embedding, node_weights


if __name__ == "__main__":
    from torch_geometric.data import Data, Batch
    model = BrainTumorGNN(in_channels=64, hidden_dim=128, out_dim=128)
    
    # Dummy graph batch with 2 graphs of 49 nodes each
    g1 = Data(x=torch.randn(49, 64), edge_index=torch.randint(0, 49, (2, 200)))
    g2 = Data(x=torch.randn(49, 64), edge_index=torch.randint(0, 49, (2, 200)))
    batch_graphs = Batch.from_data_list([g1, g2])

    emb, weights = model(batch_graphs.x, batch_graphs.edge_index, batch_graphs.batch)
    print("GNN Graph Embedding Shape:", emb.shape)
    print("GNN Node Attention Weights Shape:", weights.shape)
