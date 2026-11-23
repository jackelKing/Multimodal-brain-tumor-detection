"""
Multimodal CNN-GNN-Clinical Fusion Model for Brain Tumor Detection, Classification, and Grade Prediction.
Integrates visual convolutional features, spatial topological graph connectivity, and clinical biomarkers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from typing import Dict, Any, Tuple, Optional

from src.models.cnn_backbone import BrainTumorCNN
from src.models.gnn_backbone import BrainTumorGNN
from src.models.clinical_mlp import ClinicalMLP
from src.config import (
    CNN_EMBEDDING_DIM,
    GNN_EMBEDDING_DIM,
    CLINICAL_FEATURE_DIM,
    CLINICAL_EMBED_DIM,
    FUSION_HIDDEN_DIM,
    NUM_CLASSES,
    NUM_GRADES
)


class GatedMultimodalFusion(nn.Module):
    """Gated Cross-Modal Fusion mechanism dynamically weighting visual, graph, and clinical modalities."""
    def __init__(self, cnn_dim: int, gnn_dim: int, clin_dim: int, fused_dim: int):
        super().__init__()
        # Modal gating networks
        self.gate_cnn = nn.Sequential(nn.Linear(cnn_dim, cnn_dim), nn.Sigmoid())
        self.gate_gnn = nn.Sequential(nn.Linear(gnn_dim, gnn_dim), nn.Sigmoid())
        self.gate_clin = nn.Sequential(nn.Linear(clin_dim, clin_dim), nn.Sigmoid())

        # Joint projection network
        total_in = cnn_dim + gnn_dim + clin_dim
        self.fusion_fc = nn.Sequential(
            nn.Linear(total_in, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(fused_dim, fused_dim),
            nn.BatchNorm1d(fused_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(
        self,
        h_cnn: torch.Tensor,
        h_gnn: torch.Tensor,
        h_clin: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Compute dynamic modal gates
        g_c = self.gate_cnn(h_cnn)
        g_g = self.gate_gnn(h_gnn)
        g_cl = self.gate_clin(h_clin)

        # Modulated features
        mod_cnn = h_cnn * g_c
        mod_gnn = h_gnn * g_g
        mod_clin = h_clin * g_cl

        # Concatenate and project to joint multimodal latent space
        fused_concat = torch.cat([mod_cnn, mod_gnn, mod_clin], dim=1)
        z_fused = self.fusion_fc(fused_concat)

        gate_weights = {
            "cnn_gate": g_c.mean(dim=1),
            "gnn_gate": g_g.mean(dim=1),
            "clinical_gate": g_cl.mean(dim=1)
        }

        return z_fused, gate_weights


class MultimodalBrainTumorNet(nn.Module):
    """
    Complete Multimodal Neural Network combining:
      1. CNN Backbone: MRI spatial feature maps and global representation
      2. GNN Backbone: Graph neural network on tissue patch topology
      3. Clinical MLP: Patient biomarker & symptom encoder
      4. Gated Multimodal Fusion
      5. Multi-Task Heads: Binary Detection, 4-Class Classification, WHO Grade Prediction
    """
    def __init__(
        self,
        cnn_dim: int = CNN_EMBEDDING_DIM,
        gnn_dim: int = GNN_EMBEDDING_DIM,
        clin_dim: int = CLINICAL_EMBED_DIM,
        fused_dim: int = FUSION_HIDDEN_DIM,
        num_classes: int = NUM_CLASSES,
        num_grades: int = NUM_GRADES,
        pretrained_cnn: bool = True
    ):
        super().__init__()
        # 1. Modality encoders
        self.cnn = BrainTumorCNN(embedding_dim=cnn_dim, pretrained=pretrained_cnn)
        self.gnn = BrainTumorGNN(in_channels=64, hidden_dim=gnn_dim, out_dim=gnn_dim, use_attention=True)
        self.clinical_encoder = ClinicalMLP(in_features=CLINICAL_FEATURE_DIM, hidden_dim=clin_dim, out_dim=clin_dim)

        # 2. Multimodal Fusion
        self.fusion = GatedMultimodalFusion(cnn_dim, gnn_dim, clin_dim, fused_dim)

        # 3. Multi-Task Diagnostic Prediction Heads
        # Head A: Detection (Tumor vs Normal)
        self.detection_head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 2)
        )

        # Head B: 4-Class Classification (Glioma, Meningioma, No Tumor, Pituitary)
        self.classification_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

        # Head C: Prognostic WHO Histological Grade Prediction (Grade I - IV)
        self.grade_head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, num_grades)
        )

    def forward(
        self,
        images: torch.Tensor,
        graphs: Batch,
        clinicals: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass through all modalities and heads.
        Returns dictionary containing:
          - class_logits: [B, 4]
          - detection_logits: [B, 2]
          - grade_logits: [B, 4]
          - cnn_emb: [B, cnn_dim]
          - gnn_emb: [B, gnn_dim]
          - fused_emb: [B, fused_dim]
          - node_weights: [Total_Nodes, 1]
          - gate_weights: Dict of modality weights
        """
        # 1. Extract CNN embeddings and spatial feature maps
        h_cnn, fmap = self.cnn(images)

        # 2. Extract GNN topological embeddings and node importance
        h_gnn, node_weights = self.gnn(graphs.x, graphs.edge_index, graphs.batch)

        # 3. Extract Clinical tabular embeddings
        h_clin = self.clinical_encoder(clinicals)

        # 4. Multimodal Fusion
        z_fused, gate_weights = self.fusion(h_cnn, h_gnn, h_clin)

        # 5. Multi-Task Predictions
        detection_logits = self.detection_head(z_fused)
        class_logits = self.classification_head(z_fused)
        grade_logits = self.grade_head(z_fused)

        return {
            "class_logits": class_logits,
            "detection_logits": detection_logits,
            "grade_logits": grade_logits,
            "cnn_emb": h_cnn,
            "gnn_emb": h_gnn,
            "fused_emb": z_fused,
            "feature_map": fmap,
            "node_weights": node_weights,
            "gate_weights": gate_weights
        }


if __name__ == "__main__":
    from src.dataset.graph_utils import build_grid_graph
    model = MultimodalBrainTumorNet(pretrained_cnn=False)
    dummy_imgs = torch.randn(2, 3, 224, 224)
    g1 = build_grid_graph(dummy_imgs[0])
    g2 = build_grid_graph(dummy_imgs[1])
    batch_graphs = Batch.from_data_list([g1, g2])
    dummy_clin = torch.randn(2, 8)

    outputs = model(dummy_imgs, batch_graphs, dummy_clin)
    print("Multi-Class Logits Shape:", outputs["class_logits"].shape)
    print("Detection Logits Shape:", outputs["detection_logits"].shape)
    print("Grade Logits Shape:", outputs["grade_logits"].shape)
    print("Fused Multimodal Embedding Shape:", outputs["fused_emb"].shape)
