"""
Cross-Modal Transformer Attention Fusion Network for Brain Tumor Diagnosis.
Fuses CNN visual features, GNN topological graph reasoning, and clinical biomarkers
via Multi-Head Cross-Modal Attention and Gated Bilinear Projections.
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


class CrossModalAttentionFusion(nn.Module):
    """
    Multi-Head Cross-Modal Attention Transformer Block.
    Tokens: [Visual CNN, Topological GNN, Clinical Prior].
    Allows reciprocal information exchange between neuroimaging textures,
    tissue connectivity graphs, and patient molecular biomarkers.
    """
    def __init__(self, cnn_dim: int, gnn_dim: int, clin_dim: int, embed_dim: int = 256, n_heads: int = 4):
        super().__init__()
        self.embed_dim = embed_dim

        # Project all modalities into unified token dimension
        self.proj_cnn = nn.Linear(cnn_dim, embed_dim)
        self.proj_gnn = nn.Linear(gnn_dim, embed_dim)
        self.proj_clin = nn.Linear(clin_dim, embed_dim)

        # Learnable modality position/type embeddings
        self.modality_embed = nn.Parameter(torch.randn(1, 3, embed_dim) * 0.02)

        # Multi-Head Cross-Modal Attention
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=n_heads, batch_first=True, dropout=0.1)
        self.ln1 = nn.LayerNorm(embed_dim)

        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        self.ln2 = nn.LayerNorm(embed_dim)

        # Gated bilinear fusion
        self.gate_cnn = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())
        self.gate_gnn = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())
        self.gate_clin = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.Sigmoid())

        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2)
        )

    def forward(
        self,
        h_cnn: torch.Tensor,
        h_gnn: torch.Tensor,
        h_clin: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B = h_cnn.shape[0]

        # 1. Project to common embedding dimension
        t_c = self.proj_cnn(h_cnn).unsqueeze(1)    # [B, 1, embed_dim]
        t_g = self.proj_gnn(h_gnn).unsqueeze(1)    # [B, 1, embed_dim]
        t_cl = self.proj_clin(h_clin).unsqueeze(1) # [B, 1, embed_dim]

        # 2. Token sequence: [B, 3, embed_dim]
        tokens = torch.cat([t_c, t_g, t_cl], dim=1) + self.modality_embed

        # 3. Cross-Modal Attention
        attn_out, _ = self.mha(tokens, tokens, tokens)
        tokens = self.ln1(tokens + attn_out)

        # 4. Feedforward
        ffn_out = self.ffn(tokens)
        tokens = self.ln2(tokens + ffn_out)

        # 5. Extract attended modality representations
        c_att = tokens[:, 0, :]
        g_att = tokens[:, 1, :]
        cl_att = tokens[:, 2, :]

        # Gating
        gc = self.gate_cnn(c_att)
        gg = self.gate_gnn(g_att)
        gcl = self.gate_clin(cl_att)

        mod_c = c_att * gc
        mod_g = g_att * gg
        mod_cl = cl_att * gcl

        fused = torch.cat([mod_c, mod_g, mod_cl], dim=1) # [B, embed_dim * 3]
        z_fused = self.out_proj(fused)                   # [B, embed_dim]

        gate_weights = {
            "cnn_gate": gc.mean(dim=1),
            "gnn_gate": gg.mean(dim=1),
            "clinical_gate": gcl.mean(dim=1)
        }

        return z_fused, gate_weights


class MultimodalBrainTumorNet(nn.Module):
    """
    Advanced Multimodal Neural Network combining:
      1. CNN (ResNet18) for high-resolution visual feature maps & Grad-CAM
      2. GNN (GATv2 Multi-Head) for topological tissue connectivity
      3. Clinical MLP for molecular biomarkers & neurological history
      4. Cross-Modal Transformer Attention Fusion
      5. Multi-Task Heads: Detection, 4-Class Classification, WHO Grade Prediction
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
        self.cnn = BrainTumorCNN(embedding_dim=cnn_dim, pretrained=pretrained_cnn)
        self.gnn = BrainTumorGNN(in_channels=64, hidden_dim=gnn_dim, out_dim=gnn_dim, heads=4)
        self.clinical_encoder = ClinicalMLP(in_features=CLINICAL_FEATURE_DIM, hidden_dim=clin_dim, out_dim=clin_dim)

        self.fusion = CrossModalAttentionFusion(cnn_dim, gnn_dim, clin_dim, embed_dim=fused_dim, n_heads=4)

        # Multi-Task Diagnostic Heads
        self.detection_head = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(64, 2)
        )

        self.classification_head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes)
        )

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
        h_cnn, fmap = self.cnn(images)
        h_gnn, node_weights = self.gnn(graphs.x, graphs.edge_index, graphs.batch)
        h_clin = self.clinical_encoder(clinicals)

        z_fused, gate_weights = self.fusion(h_cnn, h_gnn, h_clin)

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
