"""
Clinical Tabular MLP Encoder for Biomarkers, Symptoms, and Patient Demographics.
Projects 8-dimensional normalized clinical records into high-dimensional latent space.
"""
import torch
import torch.nn as nn


class ClinicalMLP(nn.Module):
    """Encodes patient demographic, symptom severity, and molecular biomarker vectors."""
    def __init__(self, in_features: int = 8, hidden_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.1, inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
          x: [B, in_features]
        Returns:
          [B, out_dim]
        """
        return self.net(x)


if __name__ == "__main__":
    mlp = ClinicalMLP(in_features=8, hidden_dim=64, out_dim=64)
    dummy_clin = torch.randn(4, 8)
    out = mlp(dummy_clin)
    print("Clinical MLP output shape:", out.shape)
