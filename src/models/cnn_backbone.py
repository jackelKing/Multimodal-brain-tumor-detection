"""
Convolutional Neural Network Backbone for Brain Tumor MRI Feature Extraction.
Provides deep visual representation maps and Grad-CAM hooks.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Tuple, Optional


class ConvBlock(nn.Module):
    """Standard Convolutional Block with BatchNorm and LeakyReLU."""
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) + self.shortcut(x)


class BrainTumorCNN(nn.Module):
    """
    Residual CNN Backbone extracting spatial feature maps and global visual representations.
    Utilizes ResNet18 feature layers for robust convergence on medical scans.
    """
    def __init__(self, embedding_dim: int = 256, pretrained: bool = True):
        super().__init__()
        # Load ResNet18 backbone
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        base_resnet = models.resnet18(weights=weights)

        # Retain convolutional layers up to layer4
        self.initial_conv = nn.Sequential(
            base_resnet.conv1,
            base_resnet.bn1,
            base_resnet.relu,
            base_resnet.maxpool
        )
        self.layer1 = base_resnet.layer1  # 64 channels, 56x56
        self.layer2 = base_resnet.layer2  # 128 channels, 28x28
        self.layer3 = base_resnet.layer3  # 256 channels, 14x14
        self.layer4 = base_resnet.layer4  # 512 channels, 7x7

        # Projection head to embedding_dim
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.2)
        )

        # Storage for Grad-CAM gradients and feature activations
        self.gradients = None
        self.activations = None

    def _save_gradient(self, grad: torch.Tensor):
        self.gradients = grad

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          - global_embedding: [B, embedding_dim]
          - feature_map: [B, 512, 7, 7] (last conv layer before pooling)
        """
        x = self.initial_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat_map = self.layer4(x)  # [B, 512, 7, 7]

        # Grad-CAM hook
        if feat_map.requires_grad:
            feat_map.register_hook(self._save_gradient)
        self.activations = feat_map

        # Global representation
        pooled = self.global_pool(feat_map).flatten(1)  # [B, 512]
        embedding = self.proj(pooled)  # [B, embedding_dim]

        return embedding, feat_map


if __name__ == "__main__":
    model = BrainTumorCNN(embedding_dim=256, pretrained=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    emb, fmap = model(dummy_input)
    print("CNN Output Embedding:", emb.shape)
    print("CNN Spatial Feature Map:", fmap.shape)
