"""
High-Speed Vectorized Graph Construction Utilities for Brain Tumor MRI Scans.
Extracts 8-Connected Superpixel Region Adjacency Graphs (RAG) for GNN processing.
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional
from torch_geometric.data import Data

# Precomputed static 7x7 8-connected grid edges
def _compute_static_grid_edges(rows: int = 7, cols: int = 7):
    edge_sources = []
    edge_targets = []
    edge_weights = []

    for r in range(rows):
        for c in range(cols):
            src_idx = r * cols + c
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        dst_idx = nr * cols + nc
                        edge_sources.append(src_idx)
                        edge_targets.append(dst_idx)
                        dist = (dr**2 + dc**2) ** 0.5
                        edge_weights.append(1.0 / dist)

    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(-1)
    
    positions = []
    for r in range(rows):
        for c in range(cols):
            positions.append([(c + 0.5) / cols, (r + 0.5) / rows])
    pos = torch.tensor(positions, dtype=torch.float32)

    return edge_index, edge_attr, pos

STATIC_EDGE_INDEX, STATIC_EDGE_ATTR, STATIC_POS = _compute_static_grid_edges(7, 7)


def build_grid_graph(
    image_tensor: torch.Tensor,
    grid_size: Tuple[int, int] = (7, 7),
    feature_dim: int = 64
) -> Data:
    """
    Fast vectorized Region Adjacency Graph from an MRI image tensor.
    image_tensor: [C, H, W] (e.g. [3, 224, 224])
    grid_size: (7, 7) -> 49 nodes
    """
    C, H, W = image_tensor.shape
    rows, cols = grid_size
    num_nodes = rows * cols

    patch_h = H // rows
    patch_w = W // cols

    # Unfold image into 7x7 patches
    # image_tensor: [C, 224, 224]
    if C == 3:
        # Convert to grayscale luminance: 0.2989 R + 0.5870 G + 0.1140 B
        gray = 0.2989 * image_tensor[0] + 0.5870 * image_tensor[1] + 0.1140 * image_tensor[2]
    else:
        gray = image_tensor[0]

    # Unfold grayscale into patches [7, 7, 32, 32]
    patches = gray.unfold(0, patch_h, patch_h).unfold(1, patch_w, patch_w) # [7, 7, patch_h, patch_w]
    patches = patches.contiguous().view(num_nodes, -1) # [49, patch_h * patch_w]

    # Vectorized statistical descriptors
    means = patches.mean(dim=-1, keepdim=True) # [49, 1]
    stds = patches.std(dim=-1, keepdim=True)   # [49, 1]
    maxs = patches.max(dim=-1, keepdim=True)[0] # [49, 1]
    mins = patches.min(dim=-1, keepdim=True)[0] # [49, 1]
    contrasts = maxs - mins                     # [49, 1]

    # Resized mini-patch texture (7x7 mini descriptor = 49 features)
    # Downsample patches to [49, 7, 7]
    patches_2d = patches.view(num_nodes, 1, patch_h, patch_w)
    mini_textures = F.adaptive_avg_pool2d(patches_2d, (7, 7)).view(num_nodes, 49) # [49, 49]

    # Normalized spatial coordinates from static tensor
    pos = STATIC_POS.to(image_tensor.device) # [49, 2]

    # Concatenate: [means(1), stds(1), maxs(1), mins(1), contrasts(1), pos(2)] = 7 features
    stat_feats = torch.cat([means, stds, maxs, mins, contrasts, pos], dim=-1) # [49, 7]
    
    # Combined vector [49, 64]
    # 7 stat features + 49 mini-texture features + 8 zeros/padding = 64
    x = torch.zeros(num_nodes, feature_dim, dtype=torch.float32, device=image_tensor.device)
    x[:, :7] = stat_feats
    x[:, 7:7+49] = mini_textures

    edge_index = STATIC_EDGE_INDEX.to(image_tensor.device)
    edge_attr = STATIC_EDGE_ATTR.to(image_tensor.device)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)
