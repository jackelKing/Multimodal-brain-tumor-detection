"""
Graph Construction Utilities for Brain Tumor MRI Scans.
Extracts Superpixel / Grid-Patch Region Adjacency Graphs (RAG) for GNN processing.
"""
import torch
import numpy as np
import cv2
from typing import Tuple, List, Optional
from torch_geometric.data import Data


def build_grid_graph(
    image_tensor: torch.Tensor,
    grid_size: Tuple[int, int] = (7, 7),
    feature_dim: int = 64
) -> Data:
    """
    Constructs a spatial 8-connected Region Adjacency Graph from an MRI image tensor.
    image_tensor: [C, H, W] (e.g. [3, 224, 224])
    grid_size: (rows, cols) of patches (e.g. (7, 7) -> 49 nodes)
    Returns: torch_geometric.data.Data with:
      - x: [num_nodes, node_features]
      - edge_index: [2, num_edges]
      - edge_attr: [num_edges, 1] (spatial + feature similarity weight)
      - pos: [num_nodes, 2] (normalized patch centroid coordinates)
    """
    C, H, W = image_tensor.shape
    rows, cols = grid_size
    num_nodes = rows * cols

    patch_h = H // rows
    patch_w = W // cols

    # Compute node features from local image statistics and raw pixel patches
    node_features = []
    positions = []

    # Convert image to numpy for intensity analysis
    img_np = image_tensor.detach().cpu().numpy() # [C, H, W]
    if C == 3:
        gray = 0.2989 * img_np[0] + 0.5870 * img_np[1] + 0.1140 * img_np[2]
    else:
        gray = img_np[0]

    for r in range(rows):
        for c in range(cols):
            y_start = r * patch_h
            y_end = (r + 1) * patch_h if r < rows - 1 else H
            x_start = c * patch_w
            x_end = (c + 1) * patch_w if c < cols - 1 else W

            patch = gray[y_start:y_end, x_start:x_end]
            
            # Statistical tissue texture features:
            mean_val = float(np.mean(patch))
            std_val = float(np.std(patch))
            max_val = float(np.max(patch))
            min_val = float(np.min(patch))
            p25 = float(np.percentile(patch, 25))
            p75 = float(np.percentile(patch, 75))
            contrast = max_val - min_val
            entropy = -float(np.sum(np.histogram(patch, bins=8, density=True)[0] * np.log(np.histogram(patch, bins=8, density=True)[0] + 1e-7)))

            # Normalized spatial coordinates
            norm_y = (r + 0.5) / rows
            norm_x = (c + 0.5) / cols
            positions.append([norm_x, norm_y])

            # Downsample patch to mini-descriptor
            mini_patch = cv2.resize(patch, (7, 7)).flatten() # 49 features
            
            # Combine statistical descriptors + spatial coords + mini-patch -> feature vector
            stat_feats = [mean_val, std_val, max_val, min_val, p25, p75, contrast, entropy, norm_x, norm_y]
            # Total len: 10 + 49 = 59; pad or project to feature_dim (64)
            feat_vec = np.zeros(feature_dim, dtype=np.float32)
            feat_vec[:10] = stat_feats
            feat_vec[10:10+len(mini_patch)] = mini_patch[:feature_dim-10]
            node_features.append(feat_vec)

    x = torch.tensor(np.array(node_features), dtype=torch.float32)
    pos = torch.tensor(np.array(positions), dtype=torch.float32)

    # Build 8-connected spatial adjacency edges
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
                        # Distance weight: 1.0 for orthogonal, 1.0/sqrt(2) for diagonal
                        dist = (dr**2 + dc**2) ** 0.5
                        edge_weights.append(1.0 / dist)

    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(-1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, pos=pos)


def build_graph_from_cnn_feature_map(
    feature_map: torch.Tensor,
    pos: Optional[torch.Tensor] = None
) -> Data:
    """
    Constructs a topological Graph from a deep CNN 2D spatial feature map [C, H', W'].
    Each spatial location (h, w) is a graph node with feature vector of length C.
    """
    C, H_prime, W_prime = feature_map.shape
    num_nodes = H_prime * W_prime

    # Flatten spatial dims to [num_nodes, C]
    x = feature_map.view(C, -1).t() # [H'*W', C]

    edge_sources = []
    edge_targets = []
    edge_weights = []

    for r in range(H_prime):
        for c in range(W_prime):
            src_idx = r * W_prime + c
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H_prime and 0 <= nc < W_prime:
                        dst_idx = nr * W_prime + nc
                        edge_sources.append(src_idx)
                        edge_targets.append(dst_idx)
                        dist = (dr**2 + dc**2) ** 0.5
                        edge_weights.append(1.0 / dist)

    edge_index = torch.tensor([edge_sources, edge_targets], dtype=torch.long, device=feature_map.device)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float32, device=feature_map.device).unsqueeze(-1)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
