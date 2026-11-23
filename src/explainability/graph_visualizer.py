"""
Graph Visualizer for Brain Tumor GNN Tissue Topology.
Overlays the Superpixel Region Adjacency Graph and GNN Attention on MRI scans.
"""
import torch
import numpy as np
import cv2
import matplotlib.cm as cm
from typing import Tuple, Optional


def draw_gnn_graph_overlay(
    image_tensor: torch.Tensor,
    node_weights: torch.Tensor,
    grid_size: Tuple[int, int] = (7, 7),
    edge_index: Optional[torch.Tensor] = None
) -> np.ndarray:
    """
    Overlays the GNN topological graph on top of the MRI scan.
    Args:
      image_tensor: [C, H, W] normalized image tensor
      node_weights: [49, 1] or [49] GNN attention / importance scores
      grid_size: (7, 7)
      edge_index: [2, num_edges]
    Returns:
      overlay_rgb: [H, W, 3] uint8 image with graph nodes & edges overlaid
    """
    # 1. Unnormalize image to RGB uint8
    img_np = image_tensor.detach().cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_rgb = np.clip((img_np * std + mean) * 255.0, 0, 255).astype(np.uint8).copy()

    H, W, _ = img_rgb.shape
    rows, cols = grid_size
    patch_h = H // rows
    patch_w = W // cols

    # Node coordinates
    node_coords = []
    for r in range(rows):
        for c in range(cols):
            cy = int((r + 0.5) * patch_h)
            cx = int((c + 0.5) * patch_w)
            node_coords.append((cx, cy))

    # Normalize node weights to [0, 1]
    weights_np = node_weights.detach().cpu().squeeze().numpy()
    if weights_np.ndim > 1:
        weights_np = weights_np[:rows*cols]  # Take first graph in batch
    w_min, w_max = np.min(weights_np), np.max(weights_np)
    if w_max > w_min:
        norm_weights = (weights_np - w_min) / (w_max - w_min)
    else:
        norm_weights = np.ones_like(weights_np) * 0.5

    # 2. Draw graph edges (with subtle opacity)
    overlay = img_rgb.copy()
    if edge_index is not None:
        edges = edge_index.detach().cpu().numpy()
        for i in range(edges.shape[1]):
            src, dst = edges[0, i], edges[1, i]
            if src < len(node_coords) and dst < len(node_coords):
                pt1 = node_coords[src]
                pt2 = node_coords[dst]
                # Edge alpha depends on mutual node importance
                edge_imp = (norm_weights[src] + norm_weights[dst]) / 2.0
                if edge_imp > 0.4:
                    cv2.line(overlay, pt1, pt2, (0, 255, 255), 1, cv2.LINE_AA)

    # Blend edges
    img_rgb = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)

    # 3. Draw nodes with colormap according to GNN attention
    try:
        import matplotlib
        plasma_cmap = matplotlib.colormaps['plasma']
    except Exception:
        plasma_cmap = cm.get_cmap("plasma") if hasattr(cm, "get_cmap") else lambda v: (v, 0.2, 1.0 - v, 1.0)
    for idx, (cx, cy) in enumerate(node_coords):
        val = norm_weights[idx]
        rgba = plasma_cmap(val)
        color = (int(rgba[0] * 255), int(rgba[1] * 255), int(rgba[2] * 255))
        radius = int(3 + val * 6)  # Larger radius for high-attention nodes

        # Draw outer circle and glowing center
        cv2.circle(img_rgb, (cx, cy), radius + 1, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(img_rgb, (cx, cy), radius, color, -1, cv2.LINE_AA)

    return img_rgb
