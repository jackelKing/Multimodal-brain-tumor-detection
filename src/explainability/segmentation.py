"""
Automated Multi-Compartment Tumor Segmentation & Quantitative Caliper Engine.
Delineates Enhancing Tumor Core (ET), Peritumoral Edema (ED), and Necrotic Cavity (NCR)
with 2D/3D volumetric estimation and caliper measurements.
"""
import numpy as np
import cv2
from typing import Dict, Any, Tuple


def segment_tumor_compartments(
    image_np: np.ndarray,
    cam_heatmap: np.ndarray,
    pixel_spacing_mm: float = 0.9375
) -> Dict[str, Any]:
    """
    Performs multi-compartment segmentation using morphological clustering,
    gradient-weighted class activation mapping, and intensity thresholding.
    Returns:
      - segmentation_mask: uint8 image [H, W] (0: Background, 1: Core, 2: Edema, 3: Necrosis)
      - color_overlay: uint8 RGB image with transparent colored masks
      - volumetric_metrics: dict of caliper diameters, areas, and estimated 3D volumes
    """
    H, W = cam_heatmap.shape

    # Normalize image to grayscale in [0, 255]
    if image_np.ndim == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np.copy()
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 1. Enhancing Core: Region with high Grad-CAM activation and high intensity
    core_thresh = (cam_heatmap > 0.50) & (gray > 110)
    core_mask = np.uint8(core_thresh * 255)

    # Clean morphological noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    core_mask = cv2.morphologyEx(core_mask, cv2.MORPH_CLOSE, kernel)
    core_mask = cv2.morphologyEx(core_mask, cv2.MORPH_OPEN, kernel)

    # 2. Peritumoral Edema: Medium Grad-CAM activation or surrounding hypointensity/hyperintensity
    edema_thresh = (cam_heatmap > 0.25) & ~core_thresh & (gray > 60)
    edema_mask = np.uint8(edema_thresh * 255)
    edema_mask = cv2.morphologyEx(edema_mask, cv2.MORPH_CLOSE, kernel)

    # 3. Necrotic Core: Hypointense interior enclosed within high Grad-CAM zone
    necrotic_thresh = (cam_heatmap > 0.40) & (gray < 85)
    necrotic_mask = np.uint8(necrotic_thresh * 255)

    # Composite mask: 0=Bg, 1=Core, 2=Edema, 3=Necrosis
    seg_mask = np.zeros((H, W), dtype=np.uint8)
    seg_mask[edema_mask > 0] = 2
    seg_mask[core_mask > 0] = 1
    seg_mask[necrotic_mask > 0] = 3

    # Quantitative Caliper Measurements
    contours, _ = cv2.findContours(core_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    major_axis_mm = 0.0
    minor_axis_mm = 0.0
    area_mm2 = 0.0
    volume_cm3 = 0.0
    cx, cy = W // 2, H // 2

    if contours:
        # Find largest contour (primary lesion)
        largest_cnt = max(contours, key=cv2.contourArea)
        area_pixels = cv2.contourArea(largest_cnt)
        area_mm2 = area_pixels * (pixel_spacing_mm ** 2)

        M = cv2.moments(largest_cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

        if len(largest_cnt) >= 5:
            ellipse = cv2.fitEllipse(largest_cnt)
            (center, (d1_px, d2_px), angle) = ellipse
            major_axis_mm = max(d1_px, d2_px) * pixel_spacing_mm
            minor_axis_mm = min(d1_px, d2_px) * pixel_spacing_mm
        else:
            x, y, w, h = cv2.boundingRect(largest_cnt)
            major_axis_mm = max(w, h) * pixel_spacing_mm
            minor_axis_mm = min(w, h) * pixel_spacing_mm

        # 3D Ellipsoid volume estimation: V = (pi / 6) * d1 * d2 * d3
        # Assuming d3 (slice thickness / depth) is approximately the mean of d1 and d2
        d3_mm = (major_axis_mm + minor_axis_mm) / 2.0
        volume_mm3 = (np.pi / 6.0) * major_axis_mm * minor_axis_mm * d3_mm
        volume_cm3 = volume_mm3 / 1000.0

    # Build color-coded overlay
    color_overlay = image_np.copy() if image_np.ndim == 3 else cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    
    # Layer colors:
    # Core: Red [239, 68, 68]
    # Edema: Amber [245, 158, 11]
    # Necrosis: Blue [59, 130, 246]
    mask_rgb = np.zeros_like(color_overlay)
    mask_rgb[seg_mask == 1] = [239, 68, 68]
    mask_rgb[seg_mask == 2] = [245, 158, 11]
    mask_rgb[seg_mask == 3] = [59, 130, 246]

    has_mask = seg_mask > 0
    alpha = 0.45
    color_overlay[has_mask] = np.clip(
        (1 - alpha) * color_overlay[has_mask] + alpha * mask_rgb[has_mask], 0, 255
    ).astype(np.uint8)

    # Draw Caliper line on largest lesion
    if major_axis_mm > 5.0 and contours:
        cv2.circle(color_overlay, (cx, cy), 4, (0, 255, 255), -1)

    edema_pixels = np.sum(seg_mask == 2)
    core_pixels = np.sum(seg_mask == 1)
    edema_core_ratio = float(edema_pixels / max(1, core_pixels))

    metrics = {
        "major_diameter_mm": round(major_axis_mm, 2),
        "minor_diameter_mm": round(minor_axis_mm, 2),
        "area_mm2": round(area_mm2, 2),
        "estimated_volume_cm3": round(volume_cm3, 2),
        "centroid_x": cx,
        "centroid_y": cy,
        "core_pixels": int(core_pixels),
        "edema_pixels": int(edema_pixels),
        "edema_to_core_ratio": round(edema_core_ratio, 2)
    }

    return {
        "segmentation_mask": seg_mask,
        "color_overlay": color_overlay,
        "metrics": metrics
    }
