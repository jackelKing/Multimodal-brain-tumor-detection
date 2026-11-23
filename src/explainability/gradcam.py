"""
Grad-CAM (Gradient-Weighted Class Activation Mapping) for Brain Tumor CNN Feature Maps.
Provides spatial visual localization of tumor boundaries and hot zones.
"""
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple, Optional


class GradCAM:
    """Computes Grad-CAM heatmaps for the convolutional backbone."""
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        # Attach to layer4 of CNN
        target_layer = self.model.cnn.layer4
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(
        self,
        images: torch.Tensor,
        graphs,
        clinicals: torch.Tensor,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Computes Grad-CAM heatmap for the input batch.
        Returns:
          - heatmap: normalized 2D numpy array [H, W] in [0, 1]
          - overlay: RGB uint8 image [H, W, 3] with jet colormap blended on original MRI
          - pred_class: predicted class index
        """
        self.model.eval()
        self.model.zero_grad()

        # Forward pass
        outputs = self.model(images, graphs, clinicals)
        class_logits = outputs["class_logits"]

        if target_class is None:
            target_class = int(class_logits.argmax(dim=1)[0].item())

        # Backward pass on target class score
        score = class_logits[0, target_class]
        score.backward(retain_graph=True)

        # Compute channel weights via global average pooling of gradients
        # gradients shape: [1, 512, 7, 7]
        # activations shape: [1, 512, 7, 7]
        grads = self.gradients[0]  # [512, 7, 7]
        acts = self.activations[0]  # [512, 7, 7]

        weights = torch.mean(grads, dim=(1, 2), keepdim=True)  # [512, 1, 1]
        cam = torch.sum(weights * acts, dim=0)  # [7, 7]
        cam = F.relu(cam)  # Apply ReLU

        # Normalize to [0, 1]
        cam = cam.detach().cpu().numpy()
        cam_min, cam_max = np.min(cam), np.max(cam)
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        # Upsample heatmap to original image dimensions (224, 224)
        heatmap_resized = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)

        # Create blended overlay with original MRI
        img_np = images[0].detach().cpu().permute(1, 2, 0).numpy()
        # Unnormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_rgb = np.clip((img_np * std + mean) * 255.0, 0, 255).astype(np.uint8)

        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        color_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        color_heatmap = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(img_rgb, 0.65, color_heatmap, 0.35, 0)

        return heatmap_resized, overlay, target_class
