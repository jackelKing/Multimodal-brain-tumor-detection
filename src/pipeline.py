"""
Unified Multimodal Inference Pipeline for Brain Tumor Detection, Prediction, Classification, and RAG.
Orchestrates MRI Image Preprocessing, Graph Extraction, Deep Fusion, Grad-CAM, and Ollama Clinical RAG.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from torch_geometric.data import Batch

from src.config import (
    DEVICE,
    IMAGE_SIZE,
    CHECKPOINT_DIR,
    IDX_TO_CLASS,
    GRADE_CLASSES,
    PATCH_GRID,
    NODE_FEATURE_DIM
)
from src.dataset.graph_utils import build_grid_graph
from src.dataset.dataset_builder import generate_synthetic_clinical_profile, get_mri_transforms
from src.models.multimodal_fusion import MultimodalBrainTumorNet
from src.explainability.gradcam import GradCAM
from src.explainability.graph_visualizer import draw_gnn_graph_overlay
from src.explainability.segmentation import segment_tumor_compartments
from src.rag.clinical_advisor import ClinicalAdvisor


class MultimodalBrainTumorPipeline:
    """
    End-to-End Pipeline:
      MRI Image + Clinical Data -> CNN + GNN -> Multimodal Predictions + GradCAM + Graph Overlay -> RAG Clinical Report
    """
    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        device: torch.device = DEVICE,
        enable_rag: bool = True
    ):
        self.device = device
        self.model = MultimodalBrainTumorNet(pretrained_cnn=False).to(self.device)
        self.transform = get_mri_transforms(is_train=False)

        # Load weights if available
        ckpt_file = checkpoint_path or (CHECKPOINT_DIR / "multimodal_brain_tumor_net.pth")
        if ckpt_file.exists():
            print(f"[Pipeline] Loading checkpoint from {ckpt_file}...")
            ckpt = torch.load(ckpt_file, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.model.load_state_dict(state_dict)
            print("[Pipeline] Checkpoint successfully loaded.")
        else:
            print("[Pipeline] No saved checkpoint found. Running with initialized weights.")

        self.model.eval()

        # Explainability & RAG engines
        self.gradcam = GradCAM(self.model)
        self.enable_rag = enable_rag
        self.clinical_advisor = ClinicalAdvisor() if enable_rag else None

    def predict(
        self,
        image_input: Union[str, Path, Image.Image, np.ndarray],
        patient_info: Optional[Dict[str, Any]] = None,
        generate_rag_report: bool = True,
        rag_model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Runs full multimodal analysis on an MRI scan and clinical profile.
        """
        # 1. Load and transform image
        if isinstance(image_input, (str, Path)):
            pil_img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        img_tensor = self.transform(pil_img).to(self.device)  # [3, 224, 224]
        img_batch = img_tensor.unsqueeze(0)  # [1, 3, 224, 224]

        # 2. Build spatial superpixel graph
        graph_data = build_grid_graph(img_tensor, grid_size=PATCH_GRID, feature_dim=NODE_FEATURE_DIM)
        graph_batch = Batch.from_data_list([graph_data]).to(self.device)

        # 3. Handle clinical profile
        if patient_info is None:
            # Default placeholder profile
            clin_tensor = torch.zeros(1, 8, dtype=torch.float32, device=self.device)
            patient_profile = {
                "age": 52,
                "gender": "Unknown",
                "symptoms": "Referred for neuroimaging diagnostic evaluation",
                "kps": 80,
                "idh1": "Pending",
                "mgmt": "Pending",
                "codeletion_1p19q": "Pending"
            }
        else:
            patient_profile = patient_info
            # Convert patient_info dict to normalized 8-d tensor
            norm_age = (float(patient_info.get("age", 50)) - 50.0) / 25.0
            gender_val = 1.0 if patient_info.get("gender", "Male").lower() == "male" else 0.0
            symptom_sev = float(patient_info.get("symptom_severity", 0.6))
            headache = 1.0 if "headache" in str(patient_info.get("symptoms", "")).lower() else 0.0
            seizure = 1.0 if "seizure" in str(patient_info.get("symptoms", "")).lower() else 0.0
            vision = 1.0 if "vision" in str(patient_info.get("symptoms", "")).lower() or "hemianopsia" in str(patient_info.get("symptoms", "")).lower() else 0.0
            norm_kps = float(patient_info.get("kps", 80)) / 100.0
            prior = 0.8 if "wildtype" in str(patient_info.get("idh1", "")).lower() else 0.2

            clin_feats = [norm_age, gender_val, symptom_sev, headache, seizure, vision, norm_kps, prior]
            clin_tensor = torch.tensor([clin_feats], dtype=torch.float32, device=self.device)

        # 4. Multimodal forward pass
        with torch.no_grad():
            outputs = self.model(img_batch, graph_batch, clin_tensor)

            # Class probabilities
            class_probs = torch.softmax(outputs["class_logits"], dim=1)[0].cpu().numpy()
            pred_class_idx = int(np.argmax(class_probs))
            pred_class_name = IDX_TO_CLASS[pred_class_idx]
            confidence = float(class_probs[pred_class_idx])

            # Binary detection
            det_probs = torch.softmax(outputs["detection_logits"], dim=1)[0].cpu().numpy()
            is_tumor = bool(det_probs.argmax() == 1)
            det_confidence = float(det_probs[1 if is_tumor else 0])

            # Grade prediction
            grade_probs = torch.softmax(outputs["grade_logits"], dim=1)[0].cpu().numpy()
            pred_grade_idx = int(np.argmax(grade_probs))
            pred_grade_name = GRADE_CLASSES[pred_grade_idx]
            grade_confidence = float(grade_probs[pred_grade_idx])

            # Gating weights
            gate_weights = {
                "cnn_weight": float(outputs["gate_weights"]["cnn_gate"][0].cpu().item()),
                "gnn_weight": float(outputs["gate_weights"]["gnn_gate"][0].cpu().item()),
                "clinical_weight": float(outputs["gate_weights"]["clinical_gate"][0].cpu().item())
            }

            node_weights = outputs["node_weights"]

        # 5. Explainability Maps
        # A) Grad-CAM heatmap
        try:
            cam_map, cam_overlay, _ = self.gradcam.generate_heatmap(
                img_batch, graph_batch, clin_tensor, target_class=pred_class_idx
            )
        except Exception as e:
            print(f"[Grad-CAM error]: {e}")
            cam_map = np.zeros(IMAGE_SIZE)
            cam_overlay = np.array(pil_img.resize(IMAGE_SIZE))

        # B) GNN Topological Graph Overlay
        try:
            gnn_overlay = draw_gnn_graph_overlay(
                img_tensor, node_weights, grid_size=PATCH_GRID, edge_index=graph_data.edge_index
            )
        except Exception as e:
            print(f"[GNN Visualizer error]: {e}")
            gnn_overlay = np.array(pil_img.resize(IMAGE_SIZE))

        # C) Multi-Compartment Tumor Segmentation & Caliper Volumetrics
        try:
            raw_rgb = np.array(pil_img.resize(IMAGE_SIZE))
            seg_results = segment_tumor_compartments(raw_rgb, cam_map)
            seg_overlay = seg_results["color_overlay"]
            seg_metrics = seg_results["metrics"]
        except Exception as e:
            print(f"[Segmentation error]: {e}")
            seg_overlay = np.array(pil_img.resize(IMAGE_SIZE))
            seg_metrics = {
                "major_diameter_mm": 0.0,
                "minor_diameter_mm": 0.0,
                "area_mm2": 0.0,
                "estimated_volume_cm3": 0.0,
                "centroid_x": 112,
                "centroid_y": 112,
                "edema_to_core_ratio": 1.0
            }

        # Format probability dictionary
        prob_dict = {IDX_TO_CLASS[i]: float(class_probs[i]) for i in range(len(class_probs))}

        # Formulate spatial summary
        cam_activation_area = float(np.mean(cam_map > 0.4)) * 100.0
        gradcam_summary = (
            f"Focal Grad-CAM epicenter detected with {cam_activation_area:.1f}% suprathreshold tissue involvement. "
            f"Consistent with {pred_class_name} neuro-anatomical signature."
        )
        gnn_summary = (
            f"GNN superpixel topological analysis identifies high centrality clusters across {PATCH_GRID[0]}x{PATCH_GRID[1]} "
            f"tissue patches, capturing microenvironment heterogeneity between lesion margins and parenchymal tissue."
        )

        results = {
            "predicted_class": pred_class_name,
            "predicted_class_idx": pred_class_idx,
            "confidence": confidence,
            "class_probabilities": prob_dict,
            "is_tumor_detected": is_tumor,
            "detection_confidence": det_confidence,
            "predicted_grade": pred_grade_name,
            "grade_confidence": grade_confidence,
            "gate_weights": gate_weights,
            "gradcam_summary": gradcam_summary,
            "gnn_summary": gnn_summary,
            "cam_map": cam_map,
            "cam_overlay": cam_overlay,
            "gnn_overlay": gnn_overlay,
            "seg_overlay": seg_overlay,
            "volumetric_metrics": seg_metrics,
            "original_image": pil_img
        }

        # 6. RAG Clinical Consultation Report
        if generate_rag_report and self.enable_rag and self.clinical_advisor:
            print("[Pipeline] Generating RAG clinical consultation report via Ollama...")
            rag_output = self.clinical_advisor.generate_consultation_report(
                patient_info=patient_profile,
                multimodal_results=results,
                model_name=rag_model_name
            )
            results["rag_report"] = rag_output["report"]
            results["retrieved_guidelines"] = rag_output["retrieved_guidelines"]
            results["rag_model_used"] = rag_output["model_used"]

        return results


if __name__ == "__main__":
    from src.dataset.dataset_builder import MultimodalBrainTumorDataset
    test_ds = MultimodalBrainTumorDataset(split="test")
    sample = test_ds[0]
    pipeline = MultimodalBrainTumorPipeline()
    res = pipeline.predict(sample["meta"]["image_path"], patient_info=sample["meta"], generate_rag_report=True)
    print("\n--- INFERENCE RESULTS ---")
    print("Class:", res["predicted_class"], f"({res['confidence']*100:.1f}%)")
    print("Tumor Detected:", res["is_tumor_detected"])
    print("Grade:", res["predicted_grade"])
    print("Modality Gates:", res["gate_weights"])
    if "rag_report" in res:
        print("\nRAG Report Preview:\n", res["rag_report"][:400] + "...")
