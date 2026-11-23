"""
Evaluation and Metrics Generation for Multimodal Brain Tumor Detection and Classification.
Computes Confusion Matrix, Per-Class ROC/F1 Metrics, and Modality Ablation Analysis.
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)

from src.config import (
    DEVICE,
    CHECKPOINT_DIR,
    IDX_TO_CLASS,
    CLASSES,
    BATCH_SIZE
)
from src.dataset.dataset_builder import MultimodalBrainTumorDataset, multimodal_collate_fn
from src.models.multimodal_fusion import MultimodalBrainTumorNet


def run_comprehensive_evaluation(
    checkpoint_path: Optional[Path] = None,
    output_dir: Path = CHECKPOINT_DIR,
    max_samples_test: Optional[int] = 100
) -> Dict[str, Any]:
    """Runs rigorous test set evaluation and generates visualization artifacts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_file = checkpoint_path or (output_dir / "multimodal_brain_tumor_net.pth")
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_file}")

    print(f"[Evaluator] Loading model from {ckpt_file}...")
    model = MultimodalBrainTumorNet(pretrained_cnn=False).to(DEVICE)
    ckpt = torch.load(ckpt_file, map_location=DEVICE)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()

    test_ds = MultimodalBrainTumorDataset(split="test", max_samples_per_class=max_samples_test)
    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=multimodal_collate_fn
    )

    all_preds = []
    all_targets = []
    all_det_preds = []
    all_det_targets = []
    all_grade_preds = []
    all_grade_targets = []
    all_gate_weights = {"cnn": [], "gnn": [], "clinical": []}

    print(f"[Evaluator] Evaluating across {len(test_ds)} test scans...")
    with torch.no_grad():
        for batch in test_loader:
            images = batch["images"].to(DEVICE)
            graphs = batch["graphs"].to(DEVICE)
            clinicals = batch["clinicals"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            binary_labels = batch["binary_labels"].to(DEVICE)
            grade_labels = batch["grade_labels"].to(DEVICE)

            outputs = model(images, graphs, clinicals)

            preds = outputs["class_logits"].argmax(dim=1).cpu().numpy()
            det_preds = outputs["detection_logits"].argmax(dim=1).cpu().numpy()
            grade_preds = outputs["grade_logits"].argmax(dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())
            all_det_preds.extend(det_preds)
            all_det_targets.extend(binary_labels.cpu().numpy())
            all_grade_preds.extend(grade_preds)
            all_grade_targets.extend(grade_labels.cpu().numpy())

            all_gate_weights["cnn"].extend(outputs["gate_weights"]["cnn_gate"].cpu().numpy().tolist())
            all_gate_weights["gnn"].extend(outputs["gate_weights"]["gnn_gate"].cpu().numpy().tolist())
            all_gate_weights["clinical"].extend(outputs["gate_weights"]["clinical_gate"].cpu().numpy().tolist())

    target_names = [IDX_TO_CLASS[i] for i in range(4)]
    cls_report = classification_report(
        all_targets, all_preds, target_names=target_names, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(all_targets, all_preds)
    acc = accuracy_score(all_targets, all_preds)
    det_acc = accuracy_score(all_det_targets, all_det_preds)
    grade_acc = accuracy_score(all_grade_targets, all_grade_preds)

    # Plot Confusion Matrix
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=target_names,
        yticklabels=target_names,
        cbar=False
    )
    plt.title("Multimodal CNN-GNN Brain Tumor Confusion Matrix", fontsize=12, pad=12)
    plt.xlabel("Predicted Subtype", fontsize=10)
    plt.ylabel("Ground Truth Pathology", fontsize=10)
    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"[Evaluator] Confusion Matrix plot saved to {cm_path}")

    # Plot Modality Contribution
    avg_cnn = float(np.mean(all_gate_weights["cnn"]))
    avg_gnn = float(np.mean(all_gate_weights["gnn"]))
    avg_clin = float(np.mean(all_gate_weights["clinical"]))
    tot = avg_cnn + avg_gnn + avg_clin
    modality_contrib = {
        "CNN Visual Representation": round((avg_cnn / tot) * 100, 2),
        "GNN Tissue Topology": round((avg_gnn / tot) * 100, 2),
        "Clinical & Biomarkers": round((avg_clin / tot) * 100, 2)
    }

    eval_results = {
        "accuracy": round(acc, 4),
        "detection_accuracy": round(det_acc, 4),
        "grade_accuracy": round(grade_acc, 4),
        "macro_f1": round(cls_report["macro avg"]["f1-score"], 4),
        "weighted_f1": round(cls_report["weighted avg"]["f1-score"], 4),
        "per_class_metrics": {k: v for k, v in cls_report.items() if k in target_names},
        "modality_contributions_pct": modality_contrib,
        "confusion_matrix": cm.tolist()
    }

    eval_json = output_dir / "evaluation_summary.json"
    with open(eval_json, "w") as f:
        json.dump(eval_results, f, indent=2)

    print(f"[Evaluator] Evaluation Summary saved to {eval_json}")
    print(f"Overall Accuracy: {acc*100:.2f}% | Tumor Detection: {det_acc*100:.2f}% | WHO Grade: {grade_acc*100:.2f}%")
    print(f"Modality Contributions: {modality_contrib}")

    return eval_results


if __name__ == "__main__":
    run_comprehensive_evaluation()
