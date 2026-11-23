"""
High-Accuracy Training Pipeline for Multimodal CNN-GNN Brain Tumor Model.
Uses Multi-Head Cross-Modal Transformer Fusion, Label Smoothing, and Cosine Annealing.
"""
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from src.config import (
    DEVICE,
    BATCH_SIZE,
    LEARNING_RATE,
    WEIGHT_DECAY,
    CHECKPOINT_DIR,
    IDX_TO_CLASS
)
from src.dataset.dataset_builder import MultimodalBrainTumorDataset, multimodal_collate_fn
from src.models.multimodal_fusion import MultimodalBrainTumorNet


class HighAccuracyMultimodalTrainer:
    """Trains the cross-modal transformer fusion network for superior accuracy."""
    def __init__(
        self,
        device: torch.device = DEVICE,
        checkpoint_dir: Path = CHECKPOINT_DIR
    ):
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model = MultimodalBrainTumorNet(pretrained_cnn=True).to(self.device)

        # High-accuracy loss functions with label smoothing
        self.criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.criterion_det = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.criterion_grade = nn.CrossEntropyLoss(label_smoothing=0.05)

        # Differential learning rates: smaller for pretrained CNN backbone, higher for GNN and Fusion
        cnn_params = list(self.model.cnn.parameters())
        other_params = [
            p for n, p in self.model.named_parameters() if not n.startswith("cnn.")
        ]

        self.optimizer = optim.AdamW([
            {"params": cnn_params, "lr": LEARNING_RATE * 0.5, "weight_decay": WEIGHT_DECAY},
            {"params": other_params, "lr": LEARNING_RATE * 1.5, "weight_decay": WEIGHT_DECAY}
        ])

    def train_epoch(self, dataloader: DataLoader, scheduler=None) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for batch in dataloader:
            images = batch["images"].to(self.device)
            graphs = batch["graphs"].to(self.device)
            clinicals = batch["clinicals"].to(self.device)
            labels = batch["labels"].to(self.device)
            binary_labels = batch["binary_labels"].to(self.device)
            grade_labels = batch["grade_labels"].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(images, graphs, clinicals)

            loss_cls = self.criterion_cls(outputs["class_logits"], labels)
            loss_det = self.criterion_det(outputs["detection_logits"], binary_labels)
            loss_grade = self.criterion_grade(outputs["grade_logits"], grade_labels)

            loss = loss_cls + 0.35 * loss_det + 0.35 * loss_grade
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.5)
            self.optimizer.step()

            total_loss += loss.item()
            preds = outputs["class_logits"].argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(labels.cpu().numpy())

        if scheduler:
            scheduler.step()

        acc = accuracy_score(all_targets, all_preds)
        return {"loss": total_loss / len(dataloader), "accuracy": acc}

    def evaluate(self, dataloader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_det_preds = []
        all_det_targets = []
        all_grade_preds = []
        all_grade_targets = []

        with torch.no_grad():
            for batch in dataloader:
                images = batch["images"].to(self.device)
                graphs = batch["graphs"].to(self.device)
                clinicals = batch["clinicals"].to(self.device)
                labels = batch["labels"].to(self.device)
                binary_labels = batch["binary_labels"].to(self.device)
                grade_labels = batch["grade_labels"].to(self.device)

                outputs = self.model(images, graphs, clinicals)

                loss_cls = self.criterion_cls(outputs["class_logits"], labels)
                loss_det = self.criterion_det(outputs["detection_logits"], binary_labels)
                loss_grade = self.criterion_grade(outputs["grade_logits"], grade_labels)
                loss = loss_cls + 0.35 * loss_det + 0.35 * loss_grade

                total_loss += loss.item()

                preds = outputs["class_logits"].argmax(dim=1).cpu().numpy()
                det_preds = outputs["detection_logits"].argmax(dim=1).cpu().numpy()
                grade_preds = outputs["grade_logits"].argmax(dim=1).cpu().numpy()

                all_preds.extend(preds)
                all_targets.extend(labels.cpu().numpy())
                all_det_preds.extend(det_preds)
                all_det_targets.extend(binary_labels.cpu().numpy())
                all_grade_preds.extend(grade_preds)
                all_grade_targets.extend(grade_labels.cpu().numpy())

        acc = accuracy_score(all_targets, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average="weighted", zero_division=0)
        det_acc = accuracy_score(all_det_targets, all_det_preds)
        grade_acc = accuracy_score(all_grade_targets, all_grade_preds)
        cm = confusion_matrix(all_targets, all_preds).tolist()

        return {
            "loss": total_loss / len(dataloader),
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "detection_accuracy": det_acc,
            "grade_accuracy": grade_acc,
            "confusion_matrix": cm
        }

    def train(
        self,
        num_epochs: int = 5,
        max_samples_train: Optional[int] = 500,
        max_samples_test: Optional[int] = 100,
        save_name: str = "multimodal_brain_tumor_net.pth"
    ) -> Dict[str, Any]:
        print(f"[HighAccuracyTrainer] Loading dataset on device: {self.device}...")
        train_ds = MultimodalBrainTumorDataset(split="train", max_samples_per_class=max_samples_train)
        test_ds = MultimodalBrainTumorDataset(split="test", max_samples_per_class=max_samples_test)

        train_loader = DataLoader(
            train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=multimodal_collate_fn
        )
        test_loader = DataLoader(
            test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=multimodal_collate_fn
        )

        scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)

        best_f1 = 0.0
        best_metrics = {}
        history = []

        print(f"[HighAccuracyTrainer] Training for {num_epochs} epochs ({len(train_ds)} train, {len(test_ds)} test)...")

        for epoch in range(1, num_epochs + 1):
            t0 = time.time()
            train_res = self.train_epoch(train_loader, scheduler)
            eval_res = self.evaluate(test_loader)
            elapsed = time.time() - t0

            print(
                f"Epoch [{epoch}/{num_epochs}] ({elapsed:.1f}s) | "
                f"Train Loss: {train_res['loss']:.4f}, Train Acc: {train_res['accuracy']*100:.1f}% | "
                f"Val Loss: {eval_res['loss']:.4f}, Val Acc: {eval_res['accuracy']*100:.1f}%, "
                f"Val F1: {eval_res['f1_score']:.4f}, Det Acc: {eval_res['detection_accuracy']*100:.1f}%, Grade Acc: {eval_res['grade_accuracy']*100:.1f}%"
            )

            history.append({
                "epoch": epoch,
                "train_loss": train_res["loss"],
                "train_acc": train_res["accuracy"],
                "val_loss": eval_res["loss"],
                "val_acc": eval_res["accuracy"],
                "val_f1": eval_res["f1_score"]
            })

            if eval_res["f1_score"] > best_f1:
                best_f1 = eval_res["f1_score"]
                best_metrics = eval_res
                save_path = self.checkpoint_dir / save_name
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "epoch": epoch,
                    "metrics": eval_res,
                    "history": history
                }, save_path)
                print(f" -> Checkpoint saved with Best Val F1: {best_f1:.4f} to {save_path.name}")

        metrics_file = self.checkpoint_dir / "evaluation_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump({
                "best_metrics": best_metrics,
                "history": history,
                "classes": [IDX_TO_CLASS[i] for i in range(4)]
            }, f, indent=2)

        return best_metrics


if __name__ == "__main__":
    trainer = HighAccuracyMultimodalTrainer()
    trainer.train(num_epochs=5, max_samples_train=400, max_samples_test=100)
