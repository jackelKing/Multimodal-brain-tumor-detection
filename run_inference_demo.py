"""
Standalone End-to-End Inference Demonstration.
Executes Multimodal CNN-GNN detection, classification, WHO grade prediction,
Grad-CAM heatmap generation, GNN topological graph overlay, and Ollama RAG clinical report.
"""
import os
import sys
from pathlib import Path
from PIL import Image
import numpy as np

from src.pipeline import MultimodalBrainTumorPipeline
from src.dataset.dataset_builder import MultimodalBrainTumorDataset
from src.config import CHECKPOINT_DIR


def run_demo():
    print("=" * 75)
    print(" MULTIMODAL BRAIN TUMOR DETECTION, PREDICTION & CLASSIFICATION ")
    print(" Integrating CNN, GNN, and RAG Clinical Decision Support ")
    print("=" * 75)

    # 1. Initialize pipeline
    print("\n[Step 1] Initializing Multimodal CNN-GNN Pipeline & Ollama RAG Engine...")
    pipeline = MultimodalBrainTumorPipeline(enable_rag=True)

    # 2. Select a representative test patient sample
    print("\n[Step 2] Loading a clinical test sample from the dataset...")
    test_ds = MultimodalBrainTumorDataset(split="test")
    # Pick a glioma or high grade tumor case
    sample = None
    for i in range(len(test_ds)):
        item = test_ds[i]
        if item["meta"]["class_name"] == "Glioma":
            sample = item
            break
    if sample is None:
        sample = test_ds[0]

    img_path = sample["meta"]["image_path"]
    patient_info = sample["meta"]

    print(f"\n--- PATIENT PROFILE ---")
    print(f"Age: {patient_info['age']} | Gender: {patient_info['gender']}")
    print(f"Symptoms: {patient_info['symptoms']}")
    print(f"KPS Score: {patient_info['kps']}/100")
    print(f"IDH1 Status: {patient_info['idh1']} | MGMT: {patient_info['mgmt']}")
    print(f"Ground Truth Pathology: {patient_info['class_name']}")
    print(f"MRI Image: {Path(img_path).name}")

    # 3. Execute Multimodal Inference
    print("\n[Step 3] Executing Multimodal Inference (CNN Visual + GNN Graph + Clinical Prior)...")
    results = pipeline.predict(
        image_input=img_path,
        patient_info=patient_info,
        generate_rag_report=True,
        rag_model_name="llama3.2:latest"
    )

    print("\n--- MULTIMODAL DIAGNOSTIC FINDINGS ---")
    print(f"Primary Diagnosis:      {results['predicted_class'].upper()} (Confidence: {results['confidence']*100:.2f}%)")
    print(f"Tumor Detection Status: {'[+] TUMOR DETECTED' if results['is_tumor_detected'] else '[-] NORMAL BRAIN'} (Confidence: {results['detection_confidence']*100:.2f}%)")
    print(f"Prognostic WHO Grade:   {results['predicted_grade']} (Confidence: {results['grade_confidence']*100:.2f}%)")

    print("\nClass Probabilities:")
    for cls_name, prob in results["class_probabilities"].items():
        bar = "█" * int(prob * 30)
        print(f"  {cls_name:<22}: {prob*100:5.1f}% {bar}")

    print("\nMultimodal Gating Weights:")
    gw = results["gate_weights"]
    tot = gw["cnn_weight"] + gw["gnn_weight"] + gw["clinical_weight"]
    print(f"  CNN Visual Texture:   {(gw['cnn_weight']/tot)*100:.1f}%")
    print(f"  GNN Tissue Topology:  {(gw['gnn_weight']/tot)*100:.1f}%")
    print(f"  Clinical Biomarkers:  {(gw['clinical_weight']/tot)*100:.1f}%")

    # 4. Save Explainability Overlays
    print("\n[Step 4] Generating Visual Explainability Overlays...")
    gradcam_save_path = CHECKPOINT_DIR / "demo_gradcam_overlay.png"
    gnn_save_path = CHECKPOINT_DIR / "demo_gnn_topology_overlay.png"

    Image.fromarray(results["cam_overlay"]).save(gradcam_save_path)
    Image.fromarray(results["gnn_overlay"]).save(gnn_save_path)
    print(f"  Saved Grad-CAM overlay to: {gradcam_save_path}")
    print(f"  Saved GNN graph overlay to: {gnn_save_path}")

    # 5. Display RAG Evidence & Clinical Consultation Report
    print("\n[Step 5] RAG Knowledge Base Retrieval & Ollama Report Generation...")
    print(f"Retrieved {len(results.get('retrieved_guidelines', []))} clinical guideline passages from ChromaDB:")
    for idx, guide in enumerate(results.get("retrieved_guidelines", []), 1):
        print(f"  [{idx}] {guide['title']} > {guide['header']} (Score: {guide['relevance_score']})")

    print("\n" + "=" * 75)
    print(" NEUROPATHOLOGY & ONCOLOGY CLINICAL CONSULTATION REPORT (via Ollama) ")
    print("=" * 75)
    print(results.get("rag_report", "No report generated."))
    print("=" * 75)
    print("\nDemo run successfully completed!")


if __name__ == "__main__":
    run_demo()
