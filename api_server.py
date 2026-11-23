"""
FastAPI Server for Multimodal Brain Tumor Detection, Prediction, Classification & RAG.
Provides RESTful APIs for the Advanced React Medical Frontend.
"""
import os
import io
import json
import base64
import math
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np

from src.config import (
    CHECKPOINT_DIR,
    IDX_TO_CLASS,
    GRADE_CLASSES,
    OLLAMA_BASE_URL,
    OLLAMA_CHAT_MODEL,
    OLLAMA_ALT_MODELS,
    DATA_DIR
)
from src.pipeline import MultimodalBrainTumorPipeline
from src.explainability.segmentation import segment_tumor_compartments

app = FastAPI(
    title="NeuroMultimodal AI API",
    description="Multimodal Brain Tumor Detection, Prediction & Classification API (CNN + GNN + RAG)",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = None


def get_pipeline():
    global pipeline
    if pipeline is None:
        print("[API] Initializing MultimodalBrainTumorPipeline...")
        pipeline = MultimodalBrainTumorPipeline(enable_rag=True)
    return pipeline


def image_to_base64(img: Image.Image, format: str = "JPEG") -> str:
    buffered = io.BytesIO()
    img.save(buffered, format=format)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def array_to_base64(arr: Any) -> str:
    pil_img = Image.fromarray(arr)
    return image_to_base64(pil_img)


PRESET_CASES = {
    "case_glioma": {
        "id": "case_glioma",
        "title": "Case 1: Glioblastoma Multiforme (Grade IV)",
        "target_class": "Glioma",
        "description": "58-year-old male presenting with new-onset focal motor seizures, worsening headaches, and left-sided hemiparesis.",
        "patient_info": {
            "age": 58,
            "gender": "Male",
            "symptoms": "New-onset focal motor seizures, worsening headaches, left-sided hemiparesis",
            "kps": 75,
            "idh1": "Wildtype",
            "mgmt": "Unmethylated",
            "codeletion_1p19q": "Non-codeleted"
        }
    },
    "case_pituitary": {
        "id": "case_pituitary",
        "title": "Case 2: Sellar Pituitary Macroadenoma",
        "target_class": "Pituitary Tumor",
        "description": "34-year-old female presenting with bitemporal hemianopsia, secondary amenorrhea, and mild frontal headaches.",
        "patient_info": {
            "age": 34,
            "gender": "Female",
            "symptoms": "Bitemporal hemianopsia, secondary amenorrhea, mild frontal headaches",
            "kps": 90,
            "idh1": "Wildtype",
            "mgmt": "Unmethylated",
            "codeletion_1p19q": "Intact"
        }
    },
    "case_meningioma": {
        "id": "case_meningioma",
        "title": "Case 3: Parasagittal Meningioma (Grade I)",
        "target_class": "Meningioma",
        "description": "62-year-old female presenting with progressive dull morning headache and mild localized cranial tenderness.",
        "patient_info": {
            "age": 62,
            "gender": "Female",
            "symptoms": "Progressive dull morning headache, mild localized cranial tenderness",
            "kps": 85,
            "idh1": "Wildtype",
            "mgmt": "Unmethylated",
            "codeletion_1p19q": "Intact"
        }
    },
    "case_normal": {
        "id": "case_normal",
        "title": "Case 4: Healthy Asymptomatic Control",
        "target_class": "No Tumor (Healthy)",
        "description": "42-year-old male presenting for routine executive health physical without any focal neurological deficit.",
        "patient_info": {
            "age": 42,
            "gender": "Male",
            "symptoms": "Occasional tension headache, routine health checkup, no neurological deficit",
            "kps": 100,
            "idh1": "Normal",
            "mgmt": "Normal",
            "codeletion_1p19q": "Intact"
        }
    }
}


@app.get("/api/status")
def get_status():
    pipe = get_pipeline()
    models = pipe.clinical_advisor.get_available_models() if pipe.clinical_advisor else OLLAMA_ALT_MODELS
    chroma_count = pipe.clinical_advisor.retriever.collection.count() if pipe.clinical_advisor else 0
    return {
        "status": "online",
        "device": str(pipe.device),
        "checkpoint": "multimodal_brain_tumor_net.pth",
        "models": models,
        "active_model": OLLAMA_CHAT_MODEL,
        "chroma_chunks": chroma_count
    }


@app.get("/api/cases")
def get_cases():
    pipe = get_pipeline()
    records_file = DATA_DIR / "clinical_records.csv"
    cases = []

    for key, c in PRESET_CASES.items():
        case_data = dict(c)
        img_b64 = None
        if records_file.exists():
            import pandas as pd
            df = pd.read_csv(records_file)
            matches = df[df["ground_truth_class"] == c["target_class"]]
            if not matches.empty:
                img_path = matches.iloc[0]["image_path"]
                if os.path.exists(img_path):
                    pil_img = Image.open(img_path).convert("RGB")
                    img_b64 = image_to_base64(pil_img)
                    case_data["image_path"] = img_path

        case_data["image_base64"] = img_b64
        cases.append(case_data)

    return {"cases": cases}


@app.post("/api/predict")
async def predict_tumor(
    file: Optional[UploadFile] = File(None),
    case_id: Optional[str] = Form(None),
    image_base64: Optional[str] = Form(None),
    patient_info: str = Form(...)
):
    pipe = get_pipeline()

    try:
        p_info = json.loads(patient_info)
    except Exception:
        p_info = {}

    pil_img = None

    if file:
        content = await file.read()
        pil_img = Image.open(io.BytesIO(content)).convert("RGB")
    elif image_base64:
        img_bytes = base64.b64decode(image_base64.split(",")[-1])
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    elif case_id and case_id in PRESET_CASES:
        records_file = DATA_DIR / "clinical_records.csv"
        if records_file.exists():
            import pandas as pd
            df = pd.read_csv(records_file)
            matches = df[df["ground_truth_class"] == PRESET_CASES[case_id]["target_class"]]
            if not matches.empty:
                img_path = matches.iloc[0]["image_path"]
                if os.path.exists(img_path):
                    pil_img = Image.open(img_path).convert("RGB")

    if pil_img is None:
        raise HTTPException(status_code=400, detail="No valid image input provided.")

    # Execute multimodal pipeline
    results = pipe.predict(
        image_input=pil_img,
        patient_info=p_info,
        generate_rag_report=False
    )

    cam_overlay_b64 = array_to_base64(results["cam_overlay"])
    gnn_overlay_b64 = array_to_base64(results["gnn_overlay"])
    seg_overlay_b64 = array_to_base64(results["seg_overlay"])
    orig_b64 = image_to_base64(results["original_image"])

    return {
        "predicted_class": results["predicted_class"],
        "confidence": results["confidence"],
        "class_probabilities": results["class_probabilities"],
        "is_tumor_detected": results["is_tumor_detected"],
        "detection_confidence": results["detection_confidence"],
        "predicted_grade": results["predicted_grade"],
        "grade_confidence": results["grade_confidence"],
        "gate_weights": results["gate_weights"],
        "gradcam_summary": results["gradcam_summary"],
        "gnn_summary": results["gnn_summary"],
        "volumetric_metrics": results.get("volumetric_metrics", {}),
        "original_image_base64": orig_b64,
        "cam_overlay_base64": cam_overlay_b64,
        "gnn_overlay_base64": gnn_overlay_b64,
        "seg_overlay_base64": seg_overlay_b64
    }


class CaliperRequest(BaseModel):
    p1: Dict[str, float]  # {x, y}
    p2: Dict[str, float]  # {x, y}
    image_width: int = 224
    image_height: int = 224
    pixel_spacing_mm: float = 0.9375


@app.post("/api/caliper")
def compute_caliper_measurement(req: CaliperRequest):
    """Computes exact Euclidean distance and estimated 3D tumor volume from 2 caliper endpoints."""
    dx = (req.p2["x"] - req.p1["x"]) * req.pixel_spacing_mm
    dy = (req.p2["y"] - req.p1["y"]) * req.pixel_spacing_mm
    distance_mm = math.sqrt(dx**2 + dy**2)

    # Estimate volume assuming spherical/ellipsoid approximation V = (pi / 6) * d^3
    volume_mm3 = (math.pi / 6.0) * (distance_mm ** 3)
    volume_cm3 = volume_mm3 / 1000.0

    return {
        "diameter_mm": round(distance_mm, 2),
        "estimated_volume_cm3": round(volume_cm3, 2),
        "midpoint_x": round((req.p1["x"] + req.p2["x"]) / 2.0, 1),
        "midpoint_y": round((req.p1["y"] + req.p2["y"]) / 2.0, 1)
    }


class RpaRequest(BaseModel):
    age: int
    kps: int
    resection_status: str = "GTR"  # "GTR", "STR", "Biopsy"
    neurological_function: str = "Normal"  # "Normal", "Deficit"


@app.post("/api/prognosis/rpa")
def compute_rpa_class(req: RpaRequest):
    """
    Computes RTOG Recursive Partitioning Analysis (RPA) Class for Glioblastoma.
    RPA Class III: Age < 50, KPS >= 90 (Median OS: ~17.9 months)
    RPA Class IV: Age < 50, KPS < 90 OR Age >= 50, KPS >= 70, GTR/STR (Median OS: ~15.0 months)
    RPA Class V: Age >= 50, KPS >= 70, Biopsy only OR Age >= 50, KPS < 70, Normal neuro (Median OS: ~10.0 months)
    RPA Class VI: Age >= 50, KPS < 70, Neurological deficit (Median OS: ~4.6 months)
    """
    age = req.age
    kps = req.kps
    is_resection = req.resection_status in ["GTR", "STR"]
    is_normal_neuro = req.neurological_function == "Normal"

    if age < 50:
        if kps >= 90:
            rpa_class = "Class III"
            median_os = 17.9
            two_year_os = "35%"
            risk_tier = "Favorable Glioblastoma"
        else:
            rpa_class = "Class IV"
            median_os = 14.8
            two_year_os = "20%"
            risk_tier = "Intermediate Risk"
    else:  # age >= 50
        if kps >= 70:
            if is_resection:
                rpa_class = "Class IV"
                median_os = 14.5
                two_year_os = "18%"
                risk_tier = "Intermediate Risk"
            else:
                rpa_class = "Class V"
                median_os = 10.0
                two_year_os = "8%"
                risk_tier = "High Risk"
        else:  # kps < 70
            if is_normal_neuro:
                rpa_class = "Class V"
                median_os = 8.9
                two_year_os = "6%"
                risk_tier = "High Risk"
            else:
                rpa_class = "Class VI"
                median_os = 4.6
                two_year_os = "2%"
                risk_tier = "Very High Risk / Frail"

    return {
        "rpa_class": rpa_class,
        "median_overall_survival_months": median_os,
        "two_year_survival_probability": two_year_os,
        "risk_tier": risk_tier
    }


class RagReportRequest(BaseModel):
    patient_info: Dict[str, Any]
    multimodal_results: Dict[str, Any]
    model_name: Optional[str] = None
    strategy: Optional[str] = "tumor_board"  # "tumor_board", "surgery", "radiation", "patient_friendly"
    temperature: Optional[float] = 0.2


@app.post("/api/rag/report")
def generate_rag_report(req: RagReportRequest):
    pipe = get_pipeline()
    if not pipe.clinical_advisor:
        raise HTTPException(status_code=500, detail="RAG Clinical Advisor is not initialized.")

    out = pipe.clinical_advisor.generate_consultation_report(
        patient_info=req.patient_info,
        multimodal_results=req.multimodal_results,
        model_name=req.model_name
    )
    return out


class RagChatRequest(BaseModel):
    case_context: str
    question: str
    model_name: Optional[str] = None


@app.post("/api/rag/chat")
def chat_clinical_rag(req: RagChatRequest):
    pipe = get_pipeline()
    if not pipe.clinical_advisor:
        raise HTTPException(status_code=500, detail="RAG Clinical Advisor is not initialized.")

    ans = pipe.clinical_advisor.answer_clinical_question(
        case_context=req.case_context,
        user_question=req.question,
        model_name=req.model_name
    )
    return {"answer": ans}


@app.get("/api/guidelines/search")
def search_guidelines(q: str = Query(..., min_length=2)):
    pipe = get_pipeline()
    if not pipe.clinical_advisor:
        raise HTTPException(status_code=500, detail="Retriever not initialized.")

    docs = pipe.clinical_advisor.retriever.retrieve(query=q, top_k=5)
    return {"query": q, "results": docs}


@app.get("/api/metrics")
def get_metrics():
    summary_path = CHECKPOINT_DIR / "evaluation_summary.json"
    cm_path = CHECKPOINT_DIR / "confusion_matrix.png"

    summary_data = {}
    if summary_path.exists():
        with open(summary_path, "r") as f:
            summary_data = json.load(f)

    cm_b64 = None
    if cm_path.exists():
        with open(cm_path, "rb") as f:
            cm_b64 = base64.b64encode(f.read()).decode("utf-8")

    return {
        "metrics": summary_data,
        "confusion_matrix_base64": cm_b64
    }


# Mount static files for React production build
dist_dir = Path(__file__).resolve().parent / "frontend" / "dist"
if dist_dir.exists():
    app.mount("/", StaticFiles(directory=str(dist_dir), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    print("[API] Starting server on http://127.0.0.1:8000 ...")
    uvicorn.run("api_server:app", host="127.0.0.1", port=8000, reload=False)
