"""
Clinical Advisor and Decision Support Engine using RAG and local Ollama LLMs.
Synthesizes CNN-GNN multimodal diagnostic findings with retrieved WHO CNS5 oncology guidelines.
"""
import json
import requests
from typing import Dict, Any, List, Optional
from src.config import OLLAMA_BASE_URL, OLLAMA_CHAT_MODEL, OLLAMA_ALT_MODELS
from src.rag.retriever import ClinicalRetriever


class ClinicalAdvisor:
    """Orchestrates RAG retrieval and LLM generation for neuro-oncology reports and clinical Q&A."""
    def __init__(
        self,
        retriever: Optional[ClinicalRetriever] = None,
        base_url: str = OLLAMA_BASE_URL,
        default_model: str = OLLAMA_CHAT_MODEL
    ):
        self.retriever = retriever or ClinicalRetriever()
        self.base_url = base_url
        self.default_model = default_model

    def get_available_models(self) -> List[str]:
        """Queries Ollama for currently available local models."""
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                return models
        except Exception as e:
            print(f"[Warning] Failed to fetch Ollama models: {e}")
        return OLLAMA_ALT_MODELS

    def generate_consultation_report(
        self,
        patient_info: Dict[str, Any],
        multimodal_results: Dict[str, Any],
        model_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generates a comprehensive clinical neuro-oncology report by:
        1. Formulating clinical retrieval queries from patient & model results.
        2. Retrieving relevant WHO/NCCN guidelines from ChromaDB.
        3. Invoking Ollama LLM with structured clinical context.
        """
        model = model_name or self.default_model

        # Extract predictions
        predicted_class = multimodal_results.get("predicted_class", "glioma").lower()
        confidence = multimodal_results.get("confidence", 0.95)
        class_probs = multimodal_results.get("class_probabilities", {})
        predicted_grade = multimodal_results.get("predicted_grade", "Grade IV / Malignant")
        grade_confidence = multimodal_results.get("grade_confidence", 0.90)
        gradcam_info = multimodal_results.get("gradcam_summary", "Focal hyper-intense activation in cerebral hemisphere with surrounding peritumoral edema.")
        gnn_info = multimodal_results.get("gnn_summary", "High topological node centrality and dense superpixel connectivity indicating spatial heterogeneity.")

        # 1. Retrieve clinical evidence
        search_query = f"{predicted_class} {predicted_grade} surgical resection chemotherapy radiotherapy molecular markers"
        retrieved_guidelines = self.retriever.retrieve(
            query=search_query,
            tumor_class=predicted_class,
            top_k=3
        )

        guideline_context = ""
        for i, doc in enumerate(retrieved_guidelines, 1):
            guideline_context += f"\n--- GUIDELINE EXCERPT {i} [{doc['title']} - {doc['header']}] ---\n{doc['content']}\n"

        # 2. Build structured prompt
        prompt = f"""You are a world-class Senior Consultant Neuro-Oncologist and Neuroradiologist at an academic medical center.
You have been provided with comprehensive multimodal AI diagnostic findings combining:
- Convolutional Neural Network (CNN) MRI visual pattern detection
- Graph Neural Network (GNN) tissue topological connectivity & heterogeneity analysis
- Patient clinical biomarkers, history, and neurological presentation
- Evidence-based WHO CNS5 & NCCN clinical oncology guidelines retrieved from the medical knowledge base.

PATIENT PROFILE:
- Age: {patient_info.get('age', 54)} years
- Gender: {patient_info.get('gender', 'Male')}
- Chief Complaints / Symptoms: {patient_info.get('symptoms', 'Progressive headache, new-onset seizures, left hemiparesis')}
- Karnofsky Performance Score (KPS): {patient_info.get('kps', 80)} / 100
- IDH1 Status: {patient_info.get('idh1', 'Wildtype')}
- MGMT Promoter: {patient_info.get('mgmt', 'Unmethylated')}
- 1p/19q Codeletion: {patient_info.get('codeletion_1p19q', 'Non-codeleted')}

MULTIMODAL AI DIAGNOSTIC FINDINGS:
- Primary Prediction: {predicted_class.upper()} (Model Confidence: {confidence*100:.1f}%)
- Multiclass Probabilities: {json.dumps(class_probs)}
- Prognostic WHO Histological Grade: {predicted_grade} (Confidence: {grade_confidence*100:.1f}%)
- CNN Spatial Explainability (Grad-CAM): {gradcam_info}
- GNN Topological Graph Analysis: {gnn_info}

RETRIEVED CLINICAL PRACTICE GUIDELINES:
{guideline_context}

TASK:
Generate a thorough, authoritative, and medically rigorous "Multimodal Neuro-Oncology Consultation Report". Format your report with the following clear markdown headers:

### 1. Diagnostic Summary & Subtype Classification
(State primary finding, confidence, differential diagnosis considerations, and how CNN and GNN corroborate each other)

### 2. Radiological-Pathological Correlation
(Discuss MRI findings, Grad-CAM localized activation, and GNN tissue topological heterogeneity)

### 3. WHO CNS5 Grading & Prognostic Risk Stratification
(Analyze grade, impact of patient age, KPS, IDH1/MGMT biomarkers, and expected prognosis)

### 4. Multidisciplinary Treatment Recommendations
(Provide concrete, actionable protocols for:
 a) Surgical Neurosurgical Management (Resection extent, intraoperative navigation, 5-ALA)
 b) Medical Oncology & Chemotherapy (e.g., Stupp protocol TMZ dosing, TTFields, or endocrine therapy)
 c) Radiation Oncology (Target dose, fractions, margins)
 d) Supportive & Symptom Care (Dexamethasone, antiepileptics))

### 5. Follow-Up Schedule & Molecular Surveillance
(Specify post-op imaging timing and clinical monitoring)

Respond directly in professional medical terminology. Maintain clear clinical precision."""

        # 3. Call Ollama
        report_text = ""
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.2,
                        "num_predict": 1200
                    }
                },
                timeout=120
            )
            if resp.status_code == 200:
                report_text = resp.json().get("response", "").strip()
            else:
                report_text = f"Ollama generation returned HTTP status {resp.status_code}: {resp.text}"
        except Exception as e:
            report_text = f"Error communicating with local Ollama server: {e}"

        return {
            "report": report_text,
            "retrieved_guidelines": retrieved_guidelines,
            "model_used": model,
            "prompt_length": len(prompt)
        }

    def answer_clinical_question(
        self,
        case_context: str,
        user_question: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
        model_name: Optional[str] = None
    ) -> str:
        """Interactive Q&A for clinicians asking follow-up questions on the case."""
        model = model_name or self.default_model

        # Retrieve any additional guidelines relevant to the question
        extra_docs = self.retriever.retrieve(query=user_question, top_k=2)
        extra_context = "\n".join([f"[{d['title']}]: {d['content']}" for d in extra_docs])

        prompt = f"""You are an expert Neuro-Oncologist serving as an AI clinical decision support assistant.
Below is the patient's case background and diagnostic findings:

{case_context}

Relevant Medical Guidelines:
{extra_context}

Clinician's Question:
{user_question}

Provide an accurate, concise, evidence-based medical answer adhering to WHO CNS5 and NCCN standards."""

        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.3, "num_predict": 600}
                },
                timeout=90
            )
            if resp.status_code == 200:
                return resp.json().get("response", "").strip()
            return f"Error: Ollama returned status {resp.status_code}"
        except Exception as e:
            return f"Error communicating with Ollama: {e}"


if __name__ == "__main__":
    advisor = ClinicalAdvisor()
    print("Testing Clinical Advisor with llama3.2...")
    sample_patient = {
        "age": 58,
        "gender": "Male",
        "symptoms": "New-onset nocturnal focal motor seizures, headache, right hemiparesis",
        "kps": 80,
        "idh1": "Wildtype",
        "mgmt": "Unmethylated",
        "codeletion_1p19q": "Non-codeleted"
    }
    sample_results = {
        "predicted_class": "glioma",
        "confidence": 0.96,
        "class_probabilities": {"glioma": 0.96, "meningioma": 0.02, "notumor": 0.01, "pituitary": 0.01},
        "predicted_grade": "Grade IV / Malignant",
        "grade_confidence": 0.92,
        "gradcam_summary": "Intense contrast-enhancing irregular peripheral margin with necrotic central core in right frontoparietal white matter.",
        "gnn_summary": "Marked spatial heterogeneity across superpixel graph with sharp edge weight gradients at the infiltrative tumor margin."
    }
    report_data = advisor.generate_consultation_report(sample_patient, sample_results)
    print("\n--- GENERATED CLINICAL CONSULTATION REPORT ---")
    print(report_data["report"][:600] + "...")
