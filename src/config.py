"""
Multimodal Brain Tumor Detection, Prediction, and Classification System
Global Configuration & Hyperparameters
"""
import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MRI_DIR = DATA_DIR / "mri_dataset" / "timri"
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
CHROMA_DB_DIR = DATA_DIR / "chroma_db"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

# Tumor Classes
CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_TO_IDX = {"glioma": 0, "meningioma": 1, "notumor": 2, "no-tumor": 2, "pituitary": 3}
IDX_TO_CLASS = {0: "Glioma", 1: "Meningioma", 2: "No Tumor (Healthy)", 3: "Pituitary Tumor"}
NUM_CLASSES = 4

# WHO Grade Classes for prognostic prediction
GRADE_CLASSES = ["Grade I / Benign", "Grade II / Low-Grade", "Grade III / Anaplastic", "Grade IV / Malignant"]
NUM_GRADES = 4

# Ollama Configuration
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_EMBED_MODEL = "nomic-embed-text"
OLLAMA_CHAT_MODEL = "llama3.2:latest"
OLLAMA_ALT_MODELS = ["llama3.2:latest", "llama3.1:8b", "deepseek-r1:8b", "llava:7b"]

# Image Preprocessing & Model Architecture
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Graph Superpixel / Patch Configuration
PATCH_GRID = (7, 7)       # 7x7 = 49 graph nodes per MRI image
NODE_FEATURE_DIM = 64     # Node feature dimension from CNN spatial feature map
GNN_HIDDEN_DIM = 128      # GNN hidden state dimension
GNN_EMBEDDING_DIM = 128   # GNN output graph representation dimension
CNN_EMBEDDING_DIM = 256   # CNN global feature representation dimension
CLINICAL_FEATURE_DIM = 8  # Demographic and biomarker tabular features
CLINICAL_EMBED_DIM = 64   # Clinical tabular projected embedding dimension
FUSION_HIDDEN_DIM = 256   # Joint multimodal fusion latent dimension

# Device
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
