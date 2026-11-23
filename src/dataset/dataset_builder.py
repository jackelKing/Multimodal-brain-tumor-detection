"""
Multimodal Brain Tumor Dataset Builder.
Combines MRI neuroimaging, Superpixel Graphs (PyG), and Clinical Biomarker Records.
"""
import os
import random
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torch_geometric.data import Data, Batch

from src.config import (
    MRI_DIR,
    CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    IMAGE_SIZE,
    PATCH_GRID,
    NODE_FEATURE_DIM,
    DATA_DIR
)
from src.dataset.graph_utils import build_grid_graph


def get_mri_transforms(is_train: bool = True) -> T.Compose:
    """MRI image augmentation and normalization pipeline."""
    if is_train:
        return T.Compose([
            T.Resize(IMAGE_SIZE),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.1, contrast=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        return T.Compose([
            T.Resize(IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def generate_synthetic_clinical_profile(tumor_class: str, seed: Optional[int] = None) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
    """
    Generates biologically and epidemiologically realistic clinical biomarker records
    conforming to clinical oncology literature for each tumor category.
    Returns:
      - clinical_tensor: torch.Tensor of shape [8] (normalized features)
      - who_grade: int (0 = Grade I, 1 = Grade II, 2 = Grade III, 3 = Grade IV)
      - meta_dict: human readable clinical description
    """
    rng = random.Random(seed) if seed is not None else random.Random()
    c = tumor_class.lower().replace("-", "").replace(" ", "")

    if "notumor" in c:
        age = rng.randint(20, 75)
        gender = rng.choice(["Male", "Female"])
        symptoms = rng.choice([
            "Occasional tension headaches, routine screening",
            "Mild dizziness, no focal neurological deficits",
            "Workplace physical examination, asymptomatic",
            "Migraine without aura, normal neurological exam"
        ])
        kps = rng.randint(90, 100)
        idh1 = "Wildtype (Normal)"
        mgmt = "Normal"
        codeletion_1p19q = "Intact"
        headache = 1.0 if "headache" in symptoms.lower() else 0.0
        seizure = 0.0
        vision = 0.0
        symptom_sev = 0.1
        biomarker_prior = 0.0
        who_grade = 0  # No grade / normal

    elif "pituitary" in c:
        age = rng.randint(25, 65)
        gender = rng.choice(["Female", "Male", "Female"])  # Slight female skew
        vision_deficit = rng.choice([True, True, False])
        symptoms = rng.choice([
            "Bitemporal hemianopsia, visual field defects, galactorrhea" if vision_deficit else "Mild frontal headache, hyperprolactinemia",
            "Decreased visual acuity, bitemporal visual cuts, fatigue",
            "Unexplained secondary amenorrhea, persistent headaches"
        ])
        kps = rng.randint(80, 95)
        idh1 = "Wildtype"
        mgmt = "Unmethylated"
        codeletion_1p19q = "Intact"
        headache = 1.0
        seizure = 0.0
        vision = 1.0 if vision_deficit else 0.0
        symptom_sev = 0.4
        biomarker_prior = 0.1
        who_grade = 0  # Benign PitNET / Grade I

    elif "meningioma" in c:
        age = rng.randint(45, 75)
        gender = rng.choice(["Female", "Female", "Male"])  # 2:1 female preponderance
        symptoms = rng.choice([
            "Progressive dull morning headache, localized cranial tenderness",
            "New-onset focal motor seizure, olfactory dysfunction",
            "Gradual gait unsteadiness, mild focal weakness",
            "Incidental finding on trauma imaging, mild headache"
        ])
        kps = rng.randint(70, 95)
        idh1 = "Wildtype"
        mgmt = "Unmethylated"
        codeletion_1p19q = "Intact"
        headache = 1.0 if "headache" in symptoms.lower() else 0.5
        seizure = 1.0 if "seizure" in symptoms.lower() else 0.0
        vision = 0.2
        symptom_sev = 0.5
        # Grade 1 (85%) or Grade 2 (15%)
        is_atypical = rng.random() < 0.18
        who_grade = 1 if is_atypical else 0
        biomarker_prior = 0.3 if is_atypical else 0.1

    else:  # glioma
        age = rng.randint(35, 75)
        gender = rng.choice(["Male", "Male", "Female"])  # Slight male skew
        is_high_grade = rng.random() < 0.70  # Most clinical presentation gliomas are GBM / Grade IV
        if is_high_grade:
            who_grade = 3  # Grade IV / Glioblastoma
            idh1 = rng.choice(["Wildtype", "Wildtype", "Mutant (R132H)"])
            mgmt = rng.choice(["Unmethylated", "Methylated"])
            codeletion_1p19q = "Non-codeleted"
            kps = rng.randint(60, 85)
            symptoms = rng.choice([
                "Progressive contralateral hemiparesis, severe morning vomiting, papilledema",
                "New-onset grand mal seizures, cognitive decline, dysphasia",
                "Rapidly escalating headaches, personality changes, confusion",
                "Right-sided hemiplegia, expressive aphasia, intracranial hypertension"
            ])
            symptom_sev = 0.9
            biomarker_prior = 0.95
        else:
            who_grade = rng.choice([1, 2])  # Grade II or Grade III
            idh1 = "Mutant (R132H)"
            mgmt = "Methylated"
            codeletion_1p19q = rng.choice(["Codeleted (Oligodendroglioma)", "Non-codeleted (Astrocytoma)"])
            kps = rng.randint(80, 95)
            symptoms = rng.choice([
                "First-time nocturnal focal seizure, mild postictal confusion",
                "Intermittent mild headaches, mild word-finding hesitation",
                "Low-frequency partial seizures, preserved cognition"
            ])
            symptom_sev = 0.6
            biomarker_prior = 0.5

        headache = 1.0
        seizure = 1.0 if "seizure" in symptoms.lower() else 0.5
        vision = 0.3

    # Normalized 8-dimensional clinical feature vector
    norm_age = (age - 50.0) / 25.0
    gender_val = 1.0 if gender == "Male" else 0.0
    norm_kps = kps / 100.0

    features = [
        norm_age,
        gender_val,
        symptom_sev,
        headache,
        seizure,
        vision,
        norm_kps,
        biomarker_prior
    ]

    meta = {
        "age": age,
        "gender": gender,
        "symptoms": symptoms,
        "kps": kps,
        "idh1": idh1,
        "mgmt": mgmt,
        "codeletion_1p19q": codeletion_1p19q,
        "who_grade_num": who_grade
    }

    return torch.tensor(features, dtype=torch.float32), who_grade, meta


class MultimodalBrainTumorDataset(Dataset):
    """
    Multimodal Dataset returning:
      - image: [3, 224, 224]
      - graph: PyG Data (tissue superpixel adjacency graph)
      - clinical: [8]
      - label: 4-class multi-class (0: Glioma, 1: Meningioma, 2: No Tumor, 3: Pituitary)
      - binary_label: 1 if Tumor, 0 if Normal
      - grade_label: WHO grade (0, 1, 2, 3)
      - meta: clinical dictionary
    """
    def __init__(
        self,
        split: str = "train",
        base_dir: Path = MRI_DIR,
        max_samples_per_class: Optional[int] = None,
        transform: Optional[T.Compose] = None
    ):
        self.split = split
        self.base_dir = Path(base_dir)
        self.split_dir = self.base_dir / split
        self.transform = transform or get_mri_transforms(is_train=(split == "train"))
        self.samples: List[Tuple[Path, int, str]] = []

        if not self.split_dir.exists():
            raise FileNotFoundError(f"Split directory {self.split_dir} does not exist.")

        # Collect image files per class
        class_folders = [d for d in self.split_dir.iterdir() if d.is_dir()]
        for folder in class_folders:
            folder_name = folder.name.lower().replace("-", "").replace(" ", "")
            label_idx = None
            norm_class = None

            if "glioma" in folder_name:
                label_idx = 0
                norm_class = "glioma"
            elif "meningioma" in folder_name:
                label_idx = 1
                norm_class = "meningioma"
            elif "notumor" in folder_name or "no" in folder_name:
                label_idx = 2
                norm_class = "notumor"
            elif "pituitary" in folder_name:
                label_idx = 3
                norm_class = "pituitary"

            if label_idx is not None:
                files = list(folder.glob("*.jpg")) + list(folder.glob("*.png")) + list(folder.glob("*.jpeg"))
                if max_samples_per_class:
                    files = files[:max_samples_per_class]
                for f in files:
                    self.samples.append((f, label_idx, norm_class))

        # Shuffle deterministically
        random.Random(42).shuffle(self.samples)
        print(f"[Dataset] Split '{split}' loaded {len(self.samples)} multimodal samples across classes.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, label_idx, norm_class = self.samples[idx]

        # Load image
        img_pil = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(img_pil)

        # Build graph from spatial patches
        graph_data = build_grid_graph(image_tensor, grid_size=PATCH_GRID, feature_dim=NODE_FEATURE_DIM)

        # Generate realistic paired clinical features using image path hash as deterministic seed
        seed = abs(hash(str(img_path.name))) % (2**31)
        clinical_tensor, who_grade, meta = generate_synthetic_clinical_profile(norm_class, seed=seed)
        meta["image_path"] = str(img_path)
        meta["filename"] = img_path.name
        meta["class_name"] = IDX_TO_CLASS[label_idx]

        binary_label = 0 if label_idx == 2 else 1

        return {
            "image": image_tensor,
            "graph": graph_data,
            "clinical": clinical_tensor,
            "label": torch.tensor(label_idx, dtype=torch.long),
            "binary_label": torch.tensor(binary_label, dtype=torch.long),
            "grade_label": torch.tensor(who_grade, dtype=torch.long),
            "meta": meta
        }


def multimodal_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collator handling both PyTorch tensors and PyG Graph batches."""
    images = torch.stack([item["image"] for item in batch])
    graphs = Batch.from_data_list([item["graph"] for item in batch])
    clinicals = torch.stack([item["clinical"] for item in batch])
    labels = torch.stack([item["label"] for item in batch])
    binary_labels = torch.stack([item["binary_label"] for item in batch])
    grade_labels = torch.stack([item["grade_label"] for item in batch])
    metas = [item["meta"] for item in batch]

    return {
        "images": images,
        "graphs": graphs,
        "clinicals": clinicals,
        "labels": labels,
        "binary_labels": binary_labels,
        "grade_labels": grade_labels,
        "metas": metas
    }


def export_clinical_records_csv(output_path: Path = DATA_DIR / "clinical_records.csv"):
    """Exports clinical records for all test/valid samples into a CSV for the Streamlit UI."""
    records = []
    for split in ["test", "valid"]:
        try:
            ds = MultimodalBrainTumorDataset(split=split)
            for i in range(min(len(ds), 300)):
                item = ds[i]
                meta = item["meta"]
                records.append({
                    "split": split,
                    "filename": meta["filename"],
                    "image_path": meta["image_path"],
                    "ground_truth_class": meta["class_name"],
                    "who_grade": meta["who_grade_num"],
                    "age": meta["age"],
                    "gender": meta["gender"],
                    "symptoms": meta["symptoms"],
                    "kps": meta["kps"],
                    "idh1": meta["idh1"],
                    "mgmt": meta["mgmt"],
                    "codeletion_1p19q": meta["codeletion_1p19q"]
                })
        except Exception as e:
            print(f"[Warning] Failed split {split} export: {e}")

    if records:
        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)
        print(f"[Dataset] Exported {len(df)} clinical patient records to {output_path}")


if __name__ == "__main__":
    print("Testing Multimodal Brain Tumor Dataset...")
    test_ds = MultimodalBrainTumorDataset(split="test")
    sample = test_ds[0]
    print("Image shape:", sample["image"].shape)
    print("Graph nodes:", sample["graph"].x.shape, "edges:", sample["graph"].edge_index.shape)
    print("Clinical vector:", sample["clinical"])
    print("Label:", sample["label"], "Grade:", sample["grade_label"])
    print("Patient Meta:", sample["meta"])
    export_clinical_records_csv()
