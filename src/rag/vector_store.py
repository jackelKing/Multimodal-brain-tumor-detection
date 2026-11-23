"""
Vector Store and Embedding Indexing for Brain Tumor RAG Clinical Decision Support.
Uses Ollama 'nomic-embed-text' and ChromaDB for local vector storage.
"""
import os
import re
import json
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from src.config import (
    KNOWLEDGE_BASE_DIR,
    CHROMA_DB_DIR,
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_MODEL
)


class OllamaEmbeddingFunction(chromadb.EmbeddingFunction):
    """Custom ChromaDB Embedding Function that queries Ollama nomic-embed-text."""
    def __init__(self, base_url: str = OLLAMA_BASE_URL, model_name: str = OLLAMA_EMBED_MODEL):
        self.base_url = base_url
        self.model_name = model_name
        self._cache: Dict[str, List[float]] = {}

    def get_embedding(self, text: str) -> List[float]:
        text = text.strip()
        if text in self._cache:
            return self._cache[text]

        try:
            resp = requests.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model_name, "prompt": text},
                timeout=30
            )
            if resp.status_code == 200:
                data = resp.json()
                emb = data.get("embedding", [])
                if emb:
                    import numpy as np
                    v = np.array(emb, dtype=float)
                    norm = np.linalg.norm(v)
                    if norm > 0:
                        v = v / norm
                    normalized_emb = v.tolist()
                    self._cache[text] = normalized_emb
                    return normalized_emb
        except Exception as e:
            print(f"[Warning] Ollama embedding request failed: {e}. Using deterministic fallback.")

        # Deterministic hash pseudo-embedding fallback of dim 768 if ollama unreachable
        import hashlib
        import numpy as np
        h = hashlib.sha256(text.encode()).digest()
        seed = int.from_bytes(h[:4], "big")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(768).astype(float).tolist()
        return vec

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return [self.get_embedding(doc) for doc in input]


class MedicalKnowledgeBase:
    """Loads, chunks, and indexes clinical oncology knowledge documents into ChromaDB."""
    def __init__(
        self,
        kb_dir: Path = KNOWLEDGE_BASE_DIR,
        persist_dir: Path = CHROMA_DB_DIR,
        collection_name: str = "brain_tumor_guidelines"
    ):
        self.kb_dir = Path(kb_dir)
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self.embedding_fn = OllamaEmbeddingFunction()

        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=self.embedding_fn,
            metadata={"description": "WHO CNS5 and NCCN brain tumor clinical guidelines", "hnsw:space": "cosine"}
        )

    def parse_markdown_document(self, file_path: Path) -> List[Dict[str, Any]]:
        """Splits markdown file by section headers into discrete clinical guideline chunks."""
        text = file_path.read_text(encoding="utf-8")
        filename = file_path.stem

        # Infer tumor class
        tumor_class = "general"
        if "glioma" in filename:
            tumor_class = "glioma"
        elif "meningioma" in filename:
            tumor_class = "meningioma"
        elif "pituitary" in filename:
            tumor_class = "pituitary"
        elif "normal" in filename:
            tumor_class = "notumor"
        elif "multimodal" in filename:
            tumor_class = "multimodal_protocol"

        sections = re.split(r'\n(?=##?\s)', text)
        chunks = []

        doc_title = file_path.name
        first_line = text.strip().split("\n")[0]
        if first_line.startswith("# "):
            doc_title = first_line.replace("# ", "").strip()

        for idx, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue

            lines = section.split("\n")
            header = lines[0].replace("#", "").strip() if lines[0].startswith("#") else f"Section {idx+1}"
            content = section

            # Ensure minimum meaningful chunk length
            if len(content) < 50:
                continue

            chunk_id = f"{filename}_chunk_{idx}"
            chunks.append({
                "id": chunk_id,
                "text": content,
                "metadata": {
                    "source": file_path.name,
                    "title": doc_title,
                    "header": header,
                    "tumor_class": tumor_class,
                    "char_count": len(content)
                }
            })

        return chunks

    def build_index(self, force_rebuild: bool = False):
        """Indexes all markdown files into ChromaDB."""
        current_count = self.collection.count()
        if current_count > 0 and not force_rebuild:
            print(f"[VectorStore] Existing index found with {current_count} documents. Skipping rebuild.")
            return

        if force_rebuild and current_count > 0:
            print(f"[VectorStore] Resetting collection '{self.collection_name}'...")
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_fn
            )

        md_files = list(self.kb_dir.glob("*.md"))
        print(f"[VectorStore] Indexing {len(md_files)} knowledge base documents...")

        all_ids = []
        all_docs = []
        all_metas = []

        for md_file in md_files:
            chunks = self.parse_markdown_document(md_file)
            for chunk in chunks:
                all_ids.append(chunk["id"])
                all_docs.append(chunk["text"])
                all_metas.append(chunk["metadata"])

        if all_docs:
            print(f"[VectorStore] Generating embeddings and adding {len(all_docs)} chunks to ChromaDB...")
            # Add in batches
            batch_size = 10
            for i in range(0, len(all_docs), batch_size):
                b_ids = all_ids[i:i + batch_size]
                b_docs = all_docs[i:i + batch_size]
                b_metas = all_metas[i:i + batch_size]
                self.collection.add(
                    ids=b_ids,
                    documents=b_docs,
                    metadatas=b_metas
                )
            print(f"[VectorStore] Indexing complete! Total chunks indexed: {self.collection.count()}")


if __name__ == "__main__":
    kb = MedicalKnowledgeBase()
    kb.build_index(force_rebuild=True)
