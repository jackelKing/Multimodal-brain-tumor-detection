"""
Clinical Knowledge Retriever for RAG.
Performs semantic similarity search over indexed clinical guidelines.
"""
from typing import List, Dict, Any, Optional
from src.rag.vector_store import MedicalKnowledgeBase


class ClinicalRetriever:
    """Retrieves relevant clinical oncology guidelines based on diagnosis and clinical queries."""
    def __init__(self, kb: Optional[MedicalKnowledgeBase] = None):
        self.kb = kb or MedicalKnowledgeBase()
        self.collection = self.kb.collection

    def retrieve(
        self,
        query: str,
        tumor_class: Optional[str] = None,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k most relevant medical guidelines for the given query.
        Optionally filters by tumor_class ('glioma', 'meningioma', 'pituitary', 'notumor').
        """
        where_filter = None
        if tumor_class and tumor_class.lower() in ["glioma", "meningioma", "pituitary", "notumor"]:
            # Match specific tumor class or general multimodal protocol
            where_filter = {
                "$or": [
                    {"tumor_class": tumor_class.lower()},
                    {"tumor_class": "multimodal_protocol"},
                    {"tumor_class": "general"}
                ]
            }

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter
            )

            retrieved_chunks = []
            if results and results.get("documents") and results["documents"][0]:
                docs = results["documents"][0]
                metas = results["metadatas"][0]
                distances = results["distances"][0] if "distances" in results else [0.0] * len(docs)
                ids = results["ids"][0]

                for doc, meta, dist, cid in zip(docs, metas, distances, ids):
                    # Chroma with cosine space returns cosine distance in [0, 2]
                    # Cosine similarity = 1.0 - distance
                    similarity = max(0.0, min(1.0, 1.0 - dist)) if dist is not None else 0.85
                    retrieved_chunks.append({
                        "id": cid,
                        "content": doc,
                        "header": meta.get("header", "Guideline Section"),
                        "title": meta.get("title", "Clinical Protocol"),
                        "tumor_class": meta.get("tumor_class", "general"),
                        "distance": dist,
                        "relevance_score": round(similarity, 4)
                    })

            return retrieved_chunks

        except Exception as e:
            print(f"[Retriever] Query error: {e}")
            # Fallback query without filter if where_filter syntax fails
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=top_k
                )
                retrieved_chunks = []
                if results and results.get("documents") and results["documents"][0]:
                    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                        retrieved_chunks.append({
                            "content": doc,
                            "header": meta.get("header", "Guideline Section"),
                            "title": meta.get("title", "Clinical Protocol"),
                            "source": meta.get("source", "WHO Guidelines"),
                            "tumor_class": meta.get("tumor_class", "general"),
                            "relevance_score": 0.85
                        })
                return retrieved_chunks
            except Exception as e2:
                print(f"[Retriever] Fallback query error: {e2}")
                return []


if __name__ == "__main__":
    retriever = ClinicalRetriever()
    results = retriever.retrieve("What is the Stupp protocol and adjuvant chemotherapy for glioblastoma?", top_k=2)
    for r in results:
        print(f"\n[{r['title']} - {r['header']}] (Score: {r['relevance_score']})")
        print(r['content'][:250] + "...")
