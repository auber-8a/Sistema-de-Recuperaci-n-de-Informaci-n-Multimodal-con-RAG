"""
Funcionalidad de excelencia: Re-ranking (+15).

Refina el ranking inicial de FAISS (similitud vectorial CLIP) usando un
CrossEncoder texto-texto, que evalua conjuntamente la query y el documento
en vez de comparar embeddings independientes, mejorando la precision del
Top-k final que se usa como contexto del RAG.
"""
from sentence_transformers import CrossEncoder

from src import config


class Reranker:
    def __init__(self, model_name: str = config.RERANKER_MODEL_NAME, device: str | None = None):
        self.model = CrossEncoder(model_name, device=device)

    def rerankear(self, query: str, candidates: list[dict], top_k: int = config.TOP_K_FINAL) -> list[dict]:
        if not candidates:
            return []

        pairs = [(query, c["document"]) for c in candidates]
        scores = self.model.predict(pairs)

        reranked = [
            {**c, "rerank_score": float(score)}
            for c, score in zip(candidates, scores)
        ]
        reranked.sort(key=lambda c: c["rerank_score"], reverse=True)
        return reranked[:top_k]
