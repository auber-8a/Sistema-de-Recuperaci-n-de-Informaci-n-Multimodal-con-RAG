"""
Base de datos vectorial con FAISS (requisito c del PDF).

Usa un IndexFlatIP (producto interno) sobre embeddings normalizados L2, lo
que equivale a similitud coseno. Se guarda junto al indice un parquet con
los metadatos de cada documento (mismo orden que los vectores) para poder
reconstruir los resultados de busqueda.
"""
import json
from pathlib import Path

import faiss
import numpy as np
import pandas as pd

from src import config


class FaissVectorStore:
    def __init__(self, dim: int):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.metadata: pd.DataFrame | None = None

    def construir(self, embeddings: np.ndarray, metadata: pd.DataFrame):
        assert embeddings.shape[0] == len(metadata), "embeddings y metadata deben tener el mismo largo"
        assert embeddings.shape[1] == self.dim
        self.index.add(embeddings.astype("float32"))
        self.metadata = metadata.reset_index(drop=True)

    def reconstruir_por_id_producto(self, product_id: str) -> np.ndarray | None:
        """Recupera el vector original indexado para un product_id (usado por relevance feedback)."""
        matches = self.metadata.index[self.metadata["product_id"] == product_id]
        if len(matches) == 0:
            return None
        position = int(matches[0])
        return np.array(self.index.reconstruct(position))

    def buscar(self, query_embedding: np.ndarray, top_k: int = config.TOP_K_RETRIEVE) -> list[dict]:
        query_embedding = query_embedding.astype("float32").reshape(1, -1)
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            row = self.metadata.iloc[idx].to_dict()
            row["score"] = float(score)
            results.append(row)
        return results

    def guardar(self, directory: Path = config.INDEX_DIR):
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(directory / "index.faiss"))
        self.metadata.to_parquet(directory / "metadata.parquet")
        with open(directory / "meta_info.json", "w") as f:
            json.dump({"dim": self.dim}, f)

    @classmethod
    def cargar(cls, directory: Path = config.INDEX_DIR) -> "FaissVectorStore":
        with open(directory / "meta_info.json") as f:
            info = json.load(f)
        store = cls(dim=info["dim"])
        store.index = faiss.read_index(str(directory / "index.faiss"))
        store.metadata = pd.read_parquet(directory / "metadata.parquet")
        return store
