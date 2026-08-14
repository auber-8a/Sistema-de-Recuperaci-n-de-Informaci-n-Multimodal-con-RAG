"""
Funcionalidad de excelencia: Relevance Feedback (+15).

El usuario califica resultados con "Me gusta" / "No me gusta". Se guarda el
feedback por sesion y se usa para ajustar el vector de la consulta en
busquedas posteriores mediante el algoritmo de Rocchio:

    q' = alpha * q + beta * mean(vectores 'me gusta') - gamma * mean(vectores 'no me gusta')

El vector ajustado se renormaliza para poder seguir usando similitud coseno
(producto interno) contra el indice FAISS.
"""
import numpy as np

from src.vector_store import FaissVectorStore


class FeedbackStore:
    def __init__(self):
        # session_id -> {"liked": set(product_id), "disliked": set(product_id)}
        self.sessions: dict[str, dict[str, set]] = {}

    def _sesion(self, session_id: str) -> dict[str, set]:
        return self.sessions.setdefault(session_id, {"liked": set(), "disliked": set()})

    def registrar(self, session_id: str, product_id: str, liked: bool):
        session = self._sesion(session_id)
        (session["liked"] if liked else session["disliked"]).add(product_id)
        # un mismo producto no puede estar en ambos sets a la vez
        (session["disliked"] if liked else session["liked"]).discard(product_id)

    def tiene_feedback(self, session_id: str) -> bool:
        session = self.sessions.get(session_id)
        return bool(session and (session["liked"] or session["disliked"]))

    def aplicar_rocchio(self, session_id: str, query_vector: np.ndarray, store: FaissVectorStore,
                         alpha: float = 1.0, beta: float = 0.75, gamma: float = 0.15) -> np.ndarray:
        session = self.sessions.get(session_id)
        if not session or not (session["liked"] or session["disliked"]):
            return query_vector

        liked_vecs = [store.reconstruir_por_id_producto(pid) for pid in session["liked"]]
        liked_vecs = [v for v in liked_vecs if v is not None]
        disliked_vecs = [store.reconstruir_por_id_producto(pid) for pid in session["disliked"]]
        disliked_vecs = [v for v in disliked_vecs if v is not None]

        new_vector = alpha * query_vector
        if liked_vecs:
            new_vector = new_vector + beta * np.mean(liked_vecs, axis=0)
        if disliked_vecs:
            new_vector = new_vector - gamma * np.mean(disliked_vecs, axis=0)

        norm = np.linalg.norm(new_vector)
        return (new_vector / norm).astype("float32") if norm > 0 else query_vector
