"""
Interfaz web conversacional (requisito e del PDF).

Chat tipo asistente de compras: el usuario escribe una consulta, el sistema
recupera productos relevantes (FAISS + CLIP, con query expansion y
re-ranking), genera una respuesta con el LLM, y muestra las evidencias
usadas (imagen, texto y score) para trazabilidad. Incluye botones de
Me gusta / No me gusta (relevance feedback) y memoria conversacional.

Ejecutar con: streamlit run app/streamlit_app.py
"""
import os
import sys
import uuid
from pathlib import Path

import streamlit as st

sys.path.append(str(Path(__file__).resolve().parent.parent))

# En Streamlit Community Cloud las API keys se configuran en "Secrets" (no hay
# archivo .env). Si esta corriendo ahi, se puentea a variable de entorno ANTES
# de importar src.config, que es quien la lee al importarse. En local, si no
# existe ningun secrets.toml, st.secrets lanza excepcion al accederlo: se
# ignora y se sigue usando el .env de siempre.
try:
    if "GEMINI_API_KEY" in st.secrets:
        os.environ["GEMINI_API_KEY"] = st.secrets["GEMINI_API_KEY"]
except st.errors.StreamlitSecretNotFoundError:
    pass

from src import config
from src.embeddings import ClipEmbedder
from src.feedback import FeedbackStore
from src.memory import ConversationMemory
from src.rag import rag_pipeline
from src.reranker import Reranker
from src.vector_store import FaissVectorStore

st.set_page_config(page_title="RAG Multimodal de Productos", page_icon="🛍️", layout="wide")


@st.cache_resource
def load_components():
    embedder = ClipEmbedder()
    store = FaissVectorStore.load(config.INDEX_DIR)
    reranker = Reranker()
    return embedder, store, reranker


embedder, store, reranker = load_components()

if "memory" not in st.session_state:
    st.session_state.memory = ConversationMemory()
if "feedback" not in st.session_state:
    st.session_state.feedback = FeedbackStore()
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []  # [{"role": "user"/"assistant", "content": ..., "result": {...}}]

st.title("🛍️ Asistente de compras multimodal (RAG)")
st.caption(
    "Corpus: Amazon Shopping Queries (ESCI) + SQID (imagenes) · Embeddings CLIP · Indice FAISS"
)

with st.sidebar:
    st.header("Opciones")
    use_expansion = st.checkbox("Query Expansion", value=True)
    use_memory = st.checkbox("Memoria conversacional", value=True)
    top_k_retrieve = st.slider("Top-k recuperacion (FAISS)", 5, 50, config.TOP_K_RETRIEVE)
    top_k_final = st.slider("Top-k final (tras re-ranking)", 1, 10, config.TOP_K_FINAL)

    if st.button("Limpiar conversacion"):
        st.session_state.memory.clear()
        st.session_state.chat_log = []
        st.rerun()

for turn in st.session_state.chat_log:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and turn.get("result"):
            result = turn["result"]
            with st.expander(f"Ver evidencias ({len(result['evidences'])})"):
                for i, ev in enumerate(result["evidences"]):
                    cols = st.columns([1, 3])
                    with cols[0]:
                        # image_path solo existe si data/images/ esta presente (dev local);
                        # en despliegue (sin las imagenes commiteadas) se usa image_url remota.
                        image_source = ev.get("image_path") if ev.get("image_path") and Path(ev["image_path"]).exists() \
                            else ev.get("image_url")
                        if image_source:
                            st.image(image_source, width=120)
                    with cols[1]:
                        score = ev.get("rerank_score", ev.get("score"))
                        st.markdown(f"**[{i + 1}] score: {score:.4f}**")
                        st.text(ev["document"][:300])
                        fb_cols = st.columns(2)
                        key_base = f"{turn['content'][:20]}_{ev['product_id']}_{i}"
                        if fb_cols[0].button("👍 Me gusta", key=f"like_{key_base}"):
                            st.session_state.feedback.record(
                                st.session_state.session_id, ev["product_id"], liked=True
                            )
                            st.success("Feedback registrado")
                        if fb_cols[1].button("👎 No me gusta", key=f"dislike_{key_base}"):
                            st.session_state.feedback.record(
                                st.session_state.session_id, ev["product_id"], liked=False
                            )
                            st.warning("Feedback registrado")

query = st.chat_input("Pregunta por un producto (ej: 'wireless bluetooth headphones')")

if query:
    st.session_state.chat_log.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Buscando y generando respuesta..."):
            result = rag_pipeline(
                query,
                embedder=embedder,
                store=store,
                reranker=reranker,
                memory=st.session_state.memory if use_memory else None,
                feedback=st.session_state.feedback,
                session_id=st.session_state.session_id,
                use_query_expansion=use_expansion,
                top_k_retrieve=top_k_retrieve,
                top_k_final=top_k_final,
            )
        st.markdown(result["answer"])
        if result["effective_query"] != query:
            st.caption(f"Consulta reformulada (memoria): *{result['effective_query']}*")

    st.session_state.chat_log.append(
        {"role": "assistant", "content": result["answer"], "result": result}
    )
    st.rerun()
