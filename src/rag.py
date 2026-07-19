"""
Pipeline de Retrieval-Augmented Generation (requisito d del PDF).

Flujo: recibir consulta -> (opcional: reescribir con memoria, expandir con
LLM) -> recuperar top-k via FAISS -> re-rankear con CrossEncoder -> construir
contexto -> generar respuesta con Gemini -> devolver respuesta + evidencias
(texto, imagen, score) para trazabilidad.
"""
import random
import time

from google.genai import errors as genai_errors
from google.genai import types

from src import config
from src.llm_client import get_client
from src.query_expansion import multi_query_retrieve

RAG_SYSTEM_INSTRUCTION = """Eres un asistente de compras que responde preguntas sobre productos
basandote UNICAMENTE en las evidencias (fichas de producto) proporcionadas.

Reglas estrictas:
1. Responde solo con informacion contenida en las evidencias.
2. Si las evidencias no contienen informacion suficiente, indicalo explicitamente.
3. Si integras informacion de varios productos, se explicito al respecto.
4. No inventes precios, caracteristicas ni marcas que no esten en las evidencias.
5. Responde en el mismo idioma de la consulta del usuario.
"""


def build_prompt(query: str, evidences: list[dict], history_text: str = "") -> str:
    context_blocks = [f"[Evidencia {i}]\n{ev['document']}" for i, ev in enumerate(evidences, start=1)]
    context_text = "\n\n".join(context_blocks)

    history_block = f"Historial de la conversacion:\n{history_text}\n\n" if history_text else ""

    return f"""{history_block}Consulta del usuario: {query}

Evidencias recuperadas del catalogo:

{context_text}

Instruccion: Responde la consulta del usuario basandote unicamente en las evidencias anteriores,
siguiendo las reglas del sistema."""


def generate_answer(query: str, evidences: list[dict], history_text: str = "",
                     temperature: float = 0.2, max_retries: int = 5) -> str:
    prompt = build_prompt(query, evidences, history_text)
    client = get_client()

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=RAG_SYSTEM_INSTRUCTION,
                    temperature=temperature,
                    max_output_tokens=800,
                ),
            )
            return response.text
        except genai_errors.ServerError as e:
            wait = (2 ** attempt) + random.uniform(0, 1)
            print(f"[rag] intento {attempt + 1}/{max_retries} fallo ({e}); reintentando en {wait:.1f}s")
            time.sleep(wait)

    return "No fue posible generar una respuesta en este momento debido a alta demanda del servicio."


def rag_pipeline(query: str, embedder, store, reranker=None, memory=None, feedback=None,
                  session_id: str = "default", use_query_expansion: bool = True,
                  top_k_retrieve: int = config.TOP_K_RETRIEVE,
                  top_k_final: int = config.TOP_K_FINAL) -> dict:
    """
    Pipeline completo end-to-end. Todos los componentes de excelencia
    (memoria, expansion, feedback, reranking) son opcionales: si no se
    pasan, el pipeline degrada al comportamiento base del PDF.
    """
    effective_query = memory.contextualize_query(query) if memory else query

    if use_query_expansion:
        candidates = multi_query_retrieve(effective_query, embedder, store, top_k=top_k_retrieve)
    else:
        query_vector = embedder.encode_query(effective_query)
        if feedback is not None:
            query_vector = feedback.apply_rocchio(session_id, query_vector, store)
        candidates = store.search(query_vector, top_k=top_k_retrieve)

    evidences = reranker.rerank(effective_query, candidates, top_k=top_k_final) if reranker \
        else candidates[:top_k_final]

    history_text = memory.history_text() if memory else ""
    answer = generate_answer(effective_query, evidences, history_text=history_text)

    if memory:
        memory.add_turn(query, answer)

    return {
        "query": query,
        "effective_query": effective_query,
        "answer": answer,
        "evidences": evidences,
    }
