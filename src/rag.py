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
from src.llm_client import obtener_cliente
from src.query_expansion import recuperar_multi_consulta

RAG_SYSTEM_INSTRUCTION = """Eres un asistente de compras que responde preguntas sobre productos
basandote UNICAMENTE en las evidencias (fichas de producto) proporcionadas.

Reglas estrictas:
1. Responde solo con informacion contenida en las evidencias.
2. Si las evidencias no contienen informacion suficiente, indicalo explicitamente.
3. Si integras informacion de varios productos, se explicito al respecto.
4. No inventes precios, caracteristicas ni marcas que no esten en las evidencias.
5. Responde en el mismo idioma de la consulta del usuario.
6. Si hay evidencias relevantes, empieza tu respuesta con una linea breve (1 sola frase) que
   resuma cuantos productos se encontraron y de que tipo son en general. Luego responde la
   consulta puntual del usuario. No hagas un analisis extenso ni compares producto por
   producto salvo que el usuario lo pida explicitamente.
"""


def construir_prompt(query: str, evidences: list[dict], history_text: str = "") -> str:
    context_blocks = [f"[Evidencia {i}]\n{ev['document']}" for i, ev in enumerate(evidences, start=1)]
    context_text = "\n\n".join(context_blocks)

    history_block = f"Historial de la conversacion:\n{history_text}\n\n" if history_text else ""

    return f"""{history_block}Consulta del usuario: {query}

Evidencias recuperadas del catalogo:

{context_text}

Instruccion: Responde la consulta del usuario basandote unicamente en las evidencias anteriores,
siguiendo las reglas del sistema."""


def generar_respuesta(query: str, evidences: list[dict], history_text: str = "",
                       temperature: float = 0.2, max_retries: int = 1) -> str:
    prompt = construir_prompt(query, evidences, history_text)
    client = obtener_cliente()

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


def pipeline_rag(query: str, embedder, store, reranker=None, memory=None, feedback=None,
                  session_id: str = "default", use_query_expansion: bool = False,
                  top_k_retrieve: int = config.TOP_K_RETRIEVE,
                  top_k_final: int = config.TOP_K_FINAL) -> dict:
    """
    Pipeline completo end-to-end. Todos los componentes de excelencia
    (memoria, expansion, feedback, reranking) son opcionales: si no se
    pasan, el pipeline degrada al comportamiento base del PDF.
    """
    effective_query = memory.contextualizar_consulta(query) if memory else query
    query_variants: list[str] = []

    if use_query_expansion:
        candidates, query_variants = recuperar_multi_consulta(effective_query, embedder, store, top_k=top_k_retrieve)
    else:
        query_vector = embedder.codificar_consulta(effective_query)
        if feedback is not None:
            query_vector = feedback.aplicar_rocchio(session_id, query_vector, store)
        candidates = store.buscar(query_vector, top_k=top_k_retrieve)

    evidences = reranker.rerankear(effective_query, candidates, top_k=top_k_final) if reranker \
        else candidates[:top_k_final]

    history_text = memory.texto_historial() if memory else ""
    answer = generar_respuesta(effective_query, evidences, history_text=history_text)

    if memory:
        memory.agregar_turno(query, answer)

    return {
        "query": query,
        "effective_query": effective_query,
        "answer": answer,
        "evidences": evidences,
        "query_variants": [v for v in query_variants if v != effective_query],
    }
