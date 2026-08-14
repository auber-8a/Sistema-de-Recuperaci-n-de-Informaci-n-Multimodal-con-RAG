"""
Funcionalidad de excelencia: Query Expansion (+15).

Usa el LLM (Gemini) para generar reformulaciones/sinonimos de la consulta
original del usuario. La recuperacion se hace luego con multi-query: se
busca en FAISS con la query original y con cada reformulacion, y se
fusionan los resultados quedandonos con el mejor score por producto
(similar a un "OR" de consultas con maximo de similitud).

Si el LLM no esta disponible (sin API key), degrada de forma segura
devolviendo solo la consulta original.
"""
import random
import time

from google.genai import errors as genai_errors
from google.genai import types

from src import config
from src.llm_client import obtener_cliente

EXPANSION_SYSTEM_INSTRUCTION = """Eres un asistente de busqueda de productos de e-commerce.
Dada una consulta de un usuario, genera reformulaciones alternativas que ayuden a recuperar
productos relevantes: sinonimos, variantes mas especificas o mas generales, y formas
alternativas de nombrar el mismo producto (traducciones incluidas si aplica).

Cada reformulacion debe ser, en si misma, una consulta de busqueda corta y valida (nunca una
explicacion, nota o fragmento de oracion). Genera como maximo 3 reformulaciones."""


def _es_variante_valida(text: str, original: str) -> bool:
    """Descarta vacios, restos sin contenido real y duplicados triviales de la consulta original."""
    if not text or not any(c.isalnum() for c in text):
        return False
    return text.strip().lower() != original.strip().lower()


def expandir_consulta(query: str, n_expansions: int = 3, max_retries: int = 2) -> list[str]:
    """Devuelve [query_original, reformulacion_1, ...]. Nunca lanza si falla el LLM."""
    variants = [query]
    client = obtener_cliente()

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=f"Consulta original: {query}",
                config=types.GenerateContentConfig(
                    system_instruction=EXPANSION_SYSTEM_INSTRUCTION,
                    temperature=0.7,
                    max_output_tokens=300,
                    response_mime_type="application/json",
                    response_schema=list[str],
                    # sin "thinking": la tarea es trivial y el razonamiento solo
                    # agregaba ~1 min de latencia por consulta sin mejorar el resultado
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                ),
            )
            # salida estructurada (JSON): evita el parseo fragil de texto libre
            # (numeracion, vinetas, o restos de markdown que devolvia el modelo)
            if response.parsed is None:
                # el modelo corto la respuesta antes de cerrar el JSON (MAX_TOKENS) u otro
                # motivo de parseo fallido; no es un error de red, pero conviene reintentar
                raise ValueError(f"respuesta sin JSON valido (finish_reason={response.candidates[0].finish_reason})")
            reformulations = response.parsed
            variants.extend([r for r in reformulations if _es_variante_valida(r, query)][:n_expansions])
            return variants
        except (genai_errors.ServerError, ValueError) as e:
            # 503 por alta demanda o JSON incompleto: ambos transitorios, vale un reintento corto
            if attempt + 1 >= max_retries:
                print(f"[query_expansion] fallo la expansion, se usa solo la query original: {e}")
                break
            wait = 1.5 * (attempt + 1) + random.uniform(0, 0.5)
            time.sleep(wait)
        except Exception as e:
            print(f"[query_expansion] fallo la expansion, se usa solo la query original: {e}")
            break

    return variants


def recuperar_multi_consulta(query: str, embedder, store,
                              top_k: int = config.TOP_K_RETRIEVE) -> tuple[list[dict], list[str]]:
    """Recupera con la query original + expansiones y fusiona por product_id (max score).

    Devuelve (resultados, variantes) para poder mostrar en la UI que reformulaciones se usaron.
    """
    variants = expandir_consulta(query)

    best_by_product: dict[str, dict] = {}
    for variant in variants:
        variant_embedding = embedder.codificar_consulta(variant)
        for hit in store.buscar(variant_embedding, top_k=top_k):
            pid = hit["product_id"]
            if pid not in best_by_product or hit["score"] > best_by_product[pid]["score"]:
                best_by_product[pid] = hit

    merged = sorted(best_by_product.values(), key=lambda h: h["score"], reverse=True)
    return merged[:top_k], variants
