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
from google.genai import types

from src import config
from src.llm_client import get_client

EXPANSION_SYSTEM_INSTRUCTION = """Eres un asistente de busqueda de productos de e-commerce.
Dada una consulta de un usuario, genera reformulaciones alternativas que ayuden a recuperar
productos relevantes: sinonimos, variantes mas especificas o mas generales, y formas
alternativas de nombrar el mismo producto.

Responde EXCLUSIVAMENTE con las reformulaciones, una por linea, sin numeracion ni texto
adicional. Genera como maximo 3 reformulaciones."""


def expand_query(query: str, n_expansions: int = 3) -> list[str]:
    """Devuelve [query_original, reformulacion_1, ...]. Nunca lanza si falla el LLM."""
    variants = [query]
    try:
        client = get_client()
        response = client.models.generate_content(
            model=config.GEMINI_MODEL,
            contents=f"Consulta original: {query}",
            config=types.GenerateContentConfig(
                system_instruction=EXPANSION_SYSTEM_INSTRUCTION,
                temperature=0.7,
                max_output_tokens=200,
            ),
        )
        lines = [l.strip("- ").strip() for l in response.text.strip().splitlines() if l.strip()]
        variants.extend(lines[:n_expansions])
    except Exception as e:
        print(f"[query_expansion] fallo la expansion, se usa solo la query original: {e}")
    return variants


def multi_query_retrieve(query: str, embedder, store, top_k: int = config.TOP_K_RETRIEVE) -> list[dict]:
    """Recupera con la query original + expansiones y fusiona por product_id (max score)."""
    variants = expand_query(query)

    best_by_product: dict[str, dict] = {}
    for variant in variants:
        variant_embedding = embedder.encode_query(variant)
        for hit in store.search(variant_embedding, top_k=top_k):
            pid = hit["product_id"]
            if pid not in best_by_product or hit["score"] > best_by_product[pid]["score"]:
                best_by_product[pid] = hit

    merged = sorted(best_by_product.values(), key=lambda h: h["score"], reverse=True)
    return merged[:top_k]
