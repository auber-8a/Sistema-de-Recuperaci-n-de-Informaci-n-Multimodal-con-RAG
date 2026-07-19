"""
Enriquecimiento OPCIONAL del corpus con Amazon Reviews 2023 (McAuley-Lab).

SQID + ESCI (data_loading.py) es el eje del corpus: aporta el join
query-producto-imagen-relevancia que se usa para indexar y evaluar. Amazon
Reviews 2023 no tiene queries de busqueda ni qrels, asi que se usa solo como
fuente adicional de texto por producto (bullet points/reviews reales) para
los ASIN que coincidan, cuando se quiera enriquecer el documento de un
producto especifico con mas contexto.

Los archivos de metadata por categoria son grandes (~200 MB o mas incluso
para categorias "chicas"), por lo que esta funcion los descarga una vez
(cache de huggingface_hub) y los recorre linea por linea sin cargarlos
completos en memoria, quedandose solo con los ASIN que interesan.
"""
import json

from huggingface_hub import hf_hub_download

from src import config


def fetch_metadata_for_asins(category: str, target_asins: set[str]) -> dict[str, dict]:
    """
    Descarga (o reutiliza el cache) el archivo de metadata de una categoria de
    Amazon Reviews 2023 y devuelve {parent_asin: metadata_dict} solo para los
    ASIN presentes en target_asins.

    Categorias disponibles, p.ej.: 'All_Beauty', 'Electronics',
    'Tools_and_Home_Improvement', 'Toys_and_Games', etc. (ver
    raw/meta_categories/ en el repo de HuggingFace).
    """
    filename = f"raw/meta_categories/meta_{category}.jsonl"
    local_path = hf_hub_download(config.AMAZON_REVIEWS_2023_REPO, filename, repo_type="dataset")

    found = {}
    with open(local_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = record.get("parent_asin")
            if asin in target_asins:
                found[asin] = record
                if len(found) == len(target_asins):
                    break
    return found


def enrich_products(products_df, category: str):
    """
    Agrega columna 'amazon_reviews_extra' con texto adicional (features +
    description del meta de Amazon Reviews 2023) para los productos del
    corpus cuyo product_id (ASIN) exista en la categoria indicada.
    """
    target_asins = set(products_df["product_id"])
    metadata_by_asin = fetch_metadata_for_asins(category, target_asins)

    def extra_text(product_id):
        meta = metadata_by_asin.get(product_id)
        if not meta:
            return None
        parts = []
        if meta.get("features"):
            parts.append("Extra features: " + " | ".join(meta["features"]))
        if meta.get("description"):
            parts.append("Extra description: " + " | ".join(meta["description"]))
        return "\n".join(parts) if parts else None

    products_df = products_df.copy()
    products_df["amazon_reviews_extra"] = products_df["product_id"].apply(extra_text)
    print(f"Coincidencias en categoria '{category}': "
          f"{products_df['amazon_reviews_extra'].notna().sum()}/{len(products_df)}")
    return products_df
