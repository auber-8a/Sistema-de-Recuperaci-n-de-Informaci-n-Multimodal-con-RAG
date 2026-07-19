"""
Carga y union de las dos fuentes de datos del corpus multimodal:

- ESCI (Amazon Shopping Queries Dataset, via el mirror parquet 'tasksource/esci'):
  aporta el texto de las queries, el texto de los productos y las etiquetas
  de relevancia (esci_label) que usamos como qrels.
- SQID (crossingminds/shopping-queries-image-dataset): aporta las URLs de
  imagen de cada producto (subconjunto test/us/small_version=1 del ESCI).

El join se hace por product_id (ASIN). Ver informe tecnico para el analisis
de cobertura (95%+ de productos con imagen disponible).
"""
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

from src import config


def _download(repo_id: str, filename: str) -> str:
    return hf_hub_download(repo_id, filename, repo_type="dataset")


def load_esci_us_small(cache_path: Path = config.RAW_DIR / "esci_us_small.parquet") -> pd.DataFrame:
    """Descarga (o lee de cache) el subconjunto ESCI test/us/small_version=1."""
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    paths = [_download(config.ESCI_REPO, f) for f in config.ESCI_TEST_FILES]
    df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)

    df_us = df[(df["small_version"] == 1) & (df["product_locale"] == "us")].copy()
    df_us = df_us.reset_index(drop=True)
    df_us.to_parquet(cache_path)
    return df_us


def load_sqid_image_urls() -> pd.DataFrame:
    path = _download(config.SQID_REPO, config.SQID_PRODUCT_IMAGE_URLS_FILE)
    return pd.read_parquet(path)


def load_sqid_query_features() -> pd.DataFrame:
    """Embeddings CLIP de queries precalculados por SQID (referencia/validacion)."""
    path = _download(config.SQID_REPO, config.SQID_QUERY_FEATURES_FILE)
    return pd.read_parquet(path)


def build_joined_dataset(cache_path: Path = config.RAW_DIR / "joined_full.parquet") -> pd.DataFrame:
    """
    Devuelve el dataframe unido ESCI + SQID (imagen), a nivel de par (query, producto).

    Columnas relevantes: query, query_id, product_id, esci_label, product_title,
    product_description, product_bullet_point, product_brand, product_color,
    product_text, image_url.
    """
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    esci = load_esci_us_small()
    images = load_sqid_image_urls()

    merged = esci.merge(images, on="product_id", how="left")
    merged.to_parquet(cache_path)
    return merged


if __name__ == "__main__":
    df = build_joined_dataset()
    print("Filas totales:", df.shape)
    print("Productos unicos:", df.product_id.nunique())
    print("Queries unicas:", df.query_id.nunique())
    print("Cobertura de imagen:", df.image_url.notna().mean())
