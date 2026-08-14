"""
Construccion del corpus final a partir del dataframe unido (ESCI + SQID):

1. Muestrea un subconjunto de queries (para mantener el computo de embeddings
   CLIP viable en CPU) y conserva TODOS los productos asociados a esas queries,
   de forma que cada query de evaluacion tenga su conjunto completo de
   candidatos con juicio de relevancia (qrels reales, no simulados).
2. Deduplica productos (un mismo producto puede aparecer para varias queries).
3. Descarga localmente la imagen de cada producto (para poder generar el
   embedding CLIP de imagen y para mostrarla como evidencia en la interfaz).
4. Construye el texto final de cada documento (titulo + descripcion + bullets
   + marca + color).
"""
import concurrent.futures as cf
from pathlib import Path

import pandas as pd
import requests
from PIL import Image
from tqdm.auto import tqdm

from src import config


def muestrear_queries(df: pd.DataFrame, n_queries: int = config.N_QUERIES_SAMPLE,
                       seed: int = config.RANDOM_SEED) -> pd.DataFrame:
    """Selecciona n_queries al azar y devuelve todas las filas (query, producto) asociadas."""
    unique_queries = df["query_id"].drop_duplicates()
    sampled_ids = unique_queries.sample(n=min(n_queries, len(unique_queries)), random_state=seed)
    sample = df[df["query_id"].isin(sampled_ids)].copy()

    if sample["product_id"].nunique() > config.MAX_PRODUCTS_SAMPLE:
        keep_products = (
            sample["product_id"].drop_duplicates()
            .sample(n=config.MAX_PRODUCTS_SAMPLE, random_state=seed)
        )
        sample = sample[sample["product_id"].isin(keep_products)]

    return sample.reset_index(drop=True)


def construir_tabla_productos(sample: pd.DataFrame) -> pd.DataFrame:
    """Colapsa el dataframe (query, producto) a nivel de producto unico (el corpus indexable)."""
    products = (
        sample.drop_duplicates(subset=["product_id"])
        [["product_id", "product_title", "product_description", "product_bullet_point",
          "product_brand", "product_color", "product_text", "image_url"]]
        .reset_index(drop=True)
    )

    def construir_documento(row):
        parts = [f"Title: {row['product_title']}"]
        if pd.notna(row.get("product_brand")):
            parts.append(f"Brand: {row['product_brand']}")
        if pd.notna(row.get("product_color")):
            parts.append(f"Color: {row['product_color']}")
        if pd.notna(row.get("product_bullet_point")):
            parts.append(f"Features: {row['product_bullet_point']}")
        if pd.notna(row.get("product_description")):
            parts.append(f"Description: {row['product_description']}")
        return "\n".join(parts)

    products["document"] = products.apply(construir_documento, axis=1)

    # Descarta fichas con titulo de una sola palabra y sin marca/descripcion/features:
    # son registros de metadata degenerada del catalogo real de Amazon (ej. "Job",
    # "Episode 1") que no aportan senal de recuperacion y ensucian el ranking.
    has_extra_info = (
        products["product_brand"].notna()
        | products["product_bullet_point"].notna()
        | products["product_description"].notna()
    )
    title_word_count = products["product_title"].str.split().str.len()
    products = products[has_extra_info | (title_word_count > 1)].reset_index(drop=True)

    return products


def _descargar_una_imagen(product_id: str, url: str, images_dir: Path) -> str | None:
    if not isinstance(url, str) or not url:
        return None
    ext = Path(url.split("?")[0]).suffix or ".jpg"
    dest = images_dir / f"{product_id}{ext}"
    if dest.exists():
        return str(dest)
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        # valida que sea una imagen legible
        Image.open(dest).verify()
        return str(dest)
    except Exception:
        dest.unlink(missing_ok=True)
        return None


def descargar_imagenes(products: pd.DataFrame, images_dir: Path = config.IMAGES_DIR,
                        max_workers: int = 16) -> pd.DataFrame:
    """Descarga en paralelo las imagenes de producto; agrega columna 'image_path' (None si fallo)."""
    images_dir.mkdir(parents=True, exist_ok=True)
    paths = [None] * len(products)

    with cf.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_descargar_una_imagen, row.product_id, row.image_url, images_dir): i
            for i, row in enumerate(products.itertuples(index=False))
        }
        for future in tqdm(cf.as_completed(futures), total=len(futures), desc="Descargando imagenes"):
            i = futures[future]
            paths[i] = future.result()

    products = products.copy()
    products["image_path"] = paths
    return products


def construir_corpus(force_rebuild: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pipeline completo: carga datos unidos -> muestrea queries -> arma tabla de
    productos con imagenes locales -> devuelve (products, qrels_sample).

    qrels_sample conserva el nivel (query, producto, esci_label) para evaluacion.
    """
    from src.data_loading import construir_dataset_unido

    products_cache = config.PROCESSED_DIR / "products.parquet"
    qrels_cache = config.PROCESSED_DIR / "qrels_sample.parquet"

    if not force_rebuild and products_cache.exists() and qrels_cache.exists():
        return pd.read_parquet(products_cache), pd.read_parquet(qrels_cache)

    joined = construir_dataset_unido()
    sample = muestrear_queries(joined)

    products = construir_tabla_productos(sample)
    products = descargar_imagenes(products)
    products = products[products["image_path"].notna()].reset_index(drop=True)

    qrels_sample = sample[sample["product_id"].isin(products["product_id"])][
        ["query_id", "query", "product_id", "esci_label"]
    ].reset_index(drop=True)

    products.to_parquet(products_cache)
    qrels_sample.to_parquet(qrels_cache)
    return products, qrels_sample


if __name__ == "__main__":
    products, qrels = construir_corpus()
    print("Productos en el corpus:", products.shape)
    print("Pares (query, producto) para evaluacion:", qrels.shape)
    print("Queries unicas para evaluacion:", qrels.query_id.nunique())
