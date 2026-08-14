"""Configuracion central del proyecto: rutas, constantes y parametros de muestreo."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
IMAGES_DIR = DATA_DIR / "images"
INDEX_DIR = DATA_DIR / "faiss_index"

for d in (RAW_DIR, PROCESSED_DIR, IMAGES_DIR, INDEX_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Datasets fuente (Hugging Face) ---
ESCI_REPO = "tasksource/esci"
ESCI_TEST_FILES = [
    "data/test-00000-of-00004-d48474212b95f33b.parquet",
    "data/test-00001-of-00004-b7602f1b5c136953.parquet",
    "data/test-00002-of-00004-a81cff173329b486.parquet",
    "data/test-00003-of-00004-22af4ca7fa1313b2.parquet",
]
SQID_REPO = "crossingminds/shopping-queries-image-dataset"
SQID_PRODUCT_IMAGE_URLS_FILE = "data/product_image_urls.parquet"
SQID_QUERY_FEATURES_FILE = "data/query_features.parquet"
SQID_PRODUCT_FEATURES_FILE = "data/product_features.parquet"

# --- Muestreo del corpus (entorno CPU-only) ---
RANDOM_SEED = 42
N_QUERIES_SAMPLE = 300       # queries de ESCI (us, small_version=1) usadas como corpus + qrels
MAX_PRODUCTS_SAMPLE = 6500   # tope de productos unicos tras el muestreo por queries

# --- Modelos ---
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
GEMINI_MODEL = "gemini-3.5-flash"

# --- Recuperacion / RAG ---
TOP_K_RETRIEVE = 20
TOP_K_FINAL = 5
EVAL_K_VALUES = [3, 5, 10]

# --- Etiquetas de relevancia ESCI -> grado numerico (para NDCG) ---
ESCI_LABEL_TO_GRADE = {
    "Exact": 3,
    "Substitute": 2,
    "Complement": 1,
    "Irrelevant": 0,
}

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
