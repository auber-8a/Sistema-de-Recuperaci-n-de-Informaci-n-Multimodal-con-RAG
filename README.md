# Sistema de Recuperación de Información Multimodal con RAG

Proyecto final de la asignatura Recuperación de Información: un sistema RAG multimodal
(texto + imagen) sobre un catálogo de productos de Amazon, con recuperación vectorial en
FAISS, embeddings CLIP, generación con Gemini y una interfaz de chat en Streamlit.

## Corpus

- **[ESCI (Amazon Shopping Queries Dataset)](https://github.com/amazon-science/esci-data)**,
  vía el mirror parquet [`tasksource/esci`](https://huggingface.co/datasets/tasksource/esci):
  texto de las queries, texto de los productos y etiquetas de relevancia (`esci_label`)
  usadas como *qrels*. Se usa el subconjunto `product_locale=us`, `small_version=1`.
- **[SQID](https://huggingface.co/datasets/crossingminds/shopping-queries-image-dataset)**:
  URLs de imagen para los productos de ese mismo subconjunto de ESCI.

El join se hace por `product_id` (ASIN). Se muestrea un subconjunto de queries (por defecto
150) conservando todos sus productos asociados, para que cada query de evaluación tenga su
conjunto completo de candidatos con juicio de relevancia real.

## Estructura del proyecto

```
src/
  config.py                    # rutas, constantes, parametros de muestreo
  data_loading.py               # descarga y join ESCI + SQID
  corpus_builder.py              # muestreo de queries, descarga de imagenes, doc final
  embeddings.py                   # wrapper CLIP (texto, imagen, query, documento multimodal)
  vector_store.py                  # indice FAISS (IndexFlatIP / similitud coseno)
  reranker.py                       # re-ranking con CrossEncoder [excelencia]
  query_expansion.py                 # expansion de consultas con LLM [excelencia]
  memory.py                           # memoria conversacional [excelencia]
  feedback.py                          # relevance feedback (Rocchio) [excelencia]
  rag.py                                # prompt, generacion (Gemini), pipeline RAG completo
  evaluation.py                          # Precision@k, Recall@k, NDCG@k contra qrels
  llm_client.py                            # cliente Gemini compartido
app/
  streamlit_app.py               # interfaz de chat con evidencias y feedback
notebooks/
  proyecto_rag_multimodal.ipynb  # notebook principal que orquesta el pipeline por fases
data/                             # generado localmente (no versionado): raw/, processed/,
                                  # images/, faiss_index/
```

## Instalación

Requiere Python 3.11+ (probado en 3.13).

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

Crear un archivo `.env` en la raíz del proyecto con tu API key de Gemini
(usada para la generación RAG, la query expansion y la memoria conversacional):

```
GEMINI_API_KEY=tu_api_key_aqui
```

## Ejecución

### 1. Construir el corpus, el índice FAISS y probar el pipeline

Abrir y ejecutar `notebooks/proyecto_rag_multimodal.ipynb` de principio a fin. Esto:

1. Descarga y une ESCI + SQID (`src/data_loading.py`).
2. Muestrea queries/productos y descarga las imágenes (`src/corpus_builder.py`).
3. Genera embeddings CLIP multimodales (`src/embeddings.py`).
4. Construye y guarda el índice FAISS en `data/faiss_index/` (`src/vector_store.py`).
5. Corre el pipeline RAG completo (recuperación + re-ranking + generación) y muestra
   evidencias (texto, imagen, score).
6. Evalúa el sistema (Precision@k, Recall@k, NDCG@k) con y sin re-ranking.
7. Demuestra las funcionalidades de excelencia (query expansion, relevance feedback,
   memoria conversacional).

La primera ejecución tarda varios minutos (descarga de datasets/imágenes + cómputo de
embeddings en CPU). Las ejecuciones siguientes reutilizan la caché en `data/`.

### 2. Levantar la interfaz de chat

Una vez generado el índice FAISS (paso anterior):

```bash
streamlit run app/streamlit_app.py
```

La interfaz permite hacer consultas conversacionales, ver la respuesta generada, inspeccionar
las evidencias (productos, imágenes y score de recuperación), calificarlas con
👍/👎 (relevance feedback) y activar/desactivar query expansion y memoria conversacional.

## Funcionalidades de excelencia implementadas

| Funcionalidad | Módulo |
|---|---|
| Re-ranking (CrossEncoder) | `src/reranker.py` |
| Query Expansion (LLM) | `src/query_expansion.py` |
| Relevance Feedback (Rocchio) | `src/feedback.py` |
| Memoria conversacional | `src/memory.py` |

## Evaluación

Las métricas (Precision@k, Recall@k, NDCG@k, k ∈ {3, 5, 10}) se calculan contra los qrels
reales derivados de `esci_label` (Exact=3, Substitute=2, Complement=1, Irrelevant=0 para
NDCG; Exact/Substitute/Complement se consideran relevantes para Precision/Recall). Ver el
notebook (Fase F) y el informe técnico para el análisis de resultados.
