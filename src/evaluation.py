"""
Evaluacion experimental (requisito f del PDF): Precision@k, Recall@k, NDCG@k
contra los qrels reales derivados de las etiquetas esci_label del dataset
ESCI (Exact/Substitute/Complement/Irrelevant).

Se consideran "relevantes" (para Precision/Recall binarios) los productos
con esci_label distinto de 'Irrelevant'. Para NDCG se usa el grado numerico
completo (ESCI_LABEL_TO_GRADE) para aprovechar la relevancia graduada.
"""
import numpy as np
import pandas as pd

from src import config


def _relevant_set(qrels: pd.DataFrame, query_id) -> set:
    rows = qrels[qrels["query_id"] == query_id]
    return set(rows[rows["esci_label"] != "Irrelevant"]["product_id"])


def _grades_by_product(qrels: pd.DataFrame, query_id) -> dict:
    rows = qrels[qrels["query_id"] == query_id]
    return {
        row.product_id: config.ESCI_LABEL_TO_GRADE.get(row.esci_label, 0)
        for row in rows.itertuples(index=False)
    }


def precision_at_k(ranked_product_ids: list[str], relevant: set, k: int) -> float:
    top_k = ranked_product_ids[:k]
    if not top_k:
        return 0.0
    return sum(1 for pid in top_k if pid in relevant) / len(top_k)


def recall_at_k(ranked_product_ids: list[str], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    top_k = ranked_product_ids[:k]
    return sum(1 for pid in top_k if pid in relevant) / len(relevant)


def dcg_at_k(ranked_product_ids: list[str], grades: dict, k: int) -> float:
    top_k = ranked_product_ids[:k]
    return sum(
        grades.get(pid, 0) / np.log2(i + 2)
        for i, pid in enumerate(top_k)
    )


def ndcg_at_k(ranked_product_ids: list[str], grades: dict, k: int) -> float:
    dcg = dcg_at_k(ranked_product_ids, grades, k)
    ideal_order = sorted(grades.values(), reverse=True)[:k]
    idcg = sum(g / np.log2(i + 2) for i, g in enumerate(ideal_order))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_query(ranked_product_ids: list[str], qrels: pd.DataFrame, query_id,
                    k_values: list[int] = config.EVAL_K_VALUES) -> dict:
    relevant = _relevant_set(qrels, query_id)
    grades = _grades_by_product(qrels, query_id)

    metrics = {"query_id": query_id}
    for k in k_values:
        metrics[f"precision@{k}"] = precision_at_k(ranked_product_ids, relevant, k)
        metrics[f"recall@{k}"] = recall_at_k(ranked_product_ids, relevant, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(ranked_product_ids, grades, k)
    return metrics


def evaluate_system(search_fn, queries: pd.DataFrame, qrels: pd.DataFrame,
                     k_values: list[int] = config.EVAL_K_VALUES) -> pd.DataFrame:
    """
    queries: dataframe con columnas ['query_id', 'query'] (una fila por query unica).
    search_fn: funcion query(str) -> lista de product_id rankeados (orden de mayor a menor score).
    """
    rows = []
    for row in queries.itertuples(index=False):
        ranked_ids = search_fn(row.query)
        rows.append(evaluate_query(ranked_ids, qrels, row.query_id, k_values))

    results = pd.DataFrame(rows)
    return results


def summarize(results: pd.DataFrame) -> pd.Series:
    return results.drop(columns=["query_id"]).mean()
