"""
Модуль метрик качества для оценки рекомендательных систем.
Включает глобальные метрики классификации (ROC-AUC, LogLoss, PR-AUC)
и пользовательские метрики качества ранжирования (NDCG@K, Recall@K, MAP@K, MRR@K).
"""

from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score, average_precision_score


def compute_ranking_metrics_per_user(
    df: pd.DataFrame,
    score_col: str,
    label_col: str = "label",
    user_col: str = "user_id",
    ks: List[int] = [5, 10, 20],
) -> Dict[str, float]:
    """
    Вычисляет метрики ранжирования (NDCG@K, Recall@K, MAP@K, MRR@K),
    усреднённые по всем пользователям, у которых есть хотя бы 1 позитивное действие в таргете,
    а также глобальные метрики классификации (ROC-AUC, PR-AUC, LogLoss).
    """
    y_true = df[label_col].values.astype(np.float64)
    y_score = df[score_col].values.astype(np.float64)

    # Защита от экстремальных вероятностей в LogLoss
    y_prob = np.clip(y_score, 1e-7, 1.0 - 1e-7)

    results: Dict[str, float] = {}
    if len(np.unique(y_true)) > 1:
        results["ROC-AUC"] = float(roc_auc_score(y_true, y_score))
        results["PR-AUC"] = float(average_precision_score(y_true, y_score))
        results["LogLoss"] = float(log_loss(y_true, y_prob))
    else:
        results["ROC-AUC"] = 0.5
        results["PR-AUC"] = 0.0
        results["LogLoss"] = 1.0

    # Сортируем кандидатов внутри каждого пользователя по убыванию предсказанного скора
    # Для стабильности при равных скорах используем mergesort
    order = np.lexsort((-y_score, df[user_col].values))
    sorted_users = df[user_col].values[order]
    sorted_labels = y_true[order]

    # Находим границы групп по user_id
    _, group_starts, group_counts = np.unique(
        sorted_users, return_index=True, return_counts=True
    )

    max_k = max(ks)
    discounts = 1.0 / np.log2(np.arange(2, max_k + 2))

    ndcg_lists = {k: [] for k in ks}
    recall_lists = {k: [] for k in ks}
    ap_lists = {k: [] for k in ks}
    mrr_lists = {k: [] for k in ks}

    for start, count in zip(group_starts, group_counts):
        user_labels = sorted_labels[start : start + count]
        total_pos = int(np.sum(user_labels))
        if total_pos == 0:
            continue

        for k in ks:
            top_k = user_labels[:k]
            hits_k = np.sum(top_k)

            # Recall@K
            recall_lists[k].append(float(hits_k / total_pos))

            # NDCG@K
            dcg = float(np.sum(top_k * discounts[: len(top_k)]))
            ideal_len = min(total_pos, k)
            idcg = float(np.sum(discounts[:ideal_len]))
            ndcg_lists[k].append(dcg / idcg if idcg > 0 else 0.0)

            # MRR@K и MAP@K
            if hits_k > 0:
                pos_indices = np.where(top_k > 0)[0]
                mrr_lists[k].append(float(1.0 / (pos_indices[0] + 1)))

                precisions_at_hits = (np.arange(1, len(pos_indices) + 1)) / (
                    pos_indices + 1.0
                )
                ap = float(np.sum(precisions_at_hits) / min(total_pos, k))
                ap_lists[k].append(ap)
            else:
                mrr_lists[k].append(0.0)
                ap_lists[k].append(0.0)

    for k in ks:
        results[f"NDCG@{k}"] = float(np.mean(ndcg_lists[k])) if ndcg_lists[k] else 0.0
        results[f"Recall@{k}"] = (
            float(np.mean(recall_lists[k])) if recall_lists[k] else 0.0
        )
        results[f"MAP@{k}"] = float(np.mean(ap_lists[k])) if ap_lists[k] else 0.0
        results[f"MRR@{k}"] = float(np.mean(mrr_lists[k])) if mrr_lists[k] else 0.0

    return results
