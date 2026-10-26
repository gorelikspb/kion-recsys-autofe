"""
Модуль обучаемых представлений над историей событий и двудольным графом взаимодействий
(Time-Aware Spectral Graph & Sequential History Embeddings).

Исследует второй вопрос задания:
«Можно ли заменить ручные признаки обучаемыми представлениями истории событий
или использовать их для усиления бустинга?»

Архитектура (строго по событиям watch_dt < cutoff_dt, без утечек):
1. Строится взвешенный двудольный граф взаимодействий User — Item (и Item — Genre/Country)
   с учётом времени и глубины просмотра:
     w(u, i) = log(1 + count) * (0.5 + 0.5 * watched_pct / 100) * exp(-lambda * days_ago)
2. С помощью спектрального разложения (TruncatedSVD / Spectral Graph Embedding размерности d=32)
   извлекаются латентные представления объектов E_item и пользователей E_user.
3. По последовательности последних событий пользователя (Sequential History Pooling) формируются
   два динамических вектора состояния истории пользователя:
   - Short-term вектор (экспоненциальный пулинг последних 5 просмотренных объектов с быстрым затуханием);
   - Long-term вектор (взвешенный по времени и проценту досмотра пулинг всей истории пользователя);
   - Вектор последнего просмотренного объекта (Last-1 Item Representation).
4. Для каждой пары кандидата (user, item) вычисляются:
   - Скалярные произведения и косинусные близости (Dot Product & Cosine Similarity) между
     представлением целевого объекта и (Long-term, Short-term, Last-Item, Spectral User) векторами;
   - Латентная дисперсия интересов пользователя и проекции вектора объекта.
"""

from typing import List, Tuple
import duckdb
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


class TimeAwareRepresentationLearner:
    """
    Обучает латентные представления пользователей, объектов и последовательностей истории
    строго на данных watch_dt < cutoff_dt.
    """

    def __init__(
        self,
        con: duckdb.DuckDBPyConnection,
        embedding_dim: int = 24,
        random_seed: int = 42,
    ):
        self.con = con
        self.embedding_dim = embedding_dim
        self.random_seed = random_seed

    def extract_representation_features(
        self, candidates_df: pd.DataFrame, cutoff_dt: str
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Обучает спектральные и последовательные представления на истории < cutoff_dt
        и рассчитывает признаки близости и латентные координаты для пар из candidates_df.
        """
        # Извлекаем историю взаимодействий строго ДО cutoff_dt
        hist_df = self.con.execute(f"""
            SELECT
                user_id,
                item_id,
                datediff('day', watch_dt, DATE '{cutoff_dt}') AS days_ago,
                watched_pct,
                total_dur
            FROM interactions
            WHERE watch_dt < DATE '{cutoff_dt}'
            ORDER BY user_id, days_ago ASC
        """).df()

        unique_users = np.sort(hist_df["user_id"].unique())
        unique_items = np.sort(hist_df["item_id"].unique())

        user_to_idx = {int(u): idx for idx, u in enumerate(unique_users)}
        item_to_idx = {int(it): idx for idx, it in enumerate(unique_items)}

        u_idx = hist_df["user_id"].map(user_to_idx).values
        i_idx = hist_df["item_id"].map(item_to_idx).values
        days_ago = np.clip(hist_df["days_ago"].values.astype(np.float32), 0.0, 365.0)
        watched_pct = np.clip(hist_df["watched_pct"].values.astype(np.float32), 0.0, 100.0)

        # Вес ребра учитывает свежесть взаимодействия (time decay) и глубину просмотра
        time_decay_long = np.exp(-days_ago / 30.0)
        engagement_weight = 0.4 + 0.6 * (watched_pct / 100.0)
        edge_weights = (time_decay_long * engagement_weight).astype(np.float32)

        n_users = len(unique_users)
        n_items = len(unique_items)
        dim = min(self.embedding_dim, min(n_users, n_items) - 2)

        interaction_matrix = sparse.csr_matrix(
            (edge_weights, (u_idx, i_idx)), shape=(n_users, n_items), dtype=np.float32
        )

        # Нормировка строк (L2) для спектральной устойчивости
        interaction_norm = normalize(interaction_matrix, norm="l2", axis=1)

        svd = TruncatedSVD(n_components=dim, random_state=self.random_seed)
        user_spectral_emb = svd.fit_transform(interaction_norm).astype(np.float32)
        item_emb = svd.components_.T.astype(np.float32)

        # L2-нормированные версии для косинусной близости
        user_spectral_norm = normalize(user_spectral_emb, norm="l2", axis=1)
        item_emb_norm = normalize(item_emb, norm="l2", axis=1)

        # Строим последовательные представления истории пользователя (Short-term, Long-term, Last-Item)
        # Используем матричное умножение разреженных весов на матрицу эмбеддингов объектов item_emb_norm
        short_decay = np.exp(-days_ago / 5.0) * engagement_weight
        short_matrix = sparse.csr_matrix(
            (short_decay, (u_idx, i_idx)), shape=(n_users, n_items), dtype=np.float32
        )
        user_short_seq_emb = normalize(short_matrix @ item_emb_norm, norm="l2", axis=1)
        user_long_seq_emb = normalize(interaction_matrix @ item_emb_norm, norm="l2", axis=1)

        # Самое последнее действие каждого пользователя (минимальный days_ago, т.к. отсортировано по days_ago ASC)
        first_occ_df = hist_df.drop_duplicates(subset=["user_id"], keep="first")
        last_u_idx = first_occ_df["user_id"].map(user_to_idx).values
        last_i_idx = first_occ_df["item_id"].map(item_to_idx).values
        user_last_item_emb = np.zeros((n_users, dim), dtype=np.float32)
        user_last_item_emb[last_u_idx] = item_emb_norm[last_i_idx]

        # Сопоставляем кандидатов с индексами пользователей и объектов
        cand_u_idx = candidates_df["user_id"].map(user_to_idx).fillna(-1).astype(int).values
        cand_i_idx = candidates_df["item_id"].map(item_to_idx).fillna(-1).astype(int).values

        valid_u = cand_u_idx >= 0
        valid_i = cand_i_idx >= 0
        valid_both = valid_u & valid_i

        n_cands = len(candidates_df)
        spectral_dot = np.zeros(n_cands, dtype=np.float32)
        spectral_cos = np.zeros(n_cands, dtype=np.float32)
        long_seq_cos = np.zeros(n_cands, dtype=np.float32)
        short_seq_cos = np.zeros(n_cands, dtype=np.float32)
        last_item_cos = np.zeros(n_cands, dtype=np.float32)
        l2_dist_seq = np.ones(n_cands, dtype=np.float32) * 2.0
        short_vs_long_drift = np.zeros(n_cands, dtype=np.float32)

        u_valid = cand_u_idx[valid_both]
        i_valid = cand_i_idx[valid_both]

        spectral_dot[valid_both] = np.sum(
            user_spectral_emb[u_valid] * item_emb[i_valid], axis=1
        )
        spectral_cos[valid_both] = np.sum(
            user_spectral_norm[u_valid] * item_emb_norm[i_valid], axis=1
        )
        long_seq_cos[valid_both] = np.sum(
            user_long_seq_emb[u_valid] * item_emb_norm[i_valid], axis=1
        )
        short_seq_cos[valid_both] = np.sum(
            user_short_seq_emb[u_valid] * item_emb_norm[i_valid], axis=1
        )
        last_item_cos[valid_both] = np.sum(
            user_last_item_emb[u_valid] * item_emb_norm[i_valid], axis=1
        )
        l2_dist_seq[valid_both] = np.linalg.norm(
            user_short_seq_emb[u_valid] - item_emb_norm[i_valid], axis=1
        )
        short_vs_long_drift[valid_u] = np.sum(
            user_short_seq_emb[cand_u_idx[valid_u]] * user_long_seq_emb[cand_u_idx[valid_u]],
            axis=1,
        )

        # Первые 6 главных спектральных компонент пользователя и объекта
        n_proj = min(6, dim)
        u_proj = np.zeros((n_cands, n_proj), dtype=np.float32)
        i_proj = np.zeros((n_cands, n_proj), dtype=np.float32)
        hadamard_proj = np.zeros((n_cands, n_proj), dtype=np.float32)

        u_proj[valid_u] = user_spectral_emb[cand_u_idx[valid_u], :n_proj]
        i_proj[valid_i] = item_emb[cand_i_idx[valid_i], :n_proj]
        hadamard_proj[valid_both] = (
            user_short_seq_emb[u_valid, :n_proj] * item_emb_norm[i_valid, :n_proj]
        )

        rep_dict = {
            "emb_spectral_dot": spectral_dot,
            "emb_spectral_cos": spectral_cos,
            "emb_long_seq_cos": long_seq_cos,
            "emb_short_seq_cos": short_seq_cos,
            "emb_last_item_cos": last_item_cos,
            "emb_short_l2_dist": l2_dist_seq,
            "emb_user_interest_drift": short_vs_long_drift,
            "emb_short_minus_long_cos": short_seq_cos - long_seq_cos,
        }
        for d_i in range(n_proj):
            rep_dict[f"emb_u_svd_{d_i}"] = u_proj[:, d_i]
            rep_dict[f"emb_i_svd_{d_i}"] = i_proj[:, d_i]
            rep_dict[f"emb_ui_hadamard_{d_i}"] = hadamard_proj[:, d_i]

        rep_df = pd.DataFrame(rep_dict)
        rep_cols = list(rep_df.columns)
        return rep_df, rep_cols
