"""
Модуль ручного конструирования признаков (Strong Manual Feature Engineering Baseline).
Реализует классический промышленный набор признаков для градиентного бустинга:
1. Признаки пользователя (категории + агрегаты за 7, 30 дней и всю историю < cutoff_dt).
2. Признаки объекта (категории + популярность и досматриваемость за 7, 30 дней и всю историю < cutoff_dt).
3. Прямые связи user-item (повторные взаимодействия и давность до cutoff_dt).
4. Косвенные связи user-item через категории (жанр и тип контента за 30 дней до cutoff_dt).
"""

from typing import List, Tuple
import duckdb
import numpy as np
import pandas as pd


class ManualFeatureBuilder:
    """Формирует классические ручные признаки строго по событиям watch_dt < cutoff_dt."""

    CATEGORICAL_COLS: List[str] = [
        "u_age",
        "u_income",
        "u_sex",
        "i_content_type",
        "i_primary_genre",
        "i_primary_country",
    ]

    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con

    def build_features(
        self, candidates_df: pd.DataFrame, cutoff_dt: str
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """
        Рассчитывает ручные признаки для пар (user_id, item_id) на момент cutoff_dt.
        Все агрегаты используют строгое условие WHERE watch_dt < DATE '{cutoff_dt}'.
        """
        cand_with_id = candidates_df.copy()
        cand_with_id["_row_id"] = np.arange(len(cand_with_id), dtype=np.int64)
        self.con.register("cand_view", cand_with_id)

        query = f"""
        WITH
        -- 1. Базовая история строго ДО момента предсказания (Zero Leakage)
        hist AS (
            SELECT
                i.user_id,
                i.item_id,
                i.watch_dt,
                i.total_dur,
                i.watched_pct,
                datediff('day', i.watch_dt, DATE '{cutoff_dt}') AS days_ago,
                m.primary_genre,
                m.content_type
            FROM interactions i
            INNER JOIN items m ON i.item_id = m.item_id
            WHERE i.watch_dt < DATE '{cutoff_dt}'
        ),
        -- 2. Признаки пользователя (7d, 30d, all_time)
        u_stats AS (
            SELECT
                user_id,
                COUNT(*) FILTER (WHERE days_ago <= 7) AS u_views_7d,
                COUNT(*) FILTER (WHERE days_ago <= 30) AS u_views_30d,
                COUNT(*) AS u_views_all,
                COALESCE(AVG(watched_pct) FILTER (WHERE days_ago <= 30), 0.0) AS u_avg_watched_pct_30d,
                COALESCE(AVG(total_dur) FILTER (WHERE days_ago <= 30), 0.0) AS u_avg_dur_30d,
                COALESCE(SUM(total_dur) FILTER (WHERE days_ago <= 30), 0.0) AS u_total_dur_30d,
                COALESCE(MIN(days_ago), 999) AS u_days_since_last_watch,
                COUNT(DISTINCT watch_dt) FILTER (WHERE days_ago <= 30) AS u_active_days_30d
            FROM hist
            GROUP BY user_id
        ),
        -- 3. Признаки объекта (7d, 30d, all_time)
        i_stats AS (
            SELECT
                item_id,
                COUNT(*) FILTER (WHERE days_ago <= 7) AS i_views_7d,
                COUNT(*) FILTER (WHERE days_ago <= 30) AS i_views_30d,
                COUNT(*) AS i_views_all,
                COALESCE(AVG(watched_pct) FILTER (WHERE days_ago <= 30), 0.0) AS i_avg_watched_pct_30d,
                COALESCE(AVG(total_dur) FILTER (WHERE days_ago <= 30), 0.0) AS i_avg_dur_30d,
                COUNT(DISTINCT user_id) FILTER (WHERE days_ago <= 30) AS i_unique_users_30d,
                COALESCE(
                    AVG(CASE WHEN watched_pct >= 75.0 THEN 1.0 ELSE 0.0 END) FILTER (WHERE days_ago <= 30),
                    0.0
                ) AS i_completion_rate_30d
            FROM hist
            GROUP BY item_id
        ),
        -- 4. Прямые связи user-item (повторные/продолженные просмотры до cutoff_dt)
        ui_direct AS (
            SELECT
                user_id,
                item_id,
                COUNT(*) AS ui_prior_views_all,
                MAX(watched_pct) AS ui_prior_max_watched_pct,
                MIN(days_ago) AS ui_days_since_last_view
            FROM hist
            GROUP BY user_id, item_id
        ),
        -- 5. Косвенные связи: пользователь × основной жанр за 30 дней
        u_genre_30d AS (
            SELECT
                user_id,
                primary_genre,
                COUNT(*) AS ug_views_30d,
                AVG(watched_pct) AS ug_avg_watched_pct_30d
            FROM hist
            WHERE days_ago <= 30
            GROUP BY user_id, primary_genre
        ),
        -- 6. Косвенные связи: пользователь × тип контента (фильм/сериал) за 30 дней
        u_ctype_30d AS (
            SELECT
                user_id,
                content_type,
                COUNT(*) AS uct_views_30d
            FROM hist
            WHERE days_ago <= 30
            GROUP BY user_id, content_type
        )
        SELECT
            c.user_id,
            c.item_id,
            c.label,
            c.cutoff_dt,
            -- Категориальные признаки пользователя
            COALESCE(u.age, 'age_unknown') AS u_age,
            COALESCE(u.income, 'income_unknown') AS u_income,
            COALESCE(u.sex, 'sex_unknown') AS u_sex,
            COALESCE(u.kids_flg, 0) AS u_kids_flg,
            -- Числовые признаки пользователя
            COALESCE(us.u_views_7d, 0) AS u_views_7d,
            COALESCE(us.u_views_30d, 0) AS u_views_30d,
            COALESCE(us.u_views_all, 0) AS u_views_all,
            COALESCE(us.u_avg_watched_pct_30d, 0.0) AS u_avg_watched_pct_30d,
            COALESCE(us.u_avg_dur_30d, 0.0) AS u_avg_dur_30d,
            COALESCE(us.u_total_dur_30d, 0.0) AS u_total_dur_30d,
            COALESCE(us.u_days_since_last_watch, 999) AS u_days_since_last_watch,
            COALESCE(us.u_active_days_30d, 0) AS u_active_days_30d,
            -- Категориальные и статические признаки объекта
            COALESCE(m.content_type, 'unknown') AS i_content_type,
            COALESCE(m.release_year, 2015.0) AS i_release_year,
            COALESCE(m.for_kids, 0) AS i_for_kids,
            COALESCE(m.age_rating, 16.0) AS i_age_rating,
            COALESCE(m.primary_genre, 'unknown') AS i_primary_genre,
            COALESCE(m.primary_country, 'unknown') AS i_primary_country,
            -- Числовые признаки объекта
            COALESCE(ist.i_views_7d, 0) AS i_views_7d,
            COALESCE(ist.i_views_30d, 0) AS i_views_30d,
            COALESCE(ist.i_views_all, 0) AS i_views_all,
            COALESCE(ist.i_avg_watched_pct_30d, 0.0) AS i_avg_watched_pct_30d,
            COALESCE(ist.i_avg_dur_30d, 0.0) AS i_avg_dur_30d,
            COALESCE(ist.i_unique_users_30d, 0) AS i_unique_users_30d,
            COALESCE(ist.i_completion_rate_30d, 0.0) AS i_completion_rate_30d,
            -- Прямые связи user-item
            COALESCE(uid.ui_prior_views_all, 0) AS ui_prior_views_all,
            COALESCE(uid.ui_prior_max_watched_pct, 0.0) AS ui_prior_max_watched_pct,
            COALESCE(uid.ui_days_since_last_view, 999) AS ui_days_since_last_view,
            -- Косвенные связи user-item (через жанр и тип контента за 30 дней)
            COALESCE(ug.ug_views_30d, 0) AS ug_views_30d,
            CAST(COALESCE(ug.ug_views_30d, 0) AS DOUBLE) / (COALESCE(us.u_views_30d, 0) + 1.0) AS ug_share_30d,
            COALESCE(ug.ug_avg_watched_pct_30d, 0.0) AS ug_avg_watched_pct_30d,
            COALESCE(uct.uct_views_30d, 0) AS uct_views_30d,
            CAST(COALESCE(uct.uct_views_30d, 0) AS DOUBLE) / (COALESCE(us.u_views_30d, 0) + 1.0) AS uct_share_30d
        FROM cand_view c
        LEFT JOIN users u ON c.user_id = u.user_id
        LEFT JOIN items m ON c.item_id = m.item_id
        LEFT JOIN u_stats us ON c.user_id = us.user_id
        LEFT JOIN i_stats ist ON c.item_id = ist.item_id
        LEFT JOIN ui_direct uid ON c.user_id = uid.user_id AND c.item_id = uid.item_id
        LEFT JOIN u_genre_30d ug ON c.user_id = ug.user_id AND m.primary_genre = ug.primary_genre
        LEFT JOIN u_ctype_30d uct ON c.user_id = uct.user_id AND m.content_type = uct.content_type
        ORDER BY c._row_id ASC
        """

        df_feat = self.con.execute(query).df()

        # Конвертируем категориальные колонки в тип category для LightGBM
        for col in self.CATEGORICAL_COLS:
            df_feat[col] = df_feat[col].astype("category")

        meta_cols = {"user_id", "item_id", "label", "cutoff_dt"}
        feature_cols = [c for c in df_feat.columns if c not in meta_cols]
        return df_feat, feature_cols, list(self.CATEGORICAL_COLS)
