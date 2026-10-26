"""
Движок автоматической генерации и отбора признаков (AutoFE Engine на базе SQL / DuckDB).

Исследовательская идея (Направление №1):
Вместо ручного подбора окон и разрезов задаётся декларативная грамматика SQL-признаков:
  G = (Entity_Relation) × (Temporal_Window) × (Aggregation / Ratio / Window_Rank)
Движок автоматически:
1. Генерирует пул из ~85+ SQL-признаков по нестандартным временным окнам (1, 2, 3, 5, 10, 14, 21, 45, 60, 90 дней),
   включая:
   - Скорость/моментум популярности (отношения коротких и длинных окон: 1d/7d, 3d/14d, 5d/21d, 10d/45d);
   - Позицию и долю объекта внутри своего жанра и типа контента (SQL Window Functions);
   - Многомерные косвенные связи User × {Primary_Genre, Secondary_Genre, Country, Release_Decade, Age_Rating};
   - Обратные демографические связи Item × {User_Age, User_Sex, User_Kids_Flg, User_Income};
   - Персональные отклонения (Z-score / diff относительно среднего года выпуска, длительности и процента досмотра).
2. Проводит двухэтапный автоматический отбор признаков СТРОГО на обучающей и валидационной выборках:
   - Этап 1: Фильтрация нулевой дисперсии и высокой мультиколлинеарности (|r| > 0.985 на Train).
   - Этап 2: Ранжирование по Gain Importance + поиск оптимальной мощности подмножества K* по метрике NDCG@10 на Val.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.metrics import compute_ranking_metrics_per_user


@dataclass
class AutoFeatureMeta:
    """Метаданные автоматически сгенерированного SQL-признака."""
    name: str
    family: str
    window: str
    aggregation: str
    description: str


class SQLAutoFEEngine:
    """
    Автоматический генератор и селектор признаков на базе SQL-шаблонов в DuckDB.
    """

    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con
        self.catalog: Dict[str, AutoFeatureMeta] = {}
        self._register_grammar_catalog()

    def _register_grammar_catalog(self) -> None:
        """Регистрирует каталог пространства поиска SQL-признаков."""
        windows_u = [1, 3, 5, 10, 14, 21, 45, 60, 90]
        for w in windows_u:
            self.catalog[f"auto_u_views_{w}d"] = AutoFeatureMeta(
                f"auto_u_views_{w}d", "User_Temporal", f"{w}d", "COUNT", f"Число просмотров юзера за {w} дн."
            )
            self.catalog[f"auto_u_pct_mean_{w}d"] = AutoFeatureMeta(
                f"auto_u_pct_mean_{w}d", "User_Temporal", f"{w}d", "AVG", f"Средний % досмотра юзера за {w} дн."
            )
            self.catalog[f"auto_i_views_{w}d"] = AutoFeatureMeta(
                f"auto_i_views_{w}d", "Item_Temporal", f"{w}d", "COUNT", f"Популярность айтема за {w} дн."
            )
            self.catalog[f"auto_i_pct_mean_{w}d"] = AutoFeatureMeta(
                f"auto_i_pct_mean_{w}d", "Item_Temporal", f"{w}d", "AVG", f"Средний % досмотра айтема за {w} дн."
            )

    def generate_sql_feature_space(
        self, candidates_df: pd.DataFrame, cutoff_dt: str
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Генерирует полное пространство автоматических SQL-признаков строго по событиям watch_dt < cutoff_dt.
        """
        cand_with_id = candidates_df.copy()
        cand_with_id["_row_id"] = np.arange(len(cand_with_id), dtype=np.int64)
        self.con.register("autofe_cand_view", cand_with_id)

        query = f"""
        WITH
        -- Базовый лог событий строго ДО cutoff_dt (Zero Leakage)
        hist AS (
            SELECT
                i.user_id,
                i.item_id,
                i.watch_dt,
                i.total_dur,
                i.watched_pct,
                datediff('day', i.watch_dt, DATE '{cutoff_dt}') AS days_ago,
                m.primary_genre,
                m.secondary_genre,
                m.content_type,
                m.primary_country,
                m.release_year,
                m.release_decade,
                m.age_rating,
                m.for_kids,
                u.age AS u_age,
                u.sex AS u_sex,
                u.income AS u_income,
                u.kids_flg AS u_kids_flg
            FROM interactions i
            INNER JOIN items m ON i.item_id = m.item_id
            LEFT JOIN users u ON i.user_id = u.user_id
            WHERE i.watch_dt < DATE '{cutoff_dt}'
        ),
        -- 1. Многооконная сетка агрегатов пользователя: W in (1, 2, 3, 5, 10, 14, 21, 45, 60, 90, all)
        u_grid AS (
            SELECT
                user_id,
                COUNT(*) FILTER (WHERE days_ago <= 1) AS auto_u_views_1d,
                COUNT(*) FILTER (WHERE days_ago <= 2) AS auto_u_views_2d,
                COUNT(*) FILTER (WHERE days_ago <= 3) AS auto_u_views_3d,
                COUNT(*) FILTER (WHERE days_ago <= 5) AS auto_u_views_5d,
                COUNT(*) FILTER (WHERE days_ago <= 10) AS auto_u_views_10d,
                COUNT(*) FILTER (WHERE days_ago <= 14) AS auto_u_views_14d,
                COUNT(*) FILTER (WHERE days_ago <= 21) AS auto_u_views_21d,
                COUNT(*) FILTER (WHERE days_ago <= 45) AS auto_u_views_45d,
                COUNT(*) FILTER (WHERE days_ago <= 60) AS auto_u_views_60d,
                COUNT(*) FILTER (WHERE days_ago <= 90) AS auto_u_views_90d,
                COUNT(*) AS auto_u_views_all,
                COALESCE(AVG(watched_pct) FILTER (WHERE days_ago <= 3), 0.0) AS auto_u_pct_mean_3d,
                COALESCE(AVG(watched_pct) FILTER (WHERE days_ago <= 10), 0.0) AS auto_u_pct_mean_10d,
                COALESCE(AVG(watched_pct) FILTER (WHERE days_ago <= 21), 0.0) AS auto_u_pct_mean_21d,
                COALESCE(AVG(watched_pct) FILTER (WHERE days_ago <= 60), 0.0) AS auto_u_pct_mean_60d,
                COALESCE(AVG(watched_pct), 0.0) AS auto_u_pct_mean_all,
                COALESCE(STDDEV_POP(watched_pct), 0.0) AS auto_u_pct_std_all,
                COALESCE(AVG(total_dur) FILTER (WHERE days_ago <= 14), 0.0) AS auto_u_dur_mean_14d,
                COALESCE(AVG(total_dur), 0.0) AS auto_u_dur_mean_all,
                COALESCE(AVG(release_year), 2015.0) AS auto_u_pref_release_year,
                COALESCE(AVG(age_rating), 16.0) AS auto_u_pref_age_rating,
                COALESCE(AVG(CAST(for_kids AS DOUBLE)), 0.0) AS auto_u_pref_for_kids,
                COALESCE(MIN(days_ago), 999) AS auto_u_recency_min,
                COALESCE(MAX(days_ago), 0) AS auto_u_tenure_days,
                COUNT(DISTINCT watch_dt) FILTER (WHERE days_ago <= 14) AS auto_u_active_days_14d,
                COUNT(DISTINCT primary_genre) AS auto_u_genre_diversity_all
            FROM hist
            GROUP BY user_id
        ),
        -- 2. Многооконная сетка агрегатов объекта: W in (1, 2, 3, 5, 10, 14, 21, 45, 60, 90, all)
        i_grid AS (
            SELECT
                item_id,
                COUNT(*) FILTER (WHERE days_ago <= 1) AS auto_i_views_1d,
                COUNT(*) FILTER (WHERE days_ago <= 2) AS auto_i_views_2d,
                COUNT(*) FILTER (WHERE days_ago <= 3) AS auto_i_views_3d,
                COUNT(*) FILTER (WHERE days_ago <= 5) AS auto_i_views_5d,
                COUNT(*) FILTER (WHERE days_ago <= 10) AS auto_i_views_10d,
                COUNT(*) FILTER (WHERE days_ago <= 14) AS auto_i_views_14d,
                COUNT(*) FILTER (WHERE days_ago <= 21) AS auto_i_views_21d,
                COUNT(*) FILTER (WHERE days_ago <= 45) AS auto_i_views_45d,
                COUNT(*) FILTER (WHERE days_ago <= 60) AS auto_i_views_60d,
                COUNT(*) FILTER (WHERE days_ago <= 90) AS auto_i_views_90d,
                COUNT(*) AS auto_i_views_all,
                COALESCE(AVG(watched_pct) FILTER (WHERE days_ago <= 5), 0.0) AS auto_i_pct_mean_5d,
                COALESCE(AVG(watched_pct) FILTER (WHERE days_ago <= 14), 0.0) AS auto_i_pct_mean_14d,
                COALESCE(AVG(watched_pct) FILTER (WHERE days_ago <= 45), 0.0) AS auto_i_pct_mean_45d,
                COALESCE(AVG(watched_pct), 0.0) AS auto_i_pct_mean_all,
                COALESCE(STDDEV_POP(watched_pct), 0.0) AS auto_i_pct_std_all,
                COALESCE(AVG(total_dur) FILTER (WHERE days_ago <= 14), 0.0) AS auto_i_dur_mean_14d,
                COALESCE(AVG(total_dur), 0.0) AS auto_i_dur_mean_all,
                COALESCE(
                    AVG(CASE WHEN watched_pct >= 80.0 THEN 1.0 ELSE 0.0 END) FILTER (WHERE days_ago <= 14),
                    0.0
                ) AS auto_i_full_watch_rate_14d,
                COALESCE(
                    AVG(CASE WHEN watched_pct <= 10.0 THEN 1.0 ELSE 0.0 END) FILTER (WHERE days_ago <= 14),
                    0.0
                ) AS auto_i_bounce_rate_14d,
                COALESCE(MIN(days_ago), 999) AS auto_i_days_since_last_watch,
                COALESCE(MAX(days_ago), 0) AS auto_i_catalog_age_days,
                COUNT(DISTINCT user_id) FILTER (WHERE days_ago <= 10) AS auto_i_uniq_users_10d,
                COUNT(DISTINCT user_id) FILTER (WHERE days_ago <= 21) AS auto_i_uniq_users_21d,
                COUNT(DISTINCT user_id) AS auto_i_uniq_users_all
            FROM hist
            GROUP BY item_id
        ),
        -- 3. Оконные ранги объекта внутри своего жанра и типа контента (SQL Window Functions)
        i_window_ranks AS (
            SELECT
                ig.item_id,
                CAST(ig.auto_i_views_3d AS DOUBLE) / (SUM(ig.auto_i_views_3d) OVER (PARTITION BY m.primary_genre) + 1.0) AS auto_i_genre_pop_share_3d,
                CAST(ig.auto_i_views_14d AS DOUBLE) / (SUM(ig.auto_i_views_14d) OVER (PARTITION BY m.primary_genre) + 1.0) AS auto_i_genre_pop_share_14d,
                CAST(ig.auto_i_views_14d AS DOUBLE) / (SUM(ig.auto_i_views_14d) OVER (PARTITION BY m.content_type) + 1.0) AS auto_i_ctype_pop_share_14d,
                PERCENT_RANK() OVER (PARTITION BY m.primary_genre ORDER BY ig.auto_i_views_14d) AS auto_i_genre_pct_rank_14d,
                PERCENT_RANK() OVER (PARTITION BY m.content_type ORDER BY ig.auto_i_views_5d) AS auto_i_ctype_pct_rank_5d
            FROM i_grid ig
            INNER JOIN items m ON ig.item_id = m.item_id
        ),
        -- 4. Прямые связи user-item (Direct Links)
        ui_direct AS (
            SELECT
                user_id,
                item_id,
                COUNT(*) AS auto_ui_views_all,
                MAX(watched_pct) AS auto_ui_max_pct,
                MIN(days_ago) AS auto_ui_min_days_ago
            FROM hist
            GROUP BY user_id, item_id
        ),
        -- 5. Косвенные связи User × Primary_Genre по нестандартным окнам (3d, 10d, 21d, 60d, all)
        u_pgenre AS (
            SELECT
                user_id,
                primary_genre,
                COUNT(*) FILTER (WHERE days_ago <= 3) AS auto_ug_views_3d,
                COUNT(*) FILTER (WHERE days_ago <= 10) AS auto_ug_views_10d,
                COUNT(*) FILTER (WHERE days_ago <= 21) AS auto_ug_views_21d,
                COUNT(*) FILTER (WHERE days_ago <= 60) AS auto_ug_views_60d,
                COUNT(*) AS auto_ug_views_all,
                COALESCE(AVG(watched_pct) FILTER (WHERE days_ago <= 21), 0.0) AS auto_ug_pct_mean_21d,
                COALESCE(AVG(watched_pct), 0.0) AS auto_ug_pct_mean_all,
                COALESCE(MIN(days_ago), 999) AS auto_ug_days_since_last
            FROM hist
            GROUP BY user_id, primary_genre
        ),
        -- 6. Косвенные связи User × Secondary_Genre
        u_sgenre AS (
            SELECT
                user_id,
                secondary_genre,
                COUNT(*) FILTER (WHERE days_ago <= 21) AS auto_usg_views_21d,
                COUNT(*) AS auto_usg_views_all,
                COALESCE(AVG(watched_pct), 0.0) AS auto_usg_pct_mean_all
            FROM hist
            WHERE secondary_genre != 'none'
            GROUP BY user_id, secondary_genre
        ),
        -- 7. Косвенные связи User × Content_Type (5d, 14d, 45d, all)
        u_ctype AS (
            SELECT
                user_id,
                content_type,
                COUNT(*) FILTER (WHERE days_ago <= 5) AS auto_uct_views_5d,
                COUNT(*) FILTER (WHERE days_ago <= 14) AS auto_uct_views_14d,
                COUNT(*) FILTER (WHERE days_ago <= 45) AS auto_uct_views_45d,
                COUNT(*) AS auto_uct_views_all,
                COALESCE(AVG(watched_pct), 0.0) AS auto_uct_pct_mean_all,
                COALESCE(MIN(days_ago), 999) AS auto_uct_days_since_last
            FROM hist
            GROUP BY user_id, content_type
        ),
        -- 8. Косвенные связи User × Primary_Country (14d, all)
        u_country AS (
            SELECT
                user_id,
                primary_country,
                COUNT(*) FILTER (WHERE days_ago <= 14) AS auto_uc_views_14d,
                COUNT(*) AS auto_uc_views_all,
                COALESCE(AVG(watched_pct), 0.0) AS auto_uc_pct_mean_all,
                COALESCE(MIN(days_ago), 999) AS auto_uc_days_since_last
            FROM hist
            GROUP BY user_id, primary_country
        ),
        -- 9. Косвенные связи User × Release_Decade & Age_Rating
        u_decade AS (
            SELECT
                user_id,
                release_decade,
                COUNT(*) AS auto_udec_views_all,
                COALESCE(AVG(watched_pct), 0.0) AS auto_udec_pct_mean_all
            FROM hist
            GROUP BY user_id, release_decade
        ),
        u_agerat AS (
            SELECT
                user_id,
                age_rating,
                COUNT(*) AS auto_uar_views_all
            FROM hist
            GROUP BY user_id, age_rating
        ),
        -- 10. Обратные косвенные связи: Популярность объекта внутри демографической когорты пользователя!
        -- Item × User_Age (14d, all)
        i_demo_age AS (
            SELECT
                item_id,
                u_age,
                COUNT(*) FILTER (WHERE days_ago <= 14) AS auto_idemo_age_views_14d,
                COUNT(*) AS auto_idemo_age_views_all,
                COALESCE(AVG(watched_pct), 0.0) AS auto_idemo_age_pct_all
            FROM hist
            GROUP BY item_id, u_age
        ),
        -- Item × User_Sex (14d, all)
        i_demo_sex AS (
            SELECT
                item_id,
                u_sex,
                COUNT(*) FILTER (WHERE days_ago <= 14) AS auto_idemo_sex_views_14d,
                COUNT(*) AS auto_idemo_sex_views_all,
                COALESCE(AVG(watched_pct), 0.0) AS auto_idemo_sex_pct_all
            FROM hist
            GROUP BY item_id, u_sex
        ),
        -- Item × User_Kids_Flg
        i_demo_kids AS (
            SELECT
                item_id,
                u_kids_flg,
                COUNT(*) FILTER (WHERE days_ago <= 14) AS auto_idemo_kids_views_14d,
                COUNT(*) AS auto_idemo_kids_views_all
            FROM hist
            GROUP BY item_id, u_kids_flg
        )
        SELECT
            c.user_id,
            c.item_id,
            c.label,
            c.cutoff_dt,

            -- Базовые сетки User
            COALESCE(ug.auto_u_views_1d, 0) AS auto_u_views_1d,
            COALESCE(ug.auto_u_views_2d, 0) AS auto_u_views_2d,
            COALESCE(ug.auto_u_views_3d, 0) AS auto_u_views_3d,
            COALESCE(ug.auto_u_views_5d, 0) AS auto_u_views_5d,
            COALESCE(ug.auto_u_views_10d, 0) AS auto_u_views_10d,
            COALESCE(ug.auto_u_views_14d, 0) AS auto_u_views_14d,
            COALESCE(ug.auto_u_views_21d, 0) AS auto_u_views_21d,
            COALESCE(ug.auto_u_views_45d, 0) AS auto_u_views_45d,
            COALESCE(ug.auto_u_views_60d, 0) AS auto_u_views_60d,
            COALESCE(ug.auto_u_views_90d, 0) AS auto_u_views_90d,
            COALESCE(ug.auto_u_views_all, 0) AS auto_u_views_all,
            COALESCE(ug.auto_u_pct_mean_3d, 0.0) AS auto_u_pct_mean_3d,
            COALESCE(ug.auto_u_pct_mean_10d, 0.0) AS auto_u_pct_mean_10d,
            COALESCE(ug.auto_u_pct_mean_21d, 0.0) AS auto_u_pct_mean_21d,
            COALESCE(ug.auto_u_pct_mean_60d, 0.0) AS auto_u_pct_mean_60d,
            COALESCE(ug.auto_u_pct_mean_all, 0.0) AS auto_u_pct_mean_all,
            COALESCE(ug.auto_u_pct_std_all, 0.0) AS auto_u_pct_std_all,
            COALESCE(ug.auto_u_dur_mean_14d, 0.0) AS auto_u_dur_mean_14d,
            COALESCE(ug.auto_u_dur_mean_all, 0.0) AS auto_u_dur_mean_all,
            COALESCE(ug.auto_u_recency_min, 999) AS auto_u_recency_min,
            COALESCE(ug.auto_u_tenure_days, 0) AS auto_u_tenure_days,
            COALESCE(ug.auto_u_active_days_14d, 0) AS auto_u_active_days_14d,
            COALESCE(ug.auto_u_genre_diversity_all, 0) AS auto_u_genre_diversity_all,

            -- Авто-отношения (тренды активности пользователя)
            CAST(COALESCE(ug.auto_u_views_3d, 0) AS DOUBLE) / (COALESCE(ug.auto_u_views_14d, 0) + 1.0) AS auto_u_accel_3d_14d,
            CAST(COALESCE(ug.auto_u_views_10d, 0) AS DOUBLE) / (COALESCE(ug.auto_u_views_45d, 0) + 1.0) AS auto_u_accel_10d_45d,

            -- Базовые сетки Item
            COALESCE(ig.auto_i_views_1d, 0) AS auto_i_views_1d,
            COALESCE(ig.auto_i_views_2d, 0) AS auto_i_views_2d,
            COALESCE(ig.auto_i_views_3d, 0) AS auto_i_views_3d,
            COALESCE(ig.auto_i_views_5d, 0) AS auto_i_views_5d,
            COALESCE(ig.auto_i_views_10d, 0) AS auto_i_views_10d,
            COALESCE(ig.auto_i_views_14d, 0) AS auto_i_views_14d,
            COALESCE(ig.auto_i_views_21d, 0) AS auto_i_views_21d,
            COALESCE(ig.auto_i_views_45d, 0) AS auto_i_views_45d,
            COALESCE(ig.auto_i_views_60d, 0) AS auto_i_views_60d,
            COALESCE(ig.auto_i_views_90d, 0) AS auto_i_views_90d,
            COALESCE(ig.auto_i_views_all, 0) AS auto_i_views_all,
            COALESCE(ig.auto_i_pct_mean_5d, 0.0) AS auto_i_pct_mean_5d,
            COALESCE(ig.auto_i_pct_mean_14d, 0.0) AS auto_i_pct_mean_14d,
            COALESCE(ig.auto_i_pct_mean_45d, 0.0) AS auto_i_pct_mean_45d,
            COALESCE(ig.auto_i_pct_mean_all, 0.0) AS auto_i_pct_mean_all,
            COALESCE(ig.auto_i_pct_std_all, 0.0) AS auto_i_pct_std_all,
            COALESCE(ig.auto_i_dur_mean_14d, 0.0) AS auto_i_dur_mean_14d,
            COALESCE(ig.auto_i_dur_mean_all, 0.0) AS auto_i_dur_mean_all,
            COALESCE(ig.auto_i_full_watch_rate_14d, 0.0) AS auto_i_full_watch_rate_14d,
            COALESCE(ig.auto_i_bounce_rate_14d, 0.0) AS auto_i_bounce_rate_14d,
            COALESCE(ig.auto_i_days_since_last_watch, 999) AS auto_i_days_since_last_watch,
            COALESCE(ig.auto_i_catalog_age_days, 0) AS auto_i_catalog_age_days,
            COALESCE(ig.auto_i_uniq_users_10d, 0) AS auto_i_uniq_users_10d,
            COALESCE(ig.auto_i_uniq_users_21d, 0) AS auto_i_uniq_users_21d,
            COALESCE(ig.auto_i_uniq_users_all, 0) AS auto_i_uniq_users_all,

            -- Авто-отношения (скорость/моментум популярности объекта по разным парам окон)
            CAST(COALESCE(ig.auto_i_views_1d, 0) AS DOUBLE) / (COALESCE(ig.auto_i_views_5d, 0) + 1.0) AS auto_i_momentum_1d_5d,
            CAST(COALESCE(ig.auto_i_views_2d, 0) AS DOUBLE) / (COALESCE(ig.auto_i_views_10d, 0) + 1.0) AS auto_i_momentum_2d_10d,
            CAST(COALESCE(ig.auto_i_views_3d, 0) AS DOUBLE) / (COALESCE(ig.auto_i_views_14d, 0) + 1.0) AS auto_i_momentum_3d_14d,
            CAST(COALESCE(ig.auto_i_views_5d, 0) AS DOUBLE) / (COALESCE(ig.auto_i_views_21d, 0) + 1.0) AS auto_i_momentum_5d_21d,
            CAST(COALESCE(ig.auto_i_views_10d, 0) AS DOUBLE) / (COALESCE(ig.auto_i_views_45d, 0) + 1.0) AS auto_i_momentum_10d_45d,
            CAST(COALESCE(ig.auto_i_views_14d, 0) AS DOUBLE) / (COALESCE(ig.auto_i_views_60d, 0) + 1.0) AS auto_i_momentum_14d_60d,
            CAST(COALESCE(ig.auto_i_views_21d, 0) AS DOUBLE) / (COALESCE(ig.auto_i_views_90d, 0) + 1.0) AS auto_i_momentum_21d_90d,
            CAST(COALESCE(ig.auto_i_views_5d, 0) AS DOUBLE) / (COALESCE(ig.auto_i_catalog_age_days, 1) + 1.0) AS auto_i_velocity_per_catalog_day,

            -- Оконные ранги объекта внутри жанра и типа контента
            COALESCE(iwr.auto_i_genre_pop_share_3d, 0.0) AS auto_i_genre_pop_share_3d,
            COALESCE(iwr.auto_i_genre_pop_share_14d, 0.0) AS auto_i_genre_pop_share_14d,
            COALESCE(iwr.auto_i_ctype_pop_share_14d, 0.0) AS auto_i_ctype_pop_share_14d,
            COALESCE(iwr.auto_i_genre_pct_rank_14d, 0.0) AS auto_i_genre_pct_rank_14d,
            COALESCE(iwr.auto_i_ctype_pct_rank_5d, 0.0) AS auto_i_ctype_pct_rank_5d,

            -- Прямые связи user-item
            COALESCE(uid.auto_ui_views_all, 0) AS auto_ui_views_all,
            COALESCE(uid.auto_ui_max_pct, 0.0) AS auto_ui_max_pct,
            COALESCE(uid.auto_ui_min_days_ago, 999) AS auto_ui_min_days_ago,

            -- Косвенные связи User × Primary_Genre
            COALESCE(upg.auto_ug_views_3d, 0) AS auto_ug_views_3d,
            COALESCE(upg.auto_ug_views_10d, 0) AS auto_ug_views_10d,
            COALESCE(upg.auto_ug_views_21d, 0) AS auto_ug_views_21d,
            COALESCE(upg.auto_ug_views_60d, 0) AS auto_ug_views_60d,
            COALESCE(upg.auto_ug_views_all, 0) AS auto_ug_views_all,
            CAST(COALESCE(upg.auto_ug_views_3d, 0) AS DOUBLE) / (COALESCE(ug.auto_u_views_3d, 0) + 1.0) AS auto_ug_share_3d,
            CAST(COALESCE(upg.auto_ug_views_10d, 0) AS DOUBLE) / (COALESCE(ug.auto_u_views_10d, 0) + 1.0) AS auto_ug_share_10d,
            CAST(COALESCE(upg.auto_ug_views_21d, 0) AS DOUBLE) / (COALESCE(ug.auto_u_views_21d, 0) + 1.0) AS auto_ug_share_21d,
            CAST(COALESCE(upg.auto_ug_views_60d, 0) AS DOUBLE) / (COALESCE(ug.auto_u_views_60d, 0) + 1.0) AS auto_ug_share_60d,
            CAST(COALESCE(upg.auto_ug_views_all, 0) AS DOUBLE) / (COALESCE(ug.auto_u_views_all, 0) + 1.0) AS auto_ug_share_all,
            COALESCE(upg.auto_ug_pct_mean_21d, 0.0) AS auto_ug_pct_mean_21d,
            COALESCE(upg.auto_ug_pct_mean_all, 0.0) AS auto_ug_pct_mean_all,
            COALESCE(upg.auto_ug_pct_mean_all, 0.0) - COALESCE(ug.auto_u_pct_mean_all, 0.0) AS auto_ug_pct_excess_vs_user,
            COALESCE(upg.auto_ug_days_since_last, 999) AS auto_ug_days_since_last,

            -- Косвенные связи User × Secondary_Genre
            COALESCE(usg.auto_usg_views_21d, 0) AS auto_usg_views_21d,
            COALESCE(usg.auto_usg_views_all, 0) AS auto_usg_views_all,
            CAST(COALESCE(usg.auto_usg_views_all, 0) AS DOUBLE) / (COALESCE(ug.auto_u_views_all, 0) + 1.0) AS auto_usg_share_all,
            COALESCE(usg.auto_usg_pct_mean_all, 0.0) AS auto_usg_pct_mean_all,

            -- Косвенные связи User × Content_Type
            COALESCE(uct.auto_uct_views_5d, 0) AS auto_uct_views_5d,
            COALESCE(uct.auto_uct_views_14d, 0) AS auto_uct_views_14d,
            COALESCE(uct.auto_uct_views_45d, 0) AS auto_uct_views_45d,
            COALESCE(uct.auto_uct_views_all, 0) AS auto_uct_views_all,
            CAST(COALESCE(uct.auto_uct_views_14d, 0) AS DOUBLE) / (COALESCE(ug.auto_u_views_14d, 0) + 1.0) AS auto_uct_share_14d,
            CAST(COALESCE(uct.auto_uct_views_all, 0) AS DOUBLE) / (COALESCE(ug.auto_u_views_all, 0) + 1.0) AS auto_uct_share_all,
            COALESCE(uct.auto_uct_pct_mean_all, 0.0) AS auto_uct_pct_mean_all,
            COALESCE(uct.auto_uct_days_since_last, 999) AS auto_uct_days_since_last,

            -- Косвенные связи User × Country
            COALESCE(uc.auto_uc_views_14d, 0) AS auto_uc_views_14d,
            COALESCE(uc.auto_uc_views_all, 0) AS auto_uc_views_all,
            CAST(COALESCE(uc.auto_uc_views_all, 0) AS DOUBLE) / (COALESCE(ug.auto_u_views_all, 0) + 1.0) AS auto_uc_share_all,
            COALESCE(uc.auto_uc_pct_mean_all, 0.0) AS auto_uc_pct_mean_all,
            COALESCE(uc.auto_uc_days_since_last, 999) AS auto_uc_days_since_last,

            -- Косвенные связи User × Release_Decade & Age_Rating & Соответствие профилю
            COALESCE(udec.auto_udec_views_all, 0) AS auto_udec_views_all,
            CAST(COALESCE(udec.auto_udec_views_all, 0) AS DOUBLE) / (COALESCE(ug.auto_u_views_all, 0) + 1.0) AS auto_udec_share_all,
            COALESCE(udec.auto_udec_pct_mean_all, 0.0) AS auto_udec_pct_mean_all,
            COALESCE(uar.auto_uar_views_all, 0) AS auto_uar_views_all,
            CAST(COALESCE(uar.auto_uar_views_all, 0) AS DOUBLE) / (COALESCE(ug.auto_u_views_all, 0) + 1.0) AS auto_uar_share_all,
            ABS(COALESCE(m.release_year, 2015.0) - COALESCE(ug.auto_u_pref_release_year, 2015.0)) AS auto_ui_release_year_diff,
            ABS(COALESCE(m.age_rating, 16.0) - COALESCE(ug.auto_u_pref_age_rating, 16.0)) AS auto_ui_age_rating_diff,
            ABS(CAST(COALESCE(m.for_kids, 0) AS DOUBLE) - COALESCE(ug.auto_u_pref_for_kids, 0.0)) AS auto_ui_kids_pref_diff,

            -- Обратные демографические связи: Item × User Cohort (Age, Sex, Kids)
            COALESCE(ida.auto_idemo_age_views_14d, 0) AS auto_idemo_age_views_14d,
            COALESCE(ida.auto_idemo_age_views_all, 0) AS auto_idemo_age_views_all,
            CAST(COALESCE(ida.auto_idemo_age_views_14d, 0) AS DOUBLE) / (COALESCE(ig.auto_i_views_14d, 0) + 1.0) AS auto_idemo_age_share_14d,
            CAST(COALESCE(ida.auto_idemo_age_views_all, 0) AS DOUBLE) / (COALESCE(ig.auto_i_views_all, 0) + 1.0) AS auto_idemo_age_share_all,
            COALESCE(ida.auto_idemo_age_pct_all, 0.0) AS auto_idemo_age_pct_all,

            COALESCE(ids.auto_idemo_sex_views_14d, 0) AS auto_idemo_sex_views_14d,
            COALESCE(ids.auto_idemo_sex_views_all, 0) AS auto_idemo_sex_views_all,
            CAST(COALESCE(ids.auto_idemo_sex_views_14d, 0) AS DOUBLE) / (COALESCE(ig.auto_i_views_14d, 0) + 1.0) AS auto_idemo_sex_share_14d,
            CAST(COALESCE(ids.auto_idemo_sex_views_all, 0) AS DOUBLE) / (COALESCE(ig.auto_i_views_all, 0) + 1.0) AS auto_idemo_sex_share_all,
            COALESCE(ids.auto_idemo_sex_pct_all, 0.0) AS auto_idemo_sex_pct_all,

            COALESCE(idk.auto_idemo_kids_views_14d, 0) AS auto_idemo_kids_views_14d,
            CAST(COALESCE(idk.auto_idemo_kids_views_14d, 0) AS DOUBLE) / (COALESCE(ig.auto_i_views_14d, 0) + 1.0) AS auto_idemo_kids_share_14d,
            CAST(COALESCE(idk.auto_idemo_kids_views_all, 0) AS DOUBLE) / (COALESCE(ig.auto_i_views_all, 0) + 1.0) AS auto_idemo_kids_share_all
        FROM autofe_cand_view c
        LEFT JOIN users u ON c.user_id = u.user_id
        LEFT JOIN items m ON c.item_id = m.item_id
        LEFT JOIN u_grid ug ON c.user_id = ug.user_id
        LEFT JOIN i_grid ig ON c.item_id = ig.item_id
        LEFT JOIN i_window_ranks iwr ON c.item_id = iwr.item_id
        LEFT JOIN ui_direct uid ON c.user_id = uid.user_id AND c.item_id = uid.item_id
        LEFT JOIN u_pgenre upg ON c.user_id = upg.user_id AND m.primary_genre = upg.primary_genre
        LEFT JOIN u_sgenre usg ON c.user_id = usg.user_id AND m.secondary_genre = usg.secondary_genre
        LEFT JOIN u_ctype uct ON c.user_id = uct.user_id AND m.content_type = uct.content_type
        LEFT JOIN u_country uc ON c.user_id = uc.user_id AND m.primary_country = uc.primary_country
        LEFT JOIN u_decade udec ON c.user_id = udec.user_id AND m.release_decade = udec.release_decade
        LEFT JOIN u_agerat uar ON c.user_id = uar.user_id AND m.age_rating = uar.age_rating
        LEFT JOIN i_demo_age ida ON c.item_id = ida.item_id AND u.age = ida.u_age
        LEFT JOIN i_demo_sex ids ON c.item_id = ids.item_id AND u.sex = ids.u_sex
        LEFT JOIN i_demo_kids idk ON c.item_id = idk.item_id AND u.kids_flg = idk.u_kids_flg
        ORDER BY c._row_id ASC
        """

        df_auto = self.con.execute(query).df()
        meta_cols = {"user_id", "item_id", "label", "cutoff_dt"}
        auto_cols = [c for c in df_auto.columns if c not in meta_cols]
        return df_auto, auto_cols

    def select_best_features_on_val(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        candidate_cols: List[str],
        categorical_cols: List[str],
        random_seed: int = 42,
    ) -> Tuple[List[str], pd.DataFrame, Dict[int, Dict[str, float]]]:
        """
        Двухэтапный алгоритм автоматического отбора признаков (строго по Train и Val):
        1. Удаление константных и сильно коррелирующих числовых признаков (|r| > 0.985 на подвыборке Train).
        2. Оценка вклада признаков через градиентный бустинг (Gain + Split Importance на Train/Val)
           и перебор мощности подмножества K in {20, 30, 40, 50, 65, All} для максимизации Val NDCG@10.
        """
        numeric_cols = [c for c in candidate_cols if c not in categorical_cols]

        # Этап 1: Фильтр нулевой дисперсии и мультиколлинеарности на Train
        sample_df = train_df[numeric_cols].iloc[: min(len(train_df), 25000)]
        stds = sample_df.std()
        non_const_num = stds[stds > 1e-8].index.tolist()

        corr_matrix = sample_df[non_const_num].corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop_corr = [column for column in upper.columns if any(upper[column] > 0.985)]

        filtered_cols = [c for c in candidate_cols if c in categorical_cols or (c in non_const_num and c not in to_drop_corr)]
        print(
            f"[AutoFE Selection] Исходных признаков: {len(candidate_cols)}, "
            f"после фильтрации коллинеарности: {len(filtered_cols)} (удалено {len(candidate_cols) - len(filtered_cols)})"
        )

        # Этап 2: Обучение разведочной модели на Train и ранжирование признаков
        cat_in_filtered = [c for c in categorical_cols if c in filtered_cols]
        probe_model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample_for_bin=50000,
            colsample_bytree=0.8,
            subsample=0.8,
            random_state=random_seed,
            n_jobs=-1,
            verbose=-1,
        )
        probe_model.fit(
            train_df[filtered_cols],
            train_df["label"],
            eval_set=[(val_df[filtered_cols], val_df["label"])],
            categorical_feature=cat_in_filtered,
            callbacks=[lgb.early_stopping(30, verbose=False)],
        )

        gains = probe_model.booster_.feature_importance(importance_type="gain")
        splits = probe_model.booster_.feature_importance(importance_type="split")
        imp_df = pd.DataFrame(
            {
                "feature": filtered_cols,
                "gain": gains,
                "split": splits,
                "gain_norm": gains / (np.sum(gains) + 1e-12),
                "split_norm": splits / (np.sum(splits) + 1e-12),
            }
        )
        imp_df["combined_score"] = 0.7 * imp_df["gain_norm"] + 0.3 * imp_df["split_norm"]
        imp_df = imp_df.sort_values("combined_score", ascending=False).reset_index(drop=True)

        # Перебираем размер подмножества K на валидации
        candidate_ks = sorted(list({20, 30, 40, 55, 70, len(filtered_cols)}))
        candidate_ks = [k for k in candidate_ks if k <= len(filtered_cols)]

        k_eval_history: Dict[int, Dict[str, float]] = {}
        best_k = len(filtered_cols)
        best_val_score = -1.0
        best_subset = filtered_cols

        for k in candidate_ks:
            top_k_feats = imp_df["feature"].iloc[:k].tolist()
            cats_k = [c for c in cat_in_filtered if c in top_k_feats]

            m_k = lgb.LGBMClassifier(
                n_estimators=350,
                learning_rate=0.05,
                num_leaves=31,
                colsample_bytree=0.8,
                subsample=0.8,
                random_state=random_seed,
                n_jobs=-1,
                verbose=-1,
            )
            m_k.fit(
                train_df[top_k_feats],
                train_df["label"],
                eval_set=[(val_df[top_k_feats], val_df["label"])],
                categorical_feature=cats_k,
                callbacks=[lgb.early_stopping(30, verbose=False)],
            )

            val_preds = m_k.predict_proba(val_df[top_k_feats])[:, 1]
            val_eval = val_df[["user_id", "item_id", "label"]].copy()
            val_eval["score"] = val_preds
            metrics_k = compute_ranking_metrics_per_user(val_eval, score_col="score", ks=[10])
            k_eval_history[k] = metrics_k

            # Комбинированный критерий на Val: NDCG@10 + 0.2 * ROC-AUC
            composite = metrics_k["NDCG@10"] + 0.2 * metrics_k["ROC-AUC"]
            print(
                f"  [AutoFE Grid K={k:2d}] Val NDCG@10 = {metrics_k['NDCG@10']:.4f} | "
                f"Val Recall@10 = {metrics_k['Recall@10']:.4f} | Val ROC-AUC = {metrics_k['ROC-AUC']:.4f}"
            )
            if composite > best_val_score:
                best_val_score = composite
                best_k = k
                best_subset = top_k_feats

        print(f"[AutoFE Selection] Оптимальное подмножество на Val: K* = {best_k} признаков.")
        return best_subset, imp_df, k_eval_history
