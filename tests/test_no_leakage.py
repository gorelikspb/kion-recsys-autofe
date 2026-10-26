"""
Автоматические тесты корректности пайплайна, метрик ранжирования
и строгого отсутствия утечек данных из будущего (Zero Data Leakage).
"""

import unittest
import duckdb
import numpy as np
import pandas as pd

from src.autofe_engine import SQLAutoFEEngine
from src.manual_features import ManualFeatureBuilder
from src.metrics import compute_ranking_metrics_per_user


class TestZeroLeakageAndMetrics(unittest.TestCase):
    def setUp(self) -> None:
        self.con = duckdb.connect(database=":memory:")

        # Создаём синтетические данные с событиями ДО и ПОСЛЕ cutoff_dt = '2021-08-10'
        users_df = pd.DataFrame(
            {
                "user_id": [1, 2],
                "age": ["age_25_34", "age_35_44"],
                "income": ["income_20_40", "income_40_60"],
                "sex": ["М", "Ж"],
                "kids_flg": [0, 1],
            }
        )
        items_df = pd.DataFrame(
            {
                "item_id": [101, 102],
                "content_type": ["film", "series"],
                "release_year": [2020.0, 2021.0],
                "release_decade": [2020, 2020],
                "for_kids": [0, 1],
                "age_rating": [16.0, 6.0],
                "genres_raw": ["боевики, триллеры", "мультфильмы, фэнтези"],
                "primary_genre": ["боевики", "мультфильмы"],
                "secondary_genre": ["триллеры", "фэнтези"],
                "n_genres": [2, 2],
                "primary_country": ["сша", "россия"],
                "primary_studio": ["universal", "союзмультфильм"],
            }
        )
        # События до 2021-08-10 и в день/после 2021-08-10
        interactions_df = pd.DataFrame(
            {
                "user_id": [1, 1, 2],
                "item_id": [101, 102, 102],
                "watch_dt": pd.to_datetime(["2021-08-05", "2021-08-10", "2021-08-15"]).date,
                "total_dur": [3600.0, 99999.0, 88888.0],
                "watched_pct": [80.0, 100.0, 100.0],
            }
        )

        self.con.register("u_v", users_df)
        self.con.register("i_v", items_df)
        self.con.register("inter_v", interactions_df)
        self.con.execute("CREATE TABLE users AS SELECT * FROM u_v")
        self.con.execute("CREATE TABLE items AS SELECT * FROM i_v")
        self.con.execute("CREATE TABLE interactions AS SELECT * FROM inter_v")

    def test_manual_features_ignore_cutoff_and_future_events(self) -> None:
        """Проверяет, что события в момент cutoff_dt и позже НЕ попадают в ручные признаки."""
        cands = pd.DataFrame(
            {
                "user_id": [1, 1, 2],
                "item_id": [101, 102, 102],
                "label": [0, 1, 1],
                "cutoff_dt": ["2021-08-10"] * 3,
            }
        )
        builder = ManualFeatureBuilder(self.con)
        feat_df, _, _ = builder.build_features(cands, "2021-08-10")

        # У юзера 1 до 2021-08-10 было ровно 1 событие (2021-08-05, item 101), событие 2021-08-10 (item 102) не должно попасть!
        row_u1_102 = feat_df[(feat_df["user_id"] == 1) & (feat_df["item_id"] == 102)].iloc[0]
        self.assertEqual(row_u1_102["u_views_all"], 1)
        self.assertEqual(row_u1_102["i_views_all"], 0)
        self.assertEqual(row_u1_102["ui_prior_views_all"], 0)

        # У юзера 2 до 2021-08-10 вообще не было событий (его событие 2021-08-15 — в будущем)
        row_u2_102 = feat_df[(feat_df["user_id"] == 2) & (feat_df["item_id"] == 102)].iloc[0]
        self.assertEqual(row_u2_102["u_views_all"], 0)

    def test_autofe_features_ignore_cutoff_and_future_events(self) -> None:
        """Проверяет, что автоматический генератор SQL-признаков не допускает утечки из будущего."""
        cands = pd.DataFrame(
            {
                "user_id": [1, 2],
                "item_id": [102, 102],
                "label": [1, 1],
                "cutoff_dt": ["2021-08-10", "2021-08-10"],
            }
        )
        engine = SQLAutoFEEngine(self.con)
        auto_df, _ = engine.generate_sql_feature_space(cands, "2021-08-10")

        # Объект 102 смотрели только 2021-08-10 и 2021-08-15, поэтому на момент < 2021-08-10 у него 0 просмотров
        self.assertEqual(auto_df.iloc[0]["auto_i_views_all"], 0)
        self.assertEqual(auto_df.iloc[0]["auto_u_views_all"], 1)
        self.assertEqual(auto_df.iloc[1]["auto_u_views_all"], 0)

    def test_ranking_metrics_correctness(self) -> None:
        """Проверяет корректность расчёта NDCG@K и Recall@K при идеальном ранжировании."""
        df = pd.DataFrame(
            {
                "user_id": [1, 1, 1, 2, 2, 2],
                "item_id": [10, 20, 30, 10, 20, 30],
                "label": [1, 0, 0, 0, 1, 0],
                "score": [0.9, 0.2, 0.1, 0.1, 0.8, 0.3],
            }
        )
        metrics = compute_ranking_metrics_per_user(df, score_col="score", ks=[1, 3])
        self.assertAlmostEqual(metrics["NDCG@1"], 1.0, places=5)
        self.assertAlmostEqual(metrics["Recall@1"], 1.0, places=5)
        self.assertAlmostEqual(metrics["ROC-AUC"], 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
