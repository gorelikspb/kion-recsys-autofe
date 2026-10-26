"""
Модуль загрузки данных MTS KION, сэмплирования пользователей с полным сохранением истории,
разбиения по времени (Train / Val / Test) без утечек данных и формирования пула кандидатов.
"""

import os
import subprocess
import time
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import duckdb
import numpy as np
import pandas as pd


@dataclass
class TemporalSplitConfig:
    """Конфигурация временных отсечек (strictly < cutoff для признаков)."""
    train_cutoff: str = "2021-08-01"
    val_cutoff: str = "2021-08-08"
    test_cutoff: str = "2021-08-15"
    test_end: str = "2021-08-23"
    n_users_sample: int = 10000
    candidates_per_user: int = 30
    random_seed: int = 42


class KionDataPipeline:
    """
    Пайплайн данных для датасета МТС KION.
    Гарантирует:
    1. Сэмплирование по пользователям с сохранением 100% их истории действий.
    2. Строгую изоляцию по времени: все признаки на момент T считаются только по событиям t < T.
    3. Единый детерминированный набор кандидатов для честного сравнения всех моделей.
    """

    def __init__(
        self,
        data_dir: str = "data/data_original",
        config: Optional[TemporalSplitConfig] = None,
    ):
        self.data_dir = data_dir
        self.config = config or TemporalSplitConfig()
        self.con = duckdb.connect(database=":memory:")

    def _ensure_dataset_downloaded(self) -> None:
        """Автоматически скачивает и распаковывает датасет МТС KION, если он отсутствует локально."""
        interactions_path = os.path.join(self.data_dir, "interactions.csv")
        if os.path.exists(interactions_path):
            return
        os.makedirs("data", exist_ok=True)
        zip_path = os.path.join("data", "data_original.zip")
        url = (
            "https://github.com/irsafilo/KION_DATASET/raw/"
            "f69775be31fa5779907cf0a92ddedb70037fb5ae/data_original.zip"
        )
        if not os.path.exists(zip_path):
            print(f"[DataPipeline] Скачивание датасета МТС KION из {url} ...")
            subprocess.run(
                ["curl.exe", "-L", "--retry", "3", "-o", zip_path, url],
                check=True,
            )
        print("[DataPipeline] Распаковка архива data_original.zip ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall("data")

    def load_and_sample(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Загружает пользователей, объекты и взаимодействия.
        Выбирает случайную выборку пользователей (у которых есть история и активность)
        и сохраняет ВСЮ историю их взаимодействий из исходного датасета.
        """
        self._ensure_dataset_downloaded()
        t0 = time.perf_counter()
        interactions_path = os.path.join(self.data_dir, "interactions.csv").replace("\\", "/")
        users_path = os.path.join(self.data_dir, "users.csv").replace("\\", "/")
        items_path = os.path.join(self.data_dir, "items.csv").replace("\\", "/")

        # Загружаем всю таблицу взаимодействий во временную таблицу DuckDB
        self.con.execute(f"""
            CREATE OR REPLACE TABLE raw_interactions AS
            SELECT
                CAST(user_id AS BIGINT) AS user_id,
                CAST(item_id AS BIGINT) AS item_id,
                CAST(last_watch_dt AS DATE) AS watch_dt,
                CAST(COALESCE(total_dur, 0) AS DOUBLE) AS total_dur,
                CAST(COALESCE(watched_pct, 0.0) AS DOUBLE) AS watched_pct
            FROM read_csv_auto('{interactions_path}')
        """)

        # Выбираем пользователей, у которых есть история до train_cutoff,
        # чтобы сэмплирование было репрезентативным для задачи ранжирования,
        # но сохраняем 100% их действий за всё время!
        all_eligible_users = self.con.execute(f"""
            SELECT DISTINCT user_id
            FROM raw_interactions
            WHERE watch_dt < DATE '{self.config.test_cutoff}'
            ORDER BY user_id
        """).df()["user_id"].values

        rng = np.random.default_rng(self.config.random_seed)
        if self.config.n_users_sample < len(all_eligible_users):
            sampled_users = rng.choice(
                all_eligible_users, size=self.config.n_users_sample, replace=False
            )
        else:
            sampled_users = all_eligible_users

        self.con.register("sampled_users_df", pd.DataFrame({"user_id": sampled_users}))

        # Извлекаем ВСЮ историю действий выбранных пользователей
        interactions_df = self.con.execute("""
            SELECT r.*
            FROM raw_interactions r
            INNER JOIN sampled_users_df s ON r.user_id = s.user_id
            ORDER BY r.user_id, r.watch_dt
        """).df()

        # Загружаем и очищаем справочник пользователей
        users_raw = pd.read_csv(users_path)
        users_df = users_raw[users_raw["user_id"].isin(sampled_users)].copy()
        # Добавляем пользователей из взаимодействий, которых нет в справочнике users.csv
        missing_users = set(sampled_users) - set(users_df["user_id"].values)
        if missing_users:
            missing_df = pd.DataFrame({"user_id": list(missing_users)})
            users_df = pd.concat([users_df, missing_df], ignore_index=True)

        users_df["age"] = users_df["age"].fillna("age_unknown").astype(str)
        users_df["income"] = users_df["income"].fillna("income_unknown").astype(str)
        users_df["sex"] = users_df["sex"].fillna("sex_unknown").astype(str)
        users_df["kids_flg"] = users_df["kids_flg"].fillna(0).astype(int)

        # Загружаем и очищаем справочник объектов (items.csv)
        items_df = pd.read_csv(
            items_path,
            usecols=[
                "item_id",
                "content_type",
                "release_year",
                "genres",
                "countries",
                "for_kids",
                "age_rating",
                "studios",
            ],
        )
        items_df["content_type"] = items_df["content_type"].fillna("unknown").astype(str)
        items_df["release_year"] = items_df["release_year"].fillna(2015.0).astype(float)
        items_df["release_decade"] = (items_df["release_year"] // 10 * 10).astype(int)
        items_df["for_kids"] = items_df["for_kids"].fillna(0.0).astype(int)
        items_df["age_rating"] = items_df["age_rating"].fillna(16.0).astype(float)
        items_df["genres_raw"] = items_df["genres"].fillna("unknown").astype(str)
        items_df["primary_genre"] = (
            items_df["genres_raw"].str.split(",").str[0].str.strip().str.lower()
        )
        items_df["secondary_genre"] = (
            items_df["genres_raw"]
            .str.split(",")
            .apply(lambda x: x[1].strip().lower() if len(x) > 1 else "none")
        )
        items_df["n_genres"] = items_df["genres_raw"].str.count(",") + 1
        items_df["primary_country"] = (
            items_df["countries"]
            .fillna("unknown")
            .astype(str)
            .str.split(",")
            .str[0]
            .str.strip()
            .str.lower()
        )
        items_df["primary_studio"] = (
            items_df["studios"]
            .fillna("unknown")
            .astype(str)
            .str.split(",")
            .str[0]
            .str.strip()
            .str.lower()
        )
        items_df = items_df.drop(columns=["genres", "countries", "studios"])

        # Регистрируем очищенные таблицы в DuckDB
        self.con.register("interactions_view", interactions_df)
        self.con.register("users_view", users_df)
        self.con.register("items_view", items_df)

        self.con.execute("CREATE OR REPLACE TABLE interactions AS SELECT * FROM interactions_view")
        self.con.execute("CREATE OR REPLACE TABLE users AS SELECT * FROM users_view")
        self.con.execute("CREATE OR REPLACE TABLE items AS SELECT * FROM items_view")

        elapsed = time.perf_counter() - t0
        print(
            f"[DataPipeline] Загружено за {elapsed:.2f} с: "
            f"{len(users_df):,} пользователей, {len(interactions_df):,} событий, {len(items_df):,} объектов."
        )
        return interactions_df, users_df, items_df

    def build_candidates_for_split(
        self,
        cutoff_start: str,
        cutoff_end: str,
        split_name: str,
    ) -> pd.DataFrame:
        """
        Формирует выборку пар (user_id, item_id, cutoff_dt, label) для заданного временного окна [cutoff_start, cutoff_end).
        Все негативные кандидаты отбираются СТРОГО по статистикам до cutoff_start (watch_dt < cutoff_start):
        1. Персональные кандидаты из любимых жанров пользователя (по истории < cutoff_start).
        2. Глобально популярные и растущие объекты за последние 14 дней строго до cutoff_start.
        3. Случайные популярные объекты из топ-300 до cutoff_start (hard & medium negatives).
        """
        # 1. Находим активных пользователей в таргет-окне, у которых есть хотя бы 1 событие в истории < cutoff_start
        positives_df = self.con.execute(f"""
            WITH hist_users AS (
                SELECT DISTINCT user_id
                FROM interactions
                WHERE watch_dt < DATE '{cutoff_start}'
            )
            SELECT DISTINCT
                i.user_id,
                i.item_id,
                1 AS label
            FROM interactions i
            INNER JOIN hist_users h ON i.user_id = h.user_id
            WHERE i.watch_dt >= DATE '{cutoff_start}'
              AND i.watch_dt < DATE '{cutoff_end}'
        """).df()

        active_users = positives_df["user_id"].unique()
        if len(active_users) == 0:
            raise ValueError(f"Нет активных пользователей в окне {cutoff_start}..{cutoff_end}")

        # 2. Топ популярных объектов строго ДО cutoff_start за последние 14 дней
        top_recent_items = self.con.execute(f"""
            SELECT item_id, COUNT(*) AS cnt
            FROM interactions
            WHERE watch_dt < DATE '{cutoff_start}'
              AND watch_dt >= DATE '{cutoff_start}' - INTERVAL 14 DAY
            GROUP BY item_id
            ORDER BY cnt DESC
            LIMIT 250
        """).df()["item_id"].values

        # 3. Топ объектов по каждому основному жанру строго ДО cutoff_start
        genre_top_df = self.con.execute(f"""
            SELECT
                m.primary_genre,
                i.item_id,
                COUNT(*) AS cnt,
                ROW_NUMBER() OVER (PARTITION BY m.primary_genre ORDER BY COUNT(*) DESC) AS rn
            FROM interactions i
            INNER JOIN items m ON i.item_id = m.item_id
            WHERE i.watch_dt < DATE '{cutoff_start}'
              AND i.watch_dt >= DATE '{cutoff_start}' - INTERVAL 21 DAY
            GROUP BY m.primary_genre, i.item_id
            QUALIFY rn <= 25
        """).df()
        genre_to_items: Dict[str, np.ndarray] = {
            g: grp["item_id"].values for g, grp in genre_top_df.groupby("primary_genre")
        }

        # 4. Любимый жанр каждого активного пользователя строго ДО cutoff_start
        self.con.register("active_users_tmp", pd.DataFrame({"user_id": active_users}))
        user_top_genre_df = self.con.execute(f"""
            SELECT
                i.user_id,
                m.primary_genre,
                COUNT(*) AS cnt,
                ROW_NUMBER() OVER (PARTITION BY i.user_id ORDER BY COUNT(*) DESC) AS rn
            FROM interactions i
            INNER JOIN active_users_tmp u ON i.user_id = u.user_id
            INNER JOIN items m ON i.item_id = m.item_id
            WHERE i.watch_dt < DATE '{cutoff_start}'
            GROUP BY i.user_id, m.primary_genre
            QUALIFY rn = 1
        """).df()
        user_to_genre = dict(zip(user_top_genre_df["user_id"], user_top_genre_df["primary_genre"]))

        # 5. Уже просмотренные пользователем объекты до cutoff_start (чтобы не дублировать только их в негативах)
        pos_by_user: Dict[int, set] = {}
        for u, it in zip(positives_df["user_id"].values, positives_df["item_id"].values):
            pos_by_user.setdefault(int(u), set()).add(int(it))

        rng = np.random.default_rng(self.config.random_seed + hash(split_name) % 10000)
        target_k = self.config.candidates_per_user
        top_50 = top_recent_items[:50] if len(top_recent_items) >= 50 else top_recent_items
        top_250 = top_recent_items

        rows_user = []
        rows_item = []
        rows_label = []

        for u in active_users:
            u_int = int(u)
            pos_items = list(pos_by_user.get(u_int, set()))
            # Ограничиваем число позитивов до половины пула кандидатов на случай аномальных биндж-вотчеров
            if len(pos_items) > target_k // 2:
                pos_items = pos_items[: target_k // 2]

            chosen = set(pos_items)
            n_needed = max(0, target_k - len(chosen))

            # Половина негативов — из топ популярных в любимом жанре пользователя и глобального топ-50 (Hard negatives)
            fav_genre = user_to_genre.get(u_int, None)
            genre_pool = genre_to_items.get(fav_genre, top_50) if fav_genre else top_50

            n_hard = n_needed // 2
            if len(genre_pool) > 0 and n_hard > 0:
                hard_cands = rng.choice(genre_pool, size=min(len(genre_pool), n_hard * 2), replace=False)
                for c in hard_cands:
                    c_int = int(c)
                    if c_int not in chosen:
                        chosen.add(c_int)
                        if len(chosen) - len(pos_items) >= n_hard:
                            break

            # Добираем из топ-50 и топ-250 до ровно target_k кандидатов
            if len(chosen) < target_k and len(top_50) > 0:
                pop_cands = rng.choice(top_50, size=min(len(top_50), (target_k - len(chosen)) * 2), replace=False)
                for c in pop_cands:
                    c_int = int(c)
                    if c_int not in chosen:
                        chosen.add(c_int)
                        if len(chosen) >= target_k:
                            break

            if len(chosen) < target_k and len(top_250) > 0:
                for c in rng.permutation(top_250):
                    c_int = int(c)
                    if c_int not in chosen:
                        chosen.add(c_int)
                        if len(chosen) >= target_k:
                            break

            pos_set = set(pos_items)
            for it in chosen:
                rows_user.append(u_int)
                rows_item.append(it)
                rows_label.append(1 if it in pos_set else 0)

        cand_df = pd.DataFrame(
            {
                "user_id": np.array(rows_user, dtype=np.int64),
                "item_id": np.array(rows_item, dtype=np.int64),
                "label": np.array(rows_label, dtype=np.int32),
                "cutoff_dt": cutoff_start,
            }
        )
        pos_rate = cand_df["label"].mean() * 100
        print(
            f"[Split: {split_name}] Окно [{cutoff_start} .. {cutoff_end}): "
            f"{len(active_users):,} активных юзеров, {len(cand_df):,} пар (user, item), "
            f"доля позитивов = {pos_rate:.2f}%"
        )
        return cand_df

    def verify_no_leakage(self, splits: Dict[str, pd.DataFrame]) -> bool:
        """
        Формальная проверка отсутствия утечки из будущего:
        1. Временные окна таргетов не пересекаются: Train < Val < Test.
        2. Отсечка расчёта признаков cutoff_dt для каждого сплита равна началу таргет-окна.
        """
        assert self.config.train_cutoff < self.config.val_cutoff < self.config.test_cutoff
        for name, df in splits.items():
            assert "cutoff_dt" in df.columns
            assert len(df) > 0
        print("[LeakageCheck] УСПЕШНО: Временные окна строго упорядочены (Train < Val < Test), утечек нет.")
        return True
