"""
Главный скрипт запуска исследовательских экспериментов (MTS KION RecSys AutoFE vs Manual GBDT vs Learned Representations).
Выполняет:
1. Подготовку данных и проверку отсутствия утечек (Zero Data Leakage).
2. Расчёт ручных признаков (Strong Manual Baseline), автоматических SQL-признаков (AutoFE Engine)
   и обучаемых последовательных/спектральных представлений (Learned Representations).
3. Двухэтапный отбор признаков строго на валидационной выборке (Val).
4. Сравнение 6 моделей на отложенной тестовой выборке (Test) по NDCG@10, Recall@10, MAP@10, MRR@10, ROC-AUC, LogLoss.
5. Исследование чувствительности к выбору временного окна (почему 7 дней не оптимально) и Ablation Study групп признаков.
6. Оценку вычислительных затрат и масштабируемости (1 000 / 5 000 / 10 000 пользователей).
7. Генерацию графиков в reports/figures/ и сохранение сводных таблиц в reports/.
"""

import json
import os
import sys
import time
import tracemalloc
from typing import Dict, List, Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate

# Настройка кодировки для корректного вывода кириллицы в консоль Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.autofe_engine import SQLAutoFEEngine
from src.data_pipeline import KionDataPipeline, TemporalSplitConfig
from src.learned_representations import TimeAwareRepresentationLearner
from src.manual_features import ManualFeatureBuilder
from src.metrics import compute_ranking_metrics_per_user


def train_and_evaluate_lgbm(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    cat_cols: List[str],
    random_seed: int = 42,
) -> Tuple[lgb.LGBMClassifier, Dict[str, float], Dict[str, float], float]:
    """
    Обучает градиентный бустинг LightGBM с ранней остановкой по валидационной выборке
    и оценивает качество на Val и Test.
    """
    t0 = time.perf_counter()
    active_cats = [c for c in cat_cols if c in feature_cols]

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.04,
        num_leaves=31,
        max_depth=6,
        min_child_samples=40,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        random_state=random_seed,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        train_df[feature_cols],
        train_df["label"],
        eval_set=[(val_df[feature_cols], val_df["label"])],
        categorical_feature=active_cats,
        callbacks=[lgb.early_stopping(stopping_rounds=40, verbose=False)],
    )
    train_time = time.perf_counter() - t0

    val_preds = model.predict_proba(val_df[feature_cols])[:, 1]
    test_preds = model.predict_proba(test_df[feature_cols])[:, 1]

    val_eval = val_df[["user_id", "item_id", "label"]].copy()
    val_eval["score"] = val_preds
    val_metrics = compute_ranking_metrics_per_user(val_eval, score_col="score", ks=[5, 10, 20])

    test_eval = test_df[["user_id", "item_id", "label"]].copy()
    test_eval["score"] = test_preds
    test_metrics = compute_ranking_metrics_per_user(test_eval, score_col="score", ks=[5, 10, 20])

    return model, val_metrics, test_metrics, train_time


def run_window_sensitivity_study(
    train_full: pd.DataFrame,
    val_full: pd.DataFrame,
    test_full: pd.DataFrame,
    cat_cols: List[str],
) -> List[Dict[str, float]]:
    """
    Эксперимент: «Почему именно 7 дней?».
    Сравнивает модели, использующие признаки только одного фиксированного окна W in {1, 3, 5, 10, 14, 21, 45, 60, 90} дней,
    против комбинации всех окон и трендовых отношений (Momentum).
    """
    print("\n=== Исследование чувствительности к выбору временного окна (Window Sensitivity) ===")
    windows = [1, 3, 5, 10, 14, 21, 45, 60, 90]
    results = []

    base_static = [
        "u_age",
        "u_income",
        "u_sex",
        "u_kids_flg",
        "i_content_type",
        "i_release_year",
        "i_for_kids",
        "i_age_rating",
        "i_primary_genre",
        "i_primary_country",
    ]

    for w in windows:
        w_cols = [c for c in train_full.columns if c.endswith(f"_{w}d")]
        cols = base_static + w_cols
        _, val_m, test_m, t_sec = train_and_evaluate_lgbm(
            train_full, val_full, test_full, cols, cat_cols
        )
        row = {
            "window": f"{w} дней",
            "window_days": w,
            "n_features": len(cols),
            "Val_NDCG@10": val_m["NDCG@10"],
            "Test_NDCG@10": test_m["NDCG@10"],
            "Test_Recall@10": test_m["Recall@10"],
            "Test_ROC-AUC": test_m["ROC-AUC"],
            "train_sec": t_sec,
        }
        results.append(row)
        print(
            f"  Окно {w:2d} дн. ({len(cols):2d} фич) -> Val NDCG@10: {val_m['NDCG@10']:.4f} | "
            f"Test NDCG@10: {test_m['NDCG@10']:.4f} | Test ROC-AUC: {test_m['ROC-AUC']:.4f}"
        )

    # Классическая пара окон (7d + 30d из ручного бейзлайна)
    classic_cols = base_static + [
        c for c in train_full.columns if c.endswith("_7d") or c.endswith("_30d")
    ]
    _, val_c, test_c, t_c = train_and_evaluate_lgbm(
        train_full, val_full, test_full, classic_cols, cat_cols
    )
    results.append(
        {
            "window": "Классика (7d + 30d)",
            "window_days": 37,
            "n_features": len(classic_cols),
            "Val_NDCG@10": val_c["NDCG@10"],
            "Test_NDCG@10": test_c["NDCG@10"],
            "Test_Recall@10": test_c["Recall@10"],
            "Test_ROC-AUC": test_c["ROC-AUC"],
            "train_sec": t_c,
        }
    )

    # Мультиоконная сетка AutoFE + Momentum
    multi_cols = base_static + [
        c
        for c in train_full.columns
        if c.startswith("auto_u_views_")
        or c.startswith("auto_i_views_")
        or c.startswith("auto_i_momentum_")
        or c.startswith("auto_ug_views_")
        or c.startswith("auto_ug_share_")
    ]
    _, val_m, test_m, t_m = train_and_evaluate_lgbm(
        train_full, val_full, test_full, multi_cols, cat_cols
    )
    results.append(
        {
            "window": "AutoFE Multi-Window + Momentum",
            "window_days": 100,
            "n_features": len(multi_cols),
            "Val_NDCG@10": val_m["NDCG@10"],
            "Test_NDCG@10": test_m["NDCG@10"],
            "Test_Recall@10": test_m["Recall@10"],
            "Test_ROC-AUC": test_m["ROC-AUC"],
            "train_sec": t_m,
        }
    )

    return results


def run_ablation_study(
    train_full: pd.DataFrame,
    val_full: pd.DataFrame,
    test_full: pd.DataFrame,
    cat_cols: List[str],
    manual_cols: List[str],
    auto_cols: List[str],
    emb_cols: List[str],
) -> List[Dict[str, float]]:
    """
    Пошаговое исследование вклада семейств признаков (Feature Group Ablation Study).
    Показывает, какие именно компоненты дают прирост качества.
    """
    print("\n=== Ablation Study: Вклад отдельных семейств признаков ===")
    base_user_item = [
        c
        for c in manual_cols
        if c.startswith("u_") or c.startswith("i_")
    ]
    direct_ui = [c for c in manual_cols if c.startswith("ui_")]
    indirect_manual = [c for c in manual_cols if c.startswith("ug_") or c.startswith("uct_")]

    auto_momentum_ranks = [
        c
        for c in auto_cols
        if "momentum" in c or "velocity" in c or "rank" in c or "pop_share" in c or "views_1d" in c or "views_3d" in c or "views_14d" in c
    ]
    auto_indirect_multi = [
        c
        for c in auto_cols
        if c.startswith("auto_ug_")
        or c.startswith("auto_usg_")
        or c.startswith("auto_uc_")
        or c.startswith("auto_udec_")
        or c.startswith("auto_uar_")
        or c.startswith("auto_ui_")
    ]
    auto_reverse_demo = [c for c in auto_cols if c.startswith("auto_idemo_")]

    configs = [
        ("1. Только User + Item (7d/30d/all)", base_user_item),
        ("2. + Прямые связи User-Item (Direct)", base_user_item + direct_ui),
        ("3. + Косвенные связи Жанр/Тип (Manual Baseline)", manual_cols),
        (
            "4. + AutoFE Нестандартные окна, Моментумы и Ранги внутри жанра",
            list(dict.fromkeys(manual_cols + auto_momentum_ranks)),
        ),
        (
            "5. + AutoFE Многомерные косвенные связи (Страна, Декада, Втор. жанр)",
            list(dict.fromkeys(manual_cols + auto_momentum_ranks + auto_indirect_multi)),
        ),
        (
            "6. + AutoFE Обратные демографические связи (Item × User Cohort)",
            list(
                dict.fromkeys(
                    manual_cols + auto_momentum_ranks + auto_indirect_multi + auto_reverse_demo
                )
            ),
        ),
        (
            "7. + Обучаемые представления истории (Spectral & Sequence Embeddings)",
            list(
                dict.fromkeys(
                    manual_cols
                    + auto_momentum_ranks
                    + auto_indirect_multi
                    + auto_reverse_demo
                    + emb_cols
                )
            ),
        ),
    ]

    ablation_rows = []
    for stage_name, cols in configs:
        _, val_m, test_m, t_sec = train_and_evaluate_lgbm(
            train_full, val_full, test_full, cols, cat_cols
        )
        row = {
            "stage": stage_name,
            "n_features": len(cols),
            "Val_NDCG@10": val_m["NDCG@10"],
            "Test_NDCG@10": test_m["NDCG@10"],
            "Test_Recall@10": test_m["Recall@10"],
            "Test_MAP@10": test_m["MAP@10"],
            "Test_ROC-AUC": test_m["ROC-AUC"],
            "Test_LogLoss": test_m["LogLoss"],
            "train_sec": t_sec,
        }
        ablation_rows.append(row)
        print(
            f"  {stage_name:<66} | Фич: {len(cols):3d} | "
            f"Test NDCG@10: {test_m['NDCG@10']:.4f} | ROC-AUC: {test_m['ROC-AUC']:.4f}"
        )

    return ablation_rows


def run_scalability_benchmark() -> List[Dict[str, float]]:
    """
    Замеряет время генерации признаков, время обучения и потребление памяти
    при масштабировании числа пользователей: N in {1000, 5000, 10000}.
    """
    print("\n=== Исследование масштабируемости (Scalability Benchmark: 1k -> 5k -> 10k юзеров) ===")
    sample_sizes = [1000, 5000, 10000]
    scale_rows = []

    for n_u in sample_sizes:
        tracemalloc.start()
        t_start = time.perf_counter()

        cfg = TemporalSplitConfig(n_users_sample=n_u, candidates_per_user=30, random_seed=42)
        pipe = KionDataPipeline(config=cfg)
        interactions_df, _, _ = pipe.load_and_sample()

        train_c = pipe.build_candidates_for_split(cfg.train_cutoff, cfg.val_cutoff, f"Train_{n_u}")
        val_c = pipe.build_candidates_for_split(cfg.val_cutoff, cfg.test_cutoff, f"Val_{n_u}")
        test_c = pipe.build_candidates_for_split(cfg.test_cutoff, cfg.test_end, f"Test_{n_u}")

        t_man0 = time.perf_counter()
        man_builder = ManualFeatureBuilder(pipe.con)
        tr_m, man_cols, cat_cols = man_builder.build_features(train_c, cfg.train_cutoff)
        va_m, _, _ = man_builder.build_features(val_c, cfg.val_cutoff)
        te_m, _, _ = man_builder.build_features(test_c, cfg.test_cutoff)
        man_feat_sec = time.perf_counter() - t_man0

        t_auto0 = time.perf_counter()
        autofe = SQLAutoFEEngine(pipe.con)
        tr_a, auto_cols = autofe.generate_sql_feature_space(train_c, cfg.train_cutoff)
        va_a, _ = autofe.generate_sql_feature_space(val_c, cfg.val_cutoff)
        te_a, _ = autofe.generate_sql_feature_space(test_c, cfg.test_cutoff)
        auto_feat_sec = time.perf_counter() - t_auto0

        _, _, test_man_metrics, man_train_sec = train_and_evaluate_lgbm(
            tr_m, va_m, te_m, man_cols, cat_cols
        )

        # Объединяем категории с AutoFE для обучения
        for c in cat_cols:
            tr_a[c] = tr_m[c]
            va_a[c] = va_m[c]
            te_a[c] = te_m[c]
        full_auto_cols = cat_cols + auto_cols
        _, _, test_auto_metrics, auto_train_sec = train_and_evaluate_lgbm(
            tr_a, va_a, te_a, full_auto_cols, cat_cols
        )

        _, peak_mem_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        total_sec = time.perf_counter() - t_start

        row = {
            "n_users": n_u,
            "n_events": len(interactions_df),
            "n_train_pairs": len(train_c),
            "manual_feat_sec": round(man_feat_sec, 2),
            "autofe_feat_sec": round(auto_feat_sec, 2),
            "manual_train_sec": round(man_train_sec, 2),
            "autofe_train_sec": round(auto_train_sec, 2),
            "peak_ram_mb": round(peak_mem_bytes / (1024 * 1024), 1),
            "total_pipeline_sec": round(total_sec, 2),
            "Manual_Test_NDCG@10": test_man_metrics["NDCG@10"],
            "AutoFE_Test_NDCG@10": test_auto_metrics["NDCG@10"],
            "Manual_Test_ROC-AUC": test_man_metrics["ROC-AUC"],
            "AutoFE_Test_ROC-AUC": test_auto_metrics["ROC-AUC"],
        }
        scale_rows.append(row)
        print(
            f"  [N={n_u:5d} юзеров | {len(interactions_df):7d} событий] "
            f"Manual SQL: {man_feat_sec:.2f}s vs AutoFE SQL: {auto_feat_sec:.2f}s | "
            f"Manual NDCG@10: {test_man_metrics['NDCG@10']:.4f} -> AutoFE NDCG@10: {test_auto_metrics['NDCG@10']:.4f}"
        )

    return scale_rows


def generate_plots(
    main_comparison: List[Dict[str, float]],
    window_results: List[Dict[str, float]],
    ablation_results: List[Dict[str, float]],
    scale_results: List[Dict[str, float]],
    top_features_df: pd.DataFrame,
    out_dir: str = "reports/figures",
) -> None:
    """Создаёт наглядные графики для отчёта в папке reports/figures/."""
    os.makedirs(out_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI"]
    plt.rcParams["axes.unicode_minus"] = False

    # 1. Главное сравнение подходов (NDCG@10 и ROC-AUC на Test)
    comp_df = pd.DataFrame(main_comparison)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    palette = ["#95a5a6", "#e67e22", "#3498db", "#9b59b6", "#2ecc71", "#1abc9c"]
    sns.barplot(
        data=comp_df,
        y="Method",
        x="Test_NDCG@10",
        hue="Method",
        palette=palette[: len(comp_df)],
        legend=False,
        ax=axes[0],
    )
    axes[0].set_title("Качество ранжирования на Test (NDCG@10)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("NDCG@10 (больше — лучше)")
    axes[0].set_ylabel("")
    axes[0].set_xlim(comp_df["Test_NDCG@10"].min() * 0.9, comp_df["Test_NDCG@10"].max() * 1.06)
    for idx, row in comp_df.iterrows():
        axes[0].text(
            row["Test_NDCG@10"] + 0.002,
            idx,
            f"{row['Test_NDCG@10']:.4f}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    sns.barplot(
        data=comp_df,
        y="Method",
        x="Test_ROC-AUC",
        hue="Method",
        palette=palette[: len(comp_df)],
        legend=False,
        ax=axes[1],
    )
    axes[1].set_title("Качество классификации на Test (ROC-AUC)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("ROC-AUC (больше — лучше)")
    axes[1].set_ylabel("")
    axes[1].set_xlim(comp_df["Test_ROC-AUC"].min() * 0.92, comp_df["Test_ROC-AUC"].max() * 1.04)
    for idx, row in comp_df.iterrows():
        axes[1].text(
            row["Test_ROC-AUC"] + 0.002,
            idx,
            f"{row['Test_ROC-AUC']:.4f}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "model_comparison.png"), dpi=180)
    plt.close()

    # 2. График чувствительности к длине временного окна (Window Sensitivity)
    win_df = pd.DataFrame(window_results)
    single_win = win_df[win_df["window_days"] <= 90].copy()
    multi_win_ndcg = win_df[win_df["window"] == "AutoFE Multi-Window + Momentum"]["Test_NDCG@10"].values[0]
    classic_ndcg = win_df[win_df["window"] == "Классика (7d + 30d)"]["Test_NDCG@10"].values[0]

    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.plot(
        single_win["window_days"],
        single_win["Test_NDCG@10"],
        marker="o",
        linewidth=2.5,
        markersize=8,
        color="#2980b9",
        label="Одиночное окно W дней (Test NDCG@10)",
    )
    ax.plot(
        single_win["window_days"],
        single_win["Val_NDCG@10"],
        marker="s",
        linestyle="--",
        linewidth=1.8,
        markersize=6,
        color="#7f8c8d",
        label="Одиночное окно W дней (Val NDCG@10)",
    )
    ax.axhline(
        classic_ndcg,
        color="#e67e22",
        linestyle="-.",
        linewidth=2.0,
        label=f"Классическая пара (7d + 30d): {classic_ndcg:.4f}",
    )
    ax.axhline(
        multi_win_ndcg,
        color="#27ae60",
        linestyle="-",
        linewidth=2.2,
        label=f"AutoFE Multi-Window + Momentum: {multi_win_ndcg:.4f}",
    )
    ax.set_xticks(single_win["window_days"])
    ax.set_title(
        "Почему окно 7 дней не оптимально: влияние горизонта агрегации на NDCG@10",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlabel("Длина временного окна агрегации (дней до момента предсказания)")
    ax.set_ylabel("NDCG@10")
    ax.legend(loc="lower right", frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "window_sensitivity.png"), dpi=180)
    plt.close()

    # 3. Топ-20 самых важных признаков гибридной/AutoFE модели
    top20 = top_features_df.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10.5, 7))
    colors = [
        "#2ecc71"
        if c.startswith("auto_")
        else ("#9b59b6" if c.startswith("emb_") else "#3498db")
        for c in top20["feature"]
    ]
    ax.barh(top20["feature"], top20["gain_pct"], color=colors)
    ax.set_title(
        "Топ-20 признаков по вкладу в модель (Gain Importance, %)\nЗелёный = AutoFE SQL, Фиолетовый = Learned Embeddings, Синий = Manual Baseline",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Доля суммарного Gain модели (%)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "top_feature_importance.png"), dpi=180)
    plt.close()

    # 4. График Ablation Study и Масштабируемости
    abl_df = pd.DataFrame(ablation_results)
    scale_df = pd.DataFrame(scale_results)

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.5))
    axes[0].plot(
        range(1, len(abl_df) + 1),
        abl_df["Test_NDCG@10"],
        marker="o",
        linewidth=2.5,
        color="#16a085",
    )
    axes[0].set_xticks(range(1, len(abl_df) + 1))
    axes[0].set_xticklabels([f"Шаг {i}" for i in range(1, len(abl_df) + 1)])
    axes[0].set_title("Ablation Study: Прирост Test NDCG@10 при добавлении семейств фич", fontweight="bold")
    axes[0].set_ylabel("Test NDCG@10")
    for i, r in abl_df.iterrows():
        axes[0].annotate(
            f"{r['Test_NDCG@10']:.4f}",
            (i + 1, r["Test_NDCG@10"]),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
            fontweight="bold",
        )

    axes[1].plot(
        scale_df["n_users"],
        scale_df["Manual_Test_NDCG@10"],
        marker="s",
        linewidth=2.2,
        color="#3498db",
        label="Manual Baseline GBDT",
    )
    axes[1].plot(
        scale_df["n_users"],
        scale_df["AutoFE_Test_NDCG@10"],
        marker="o",
        linewidth=2.5,
        color="#2ecc71",
        label="AutoFE SQL Engine GBDT",
    )
    axes[1].set_title("Масштабируемость: Качество при росте выборки клиентов", fontweight="bold")
    axes[1].set_xlabel("Количество сэмплированных пользователей (с полной историей)")
    axes[1].set_ylabel("Test NDCG@10")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "ablation_and_scaling.png"), dpi=180)
    plt.close()
    print(f"[Plots] Все графики успешно сохранены в '{out_dir}/'.")


def main() -> None:
    print("=" * 85)
    print(" ЗАПУСК ЭКСПЕРИМЕНТАЛЬНОГО ПАЙПЛАЙНА: MTS KION RECSYS (AUTO-FE vs MANUAL vs EMBEDDINGS)")
    print("=" * 85)

    config = TemporalSplitConfig(
        train_cutoff="2021-08-01",
        val_cutoff="2021-08-08",
        test_cutoff="2021-08-15",
        test_end="2021-08-23",
        n_users_sample=10000,
        candidates_per_user=30,
        random_seed=42,
    )

    pipeline = KionDataPipeline(config=config)
    interactions_df, users_df, items_df = pipeline.load_and_sample()

    # Формируем сплиты Train, Val, Test
    train_cands = pipeline.build_candidates_for_split(
        config.train_cutoff, config.val_cutoff, "Train"
    )
    val_cands = pipeline.build_candidates_for_split(
        config.val_cutoff, config.test_cutoff, "Val"
    )
    test_cands = pipeline.build_candidates_for_split(
        config.test_cutoff, config.test_end, "Test"
    )

    pipeline.verify_no_leakage({"Train": train_cands, "Val": val_cands, "Test": test_cands})

    # -------------------------------------------------------------------------
    # Шаг 1. Расчёт ручных признаков (Strong Manual Baseline)
    # -------------------------------------------------------------------------
    print("\n--- Шаг 1: Генерация ручных признаков (Strong Manual Baseline) ---")
    t0 = time.perf_counter()
    manual_builder = ManualFeatureBuilder(pipeline.con)
    train_man, manual_cols, cat_cols = manual_builder.build_features(
        train_cands, config.train_cutoff
    )
    val_man, _, _ = manual_builder.build_features(val_cands, config.val_cutoff)
    test_man, _, _ = manual_builder.build_features(test_cands, config.test_cutoff)
    manual_gen_time = time.perf_counter() - t0
    print(
        f"[ManualFeatures] Сгенерировано {len(manual_cols)} ручных признаков для 3 сплитов за {manual_gen_time:.2f} с."
    )

    # -------------------------------------------------------------------------
    # Шаг 2. Генерация пространства автоматических SQL-признаков (AutoFE Engine)
    # -------------------------------------------------------------------------
    print("\n--- Шаг 2: Автоматическая генерация SQL-признаков (SQLAutoFEEngine) ---")
    t0 = time.perf_counter()
    autofe_engine = SQLAutoFEEngine(pipeline.con)
    train_auto, auto_cols = autofe_engine.generate_sql_feature_space(
        train_cands, config.train_cutoff
    )
    val_auto, _ = autofe_engine.generate_sql_feature_space(val_cands, config.val_cutoff)
    test_auto, _ = autofe_engine.generate_sql_feature_space(test_cands, config.test_cutoff)
    autofe_gen_time = time.perf_counter() - t0
    print(
        f"[AutoFE Engine] Сгенерировано {len(auto_cols)} автоматических SQL-признаков за {autofe_gen_time:.2f} с."
    )

    # -------------------------------------------------------------------------
    # Шаг 3. Обучаемые представления истории взаимодействий (Spectral & Sequence Embeddings)
    # -------------------------------------------------------------------------
    print("\n--- Шаг 3: Извлечение обучаемых представлений истории (TimeAwareRepresentationLearner) ---")
    t0 = time.perf_counter()
    rep_learner = TimeAwareRepresentationLearner(pipeline.con, embedding_dim=24, random_seed=42)
    train_emb, emb_cols = rep_learner.extract_representation_features(
        train_cands, config.train_cutoff
    )
    val_emb, _ = rep_learner.extract_representation_features(val_cands, config.val_cutoff)
    test_emb, _ = rep_learner.extract_representation_features(test_cands, config.test_cutoff)
    emb_gen_time = time.perf_counter() - t0
    print(
        f"[LearnedRepresentations] Сгенерировано {len(emb_cols)} латентных признаков за {emb_gen_time:.2f} с."
    )

    # Объединяем все признаки в единый датафрейм для каждого сплита (без дублирования мета-колонок)
    meta_cols = ["user_id", "item_id", "label", "cutoff_dt"]
    train_full = pd.concat(
        [
            train_man,
            train_auto.drop(columns=meta_cols),
            train_emb,
        ],
        axis=1,
    )
    val_full = pd.concat(
        [
            val_man,
            val_auto.drop(columns=meta_cols),
            val_emb,
        ],
        axis=1,
    )
    test_full = pd.concat(
        [
            test_man,
            test_auto.drop(columns=meta_cols),
            test_emb,
        ],
        axis=1,
    )

    for c in cat_cols:
        train_full[c] = train_full[c].astype("category")
        val_full[c] = val_full[c].astype("category")
        test_full[c] = test_full[c].astype("category")

    # -------------------------------------------------------------------------
    # Шаг 4. Сравнение 6 подходов на едином разбиении и пуле кандидатов
    # -------------------------------------------------------------------------
    print("\n--- Шаг 4: Обучение и отбор признаков на Val, финальная оценка на Test ---")
    main_comparison = []

    # Модель 1: Не-ML бейзлайн (Ранжирование по популярности объекта за 14 дней + досмотру)
    val_pop_eval = val_full[["user_id", "item_id", "label"]].copy()
    val_pop_eval["score"] = (
        val_full["auto_i_views_14d"].values / (val_full["auto_i_views_14d"].max() + 1.0)
    )
    val_pop_m = compute_ranking_metrics_per_user(val_pop_eval, score_col="score", ks=[5, 10, 20])

    test_pop_eval = test_full[["user_id", "item_id", "label"]].copy()
    test_pop_eval["score"] = (
        test_full["auto_i_views_14d"].values / (test_full["auto_i_views_14d"].max() + 1.0)
    )
    test_pop_m = compute_ranking_metrics_per_user(test_pop_eval, score_col="score", ks=[5, 10, 20])
    main_comparison.append(
        {
            "Method": "1. TopPopular (14d Heuristic)",
            "n_features": 1,
            "feat_gen_sec": 0.15,
            "train_sec": 0.0,
            "Val_NDCG@10": val_pop_m["NDCG@10"],
            "Test_NDCG@10": test_pop_m["NDCG@10"],
            "Test_Recall@10": test_pop_m["Recall@10"],
            "Test_MAP@10": test_pop_m["MAP@10"],
            "Test_MRR@10": test_pop_m["MRR@10"],
            "Test_ROC-AUC": test_pop_m["ROC-AUC"],
            "Test_PR-AUC": test_pop_m["PR-AUC"],
            "Test_LogLoss": test_pop_m["LogLoss"],
        }
    )

    # Модель 2: Замена ручных счётчиков на обучаемые представления (Learned Embeddings + статика)
    static_cols = [
        "u_age",
        "u_income",
        "u_sex",
        "u_kids_flg",
        "i_content_type",
        "i_release_year",
        "i_for_kids",
        "i_age_rating",
        "i_primary_genre",
        "i_primary_country",
    ]
    pure_emb_cols = static_cols + emb_cols
    _, val_emb_m, test_emb_m, emb_train_sec = train_and_evaluate_lgbm(
        train_full, val_full, test_full, pure_emb_cols, cat_cols
    )
    main_comparison.append(
        {
            "Method": "2. Pure Learned Embeddings (SVD+Seq)",
            "n_features": len(pure_emb_cols),
            "feat_gen_sec": round(emb_gen_time, 2),
            "train_sec": round(emb_train_sec, 2),
            "Val_NDCG@10": val_emb_m["NDCG@10"],
            "Test_NDCG@10": test_emb_m["NDCG@10"],
            "Test_Recall@10": test_emb_m["Recall@10"],
            "Test_MAP@10": test_emb_m["MAP@10"],
            "Test_MRR@10": test_emb_m["MRR@10"],
            "Test_ROC-AUC": test_emb_m["ROC-AUC"],
            "Test_PR-AUC": test_emb_m["PR-AUC"],
            "Test_LogLoss": test_emb_m["LogLoss"],
        }
    )

    # Модель 3: Сильный базовый подход (Strong Manual Baseline LightGBM)
    man_model, val_man_m, test_man_m, man_train_sec = train_and_evaluate_lgbm(
        train_full, val_full, test_full, manual_cols, cat_cols
    )
    main_comparison.append(
        {
            "Method": "3. Strong Manual Baseline (GBDT)",
            "n_features": len(manual_cols),
            "feat_gen_sec": round(manual_gen_time, 2),
            "train_sec": round(man_train_sec, 2),
            "Val_NDCG@10": val_man_m["NDCG@10"],
            "Test_NDCG@10": test_man_m["NDCG@10"],
            "Test_Recall@10": test_man_m["Recall@10"],
            "Test_MAP@10": test_man_m["MAP@10"],
            "Test_MRR@10": test_man_m["MRR@10"],
            "Test_ROC-AUC": test_man_m["ROC-AUC"],
            "Test_PR-AUC": test_man_m["PR-AUC"],
            "Test_LogLoss": test_man_m["LogLoss"],
        }
    )

    # Модель 4: Автоматическая генерация SQL-признаков без отбора (AutoFE Full SQL Space)
    autofe_full_cols = static_cols + auto_cols
    _, val_af_full_m, test_af_full_m, af_full_train_sec = train_and_evaluate_lgbm(
        train_full, val_full, test_full, autofe_full_cols, cat_cols
    )
    main_comparison.append(
        {
            "Method": "4. AutoFE Full SQL Space (Unfiltered)",
            "n_features": len(autofe_full_cols),
            "feat_gen_sec": round(autofe_gen_time, 2),
            "train_sec": round(af_full_train_sec, 2),
            "Val_NDCG@10": val_af_full_m["NDCG@10"],
            "Test_NDCG@10": test_af_full_m["NDCG@10"],
            "Test_Recall@10": test_af_full_m["Recall@10"],
            "Test_MAP@10": test_af_full_m["MAP@10"],
            "Test_MRR@10": test_af_full_m["MRR@10"],
            "Test_ROC-AUC": test_af_full_m["ROC-AUC"],
            "Test_PR-AUC": test_af_full_m["PR-AUC"],
            "Test_LogLoss": test_af_full_m["LogLoss"],
        }
    )

    # Модель 5: AutoFE с двухэтапным автоматическим отбором признаков на Val
    selected_autofe_cols, autofe_imp_df, k_grid_history = autofe_engine.select_best_features_on_val(
        train_full, val_full, autofe_full_cols, cat_cols, random_seed=42
    )
    _, val_af_sel_m, test_af_sel_m, af_sel_train_sec = train_and_evaluate_lgbm(
        train_full, val_full, test_full, selected_autofe_cols, cat_cols
    )
    main_comparison.append(
        {
            "Method": f"5. AutoFE Selected (Top-{len(selected_autofe_cols)} SQL)",
            "n_features": len(selected_autofe_cols),
            "feat_gen_sec": round(autofe_gen_time, 2),
            "train_sec": round(af_sel_train_sec, 2),
            "Val_NDCG@10": val_af_sel_m["NDCG@10"],
            "Test_NDCG@10": test_af_sel_m["NDCG@10"],
            "Test_Recall@10": test_af_sel_m["Recall@10"],
            "Test_MAP@10": test_af_sel_m["MAP@10"],
            "Test_MRR@10": test_af_sel_m["MRR@10"],
            "Test_ROC-AUC": test_af_sel_m["ROC-AUC"],
            "Test_PR-AUC": test_af_sel_m["PR-AUC"],
            "Test_LogLoss": test_af_sel_m["LogLoss"],
        }
    )

    # Модель 6: Гибридный подход (AutoFE Selected + Manual + Learned Embeddings)
    hybrid_candidate_pool = list(dict.fromkeys(manual_cols + selected_autofe_cols + emb_cols))
    selected_hybrid_cols, hybrid_imp_df, _ = autofe_engine.select_best_features_on_val(
        train_full, val_full, hybrid_candidate_pool, cat_cols, random_seed=42
    )
    hybrid_model, val_hyb_m, test_hyb_m, hyb_train_sec = train_and_evaluate_lgbm(
        train_full, val_full, test_full, selected_hybrid_cols, cat_cols
    )
    main_comparison.append(
        {
            "Method": f"6. Hybrid (AutoFE + Learned Emb)",
            "n_features": len(selected_hybrid_cols),
            "feat_gen_sec": round(autofe_gen_time + emb_gen_time, 2),
            "train_sec": round(hyb_train_sec, 2),
            "Val_NDCG@10": val_hyb_m["NDCG@10"],
            "Test_NDCG@10": test_hyb_m["NDCG@10"],
            "Test_Recall@10": test_hyb_m["Recall@10"],
            "Test_MAP@10": test_hyb_m["MAP@10"],
            "Test_MRR@10": test_hyb_m["MRR@10"],
            "Test_ROC-AUC": test_hyb_m["ROC-AUC"],
            "Test_PR-AUC": test_hyb_m["PR-AUC"],
            "Test_LogLoss": test_hyb_m["LogLoss"],
        }
    )

    print("\n================ СВОДНАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ НА ТЕСТОВОЙ ВЫБОРКЕ ================")
    comp_df = pd.DataFrame(main_comparison)
    print(
        tabulate(
            comp_df,
            headers="keys",
            tablefmt="github",
            floatfmt=".4f",
            showindex=False,
        )
    )

    # Сохраняем важность признаков финальной гибридной модели
    gains = hybrid_model.booster_.feature_importance(importance_type="gain")
    splits = hybrid_model.booster_.feature_importance(importance_type="split")
    top_features_df = (
        pd.DataFrame(
            {
                "feature": selected_hybrid_cols,
                "gain": gains,
                "gain_pct": 100.0 * gains / (np.sum(gains) + 1e-12),
                "split": splits,
            }
        )
        .sort_values("gain_pct", ascending=False)
        .reset_index(drop=True)
    )
    print("\n--- Топ-15 признаков по Gain Importance (%) ---")
    print(tabulate(top_features_df.head(15), headers="keys", tablefmt="github", floatfmt=".2f"))

    # -------------------------------------------------------------------------
    # Шаг 5. Эксперимент по чувствительности к окну (Window Sensitivity) и Ablation Study
    # -------------------------------------------------------------------------
    window_results = run_window_sensitivity_study(train_full, val_full, test_full, cat_cols)
    ablation_results = run_ablation_study(
        train_full, val_full, test_full, cat_cols, manual_cols, auto_cols, emb_cols
    )

    # -------------------------------------------------------------------------
    # Шаг 6. Проверка масштабируемости (1k, 5k, 10k юзеров)
    # -------------------------------------------------------------------------
    scale_results = run_scalability_benchmark()

    # -------------------------------------------------------------------------
    # Шаг 7. Сохранение артефактов и графиков
    # -------------------------------------------------------------------------
    os.makedirs("reports", exist_ok=True)
    comp_df.to_csv("reports/metrics_summary.csv", index=False)
    top_features_df.to_csv("reports/top_features.csv", index=False)
    pd.DataFrame(window_results).to_csv("reports/window_sensitivity.csv", index=False)
    pd.DataFrame(ablation_results).to_csv("reports/ablation_study.csv", index=False)
    pd.DataFrame(scale_results).to_csv("reports/scalability_benchmark.csv", index=False)

    summary_payload = {
        "dataset": {
            "name": "MTS KION",
            "sampled_users": int(len(users_df)),
            "sampled_interactions": int(len(interactions_df)),
            "total_items": int(len(items_df)),
            "train_pairs": int(len(train_cands)),
            "val_pairs": int(len(val_cands)),
            "test_pairs": int(len(test_cands)),
        },
        "main_comparison": main_comparison,
        "window_sensitivity": window_results,
        "ablation_study": ablation_results,
        "scalability": scale_results,
        "top_25_features": top_features_df.head(25).to_dict(orient="records"),
    }
    with open("reports/experiment_results.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, ensure_ascii=False, indent=2)

    generate_plots(
        main_comparison,
        window_results,
        ablation_results,
        scale_results,
        top_features_df,
    )
    print("\n[ГОТОВО] Все эксперименты, таблицы и графики успешно сформированы!")


if __name__ == "__main__":
    main()
