# Checkpoint 8 (CP08) — Final test-set evaluation on sst_test and tweet_test -> results/test_metrics.csv

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from src.evaluate import infer_true_targets, score_to_5class
from src.fuzzy_pipeline import load_pipeline_config, polarity_to_3class, run_fuzzy_pipeline_df
from src.fuzzy_system import FuzzySentimentAnalyzer


def _metrics_row(split_name: str, model_name: str, y_true: np.ndarray, y_pred: np.ndarray, y3_true: np.ndarray, y5_true: np.ndarray, neg_th: float, pos_th: float) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y3_pred = np.array([polarity_to_3class(v, neg_thresh=neg_th, pos_thresh=pos_th) for v in y_pred], dtype=int)
    y5_pred = np.array([score_to_5class(v) for v in y_pred], dtype=int)

    return {
        "split": split_name,
        "model": model_name,
        "n_rows": int(len(y_true)),
        "pearson_r": round(float(pd.Series(y_pred).corr(pd.Series(y_true), method="pearson")), 4),
        "spearman_rho": round(float(pd.Series(y_pred).corr(pd.Series(y_true), method="spearman")), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "acc_3class": round(float(accuracy_score(y3_true, y3_pred)), 4),
        "macro_f1_3class": round(float(f1_score(y3_true, y3_pred, average="macro", zero_division=0)), 4),
        "acc_5class": round(float(accuracy_score(y5_true, y5_pred)), 4),
        "macro_f1_5class": round(float(f1_score(y5_true, y5_pred, average="macro", zero_division=0)), 4),
    }


def run_test_evaluation(
    data_dir: str = "data/processed",
    out_csv: str = "results/test_metrics.csv",
) -> pd.DataFrame:
    """Evaluate B1_VADER, B2_RoBERTa, and Fuzzy_System on sst_test and tweet_test."""
    cfg = load_pipeline_config()
    w_tuned = cfg["w"]
    neg_th = cfg["pol_neg_thresh"]
    pos_th = cfg["pol_pos_thresh"]

    engine = FuzzySentimentAnalyzer("config/rules.yaml")
    splits = ["sst_test", "tweet_test"]
    rows = []

    for split in splits:
        feat_path = os.path.join(data_dir, f"{split}_features.parquet")
        df = pd.read_parquet(feat_path)
        y_cont, y3_true, y5_true = infer_true_targets(df)

        b1_pred = np.clip(df["compound"].astype(float).values, -1.0, 1.0)
        b2_pred = np.clip((df["p_pos"] - df["p_neg"]).astype(float).values, -1.0, 1.0)

        scored = run_fuzzy_pipeline_df(
            df,
            analyzer=engine,
            w=w_tuned,
            pol_neg_thresh=neg_th,
            pol_pos_thresh=pos_th,
            recompute_features=True,
        )
        fuz_pred = scored["fuzzy_polarity"].astype(float).values

        rows.append(_metrics_row(split, "B1_VADER", y_cont, b1_pred, y3_true, y5_true, neg_th, pos_th))
        rows.append(_metrics_row(split, "B2_RoBERTa", y_cont, b2_pred, y3_true, y5_true, neg_th, pos_th))
        rows.append(_metrics_row(split, f"Fuzzy_System(w={w_tuned})", y_cont, fuz_pred, y3_true, y5_true, neg_th, pos_th))

    res_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    res_df.to_csv(out_csv, index=False)
    print(f"Saved final test metrics -> {out_csv}\n", flush=True)
    print(res_df.to_string(index=False), flush=True)
    return res_df


if __name__ == "__main__":
    run_test_evaluation()