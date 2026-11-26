# Checkpoint 4 (CP04) — Tasks 4.4 & 4.5: Evaluation metrics and dev baseline comparison (B1 VADER, B2 RoBERTa)

import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

LABEL_3CLASS_MAP = {
    "neg": 0,
    "negative": 0,
    "0": 0,
    0: 0,
    "neu": 1,
    "neutral": 1,
    "1": 1,
    1: 1,
    "pos": 2,
    "positive": 2,
    "2": 2,
    2: 2,
}


def score_to_3class(val: float) -> int:
    """Map a [-1, 1] continuous score into 3 classes: 0=neg, 1=neu, 2=pos."""
    if val < -0.1:
        return 0
    if val > 0.1:
        return 2
    return 1


def score_to_5class(val: float) -> int:
    """Map a [-1, 1] continuous score into 5 SST classes: 0..4."""
    clipped = float(np.clip(val, -1.0, 0.999999))
    scaled = (clipped + 1.0) / 2.0  # [0, 1)
    return int(np.floor(scaled * 5.0))


def infer_true_targets(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract continuous target y_true (mapped to [-1, 1] if in [0, 1]),
    3-class ground truth y3_true, and 5-class ground truth y5_true.
    """
    if "score" in df.columns:
        raw_y = df["score"].astype(float).values
    elif "label" in df.columns:
        raw_y = df["label"].astype(float).values
    else:
        raise ValueError("No 'score' or 'label' column found in dataframe.")

    if raw_y.min() >= 0.0 and raw_y.max() <= 1.0:
        y_cont = np.clip(2.0 * raw_y - 1.0, -1.0, 1.0)
    else:
        y_cont = np.clip(raw_y, -1.0, 1.0)

    if "label_3class" in df.columns:
        mapped = df["label_3class"].map(
            lambda v: LABEL_3CLASS_MAP.get(v.strip().lower() if isinstance(v, str) else v, np.nan)
        )
        if mapped.isna().any():
            bad_vals = df.loc[mapped.isna(), "label_3class"].unique().tolist()
            raise ValueError(f"Unmapped label_3class values: {bad_vals}")
        y3 = mapped.astype(int).values
    else:
        y3 = np.array([score_to_3class(v) for v in y_cont], dtype=int)

    if "label_5class" in df.columns and pd.api.types.is_numeric_dtype(df["label_5class"]):
        y5 = df["label_5class"].astype(int).values
    elif "fine_label" in df.columns and pd.api.types.is_numeric_dtype(df["fine_label"]):
        y5 = df["fine_label"].astype(int).values
    else:
        y5 = np.array([score_to_5class(v) for v in y_cont], dtype=int)

    return y_cont, y3, y5


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y3_true: np.ndarray, y5_true: np.ndarray) -> dict:
    """Compute Pearson, Spearman, MAE, 5-class & 3-class macro-F1, and 5-class & 3-class accuracy."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    pearson_r = float(pd.Series(y_pred).corr(pd.Series(y_true), method="pearson"))
    spearman_rho = float(pd.Series(y_pred).corr(pd.Series(y_true), method="spearman"))
    mae = float(mean_absolute_error(y_true, y_pred))

    y3_pred = np.array([score_to_3class(v) for v in y_pred], dtype=int)
    y5_pred = np.array([score_to_5class(v) for v in y_pred], dtype=int)

    acc_3 = float(accuracy_score(y3_true, y3_pred))
    f1_3 = float(f1_score(y3_true, y3_pred, average="macro", zero_division=0))

    acc_5 = float(accuracy_score(y5_true, y5_pred))
    f1_5 = float(f1_score(y5_true, y5_pred, average="macro", zero_division=0))

    return {
        "pearson_r": round(pearson_r, 4),
        "spearman_rho": round(spearman_rho, 4),
        "mae": round(mae, 4),
        "acc_3class": round(acc_3, 4),
        "macro_f1_3class": round(f1_3, 4),
        "acc_5class": round(acc_5, 4),
        "macro_f1_5class": round(f1_5, 4),
    }


def run_dev_baselines(
    dev_path: str = "data/processed/sst_dev_features.parquet",
    out_csv: str = "results/baselines_dev.csv",
) -> pd.DataFrame:
    """Evaluate B1 (VADER compound) and B2 (RoBERTa p_pos - p_neg) on dev and save CSV."""
    df = pd.read_parquet(dev_path)
    y_cont, y3_true, y5_true = infer_true_targets(df)

    # B1: VADER-only (compound score is already in [-1, 1])
    b1_pred = np.clip(df["compound"].astype(float).values, -1.0, 1.0)
    b1_metrics = evaluate_predictions(y_cont, b1_pred, y3_true, y5_true)
    b1_metrics["model"] = "B1_VADER"

    # B2: RoBERTa-only (net polarity p_pos - p_neg is in [-1, 1])
    b2_pred = np.clip((df["p_pos"] - df["p_neg"]).astype(float).values, -1.0, 1.0)
    b2_metrics = evaluate_predictions(y_cont, b2_pred, y3_true, y5_true)
    b2_metrics["model"] = "B2_RoBERTa"

    res_df = pd.DataFrame([b1_metrics, b2_metrics])[
        ["model", "pearson_r", "spearman_rho", "mae", "acc_3class", "macro_f1_3class", "acc_5class", "macro_f1_5class"]
    ]

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    res_df.to_csv(out_csv, index=False)
    print(f"Saved dev baselines -> {out_csv}\n", flush=True)
    print(res_df.to_string(index=False), flush=True)
    return res_df


if __name__ == "__main__":
    run_dev_baselines()