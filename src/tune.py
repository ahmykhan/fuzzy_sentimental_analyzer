# Checkpoint 6 (CP06) — Tasks 6.2, 6.3 & 6.4: Dev sweep over fusion weight w and thresholds -> results/tuning_dev.csv

import os
import yaml
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from src.evaluate import infer_true_targets, score_to_5class
from src.features import fuse_features_df
from src.fuzzy_pipeline import polarity_to_3class
from src.fuzzy_system import FuzzySentimentAnalyzer


def tune_on_dev(
    dev_path: str = "data/processed/sst_dev_features.parquet",
    out_csv: str = "results/tuning_dev.csv",
    config_path: str = "config/config.yaml",
    update_config: bool = True,
) -> pd.DataFrame:
    """Sweep w in [0.0, 1.0] and 3-class polarity thresholds on dev set only."""
    df = pd.read_parquet(dev_path)
    y_cont, y3_true, y5_true = infer_true_targets(df)
    engine = FuzzySentimentAnalyzer("config/rules.yaml")

    weights = [round(x, 1) for x in np.linspace(0.0, 1.0, 11)]
    threshold_pairs = [
        (-0.05, 0.05),
        (-0.10, 0.10),
        (-0.15, 0.15),
        (-0.20, 0.20),
    ]

    records = []
    for w in weights:
        fused = fuse_features_df(df, df, w=w)
        pol_arr, amb_arr = engine.infer_batch(
            fused["POS"].astype(float).values,
            fused["NEG"].astype(float).values,
            fused["INT"].astype(float).values,
        )

        pearson_r = float(pd.Series(pol_arr).corr(pd.Series(y_cont), method="pearson"))
        spearman_rho = float(pd.Series(pol_arr).corr(pd.Series(y_cont), method="spearman"))
        mae = float(mean_absolute_error(y_cont, pol_arr))
        y5_pred = np.array([score_to_5class(v) for v in pol_arr], dtype=int)
        acc_5 = float(accuracy_score(y5_true, y5_pred))
        f1_5 = float(f1_score(y5_true, y5_pred, average="macro", zero_division=0))

        for neg_th, pos_th in threshold_pairs:
            y3_pred = np.array(
                [polarity_to_3class(v, neg_thresh=neg_th, pos_thresh=pos_th) for v in pol_arr],
                dtype=int,
            )
            acc_3 = float(accuracy_score(y3_true, y3_pred))
            f1_3 = float(f1_score(y3_true, y3_pred, average="macro", zero_division=0))

            records.append(
                {
                    "w": w,
                    "neg_thresh": neg_th,
                    "pos_thresh": pos_th,
                    "pearson_r": round(pearson_r, 4),
                    "spearman_rho": round(spearman_rho, 4),
                    "mae": round(mae, 4),
                    "acc_3class": round(acc_3, 4),
                    "macro_f1_3class": round(f1_3, 4),
                    "acc_5class": round(acc_5, 4),
                    "macro_f1_5class": round(f1_5, 4),
                    "mean_ambivalence": round(float(np.mean(amb_arr)), 4),
                }
            )

    grid_df = pd.DataFrame(records)
    # Rank primarily by 3-class macro-F1, then Pearson r, then lower MAE
    grid_df = grid_df.sort_values(
        by=["macro_f1_3class", "pearson_r", "mae"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    grid_df.to_csv(out_csv, index=False)

    best = grid_df.iloc[0]
    if update_config and os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        cfg.setdefault("fusion", {})["w"] = float(best["w"])
        cfg["thresholds"] = {
            "pol_neg": float(best["neg_thresh"]),
            "pol_pos": float(best["pos_thresh"]),
        }
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"Saved {len(grid_df)} tuning rows -> {out_csv}\n", flush=True)
    print("Top 5 dev configurations:", flush=True)
    print(grid_df.head(5).to_string(index=False), flush=True)
    return grid_df


if __name__ == "__main__":
    tune_on_dev()