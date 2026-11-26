# Checkpoint 6 (CP06) — Task 6.1: End-to-end fuzzy inference pipeline over feature tables

import os
import yaml
import numpy as np
import pandas as pd
from src.features import fuse_features_df, load_fusion_weight
from src.fuzzy_system import FuzzySentimentAnalyzer


def load_pipeline_config(config_path: str = "config/config.yaml") -> dict:
    """Load fusion and threshold settings from config/config.yaml."""
    defaults = {
        "w": 0.7,
        "mixed_threshold": 0.40,
        "pol_pos_thresh": 0.10,
        "pol_neg_thresh": -0.10,
    }
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        fusion_cfg = cfg.get("fusion", {})
        thresh_cfg = cfg.get("thresholds", {})
        defaults["w"] = float(fusion_cfg.get("w", defaults["w"]))
        defaults["mixed_threshold"] = float(fusion_cfg.get("mixed_threshold", defaults["mixed_threshold"]))
        defaults["pol_pos_thresh"] = float(thresh_cfg.get("pol_pos", defaults["pol_pos_thresh"]))
        defaults["pol_neg_thresh"] = float(thresh_cfg.get("pol_neg", defaults["pol_neg_thresh"]))
    return defaults


def polarity_to_3class(val: float, neg_thresh: float = -0.10, pos_thresh: float = 0.10) -> int:
    """Convert defuzzified polarity in [-1, 1] to 3 classes: 0=neg, 1=neu, 2=pos."""
    if val < neg_thresh:
        return 0
    if val > pos_thresh:
        return 2
    return 1


def polarity_to_5class(val: float) -> int:
    """Convert defuzzified polarity in [-1, 1] to 5 SST classes: 0..4."""
    clipped = float(np.clip(val, -1.0, 0.999999))
    scaled = (clipped + 1.0) / 2.0
    return int(np.floor(scaled * 5.0))


def run_fuzzy_pipeline_df(
    df: pd.DataFrame,
    analyzer: FuzzySentimentAnalyzer | None = None,
    w: float | None = None,
    mixed_threshold: float | None = None,
    pol_neg_thresh: float | None = None,
    pol_pos_thresh: float | None = None,
    recompute_features: bool = False,
) -> pd.DataFrame:
    """
    Run fuzzy inference across a dataframe.
    If recompute_features=True (or POS/NEG/INT missing), fuses features using weight w.
    """
    cfg = load_pipeline_config()
    w_val = cfg["w"] if w is None else float(w)
    mix_thresh = cfg["mixed_threshold"] if mixed_threshold is None else float(mixed_threshold)
    neg_th = cfg["pol_neg_thresh"] if pol_neg_thresh is None else float(pol_neg_thresh)
    pos_th = cfg["pol_pos_thresh"] if pol_pos_thresh is None else float(pol_pos_thresh)

    if recompute_features or not {"POS", "NEG", "INT"}.issubset(df.columns):
        out_df = fuse_features_df(df, df, w=w_val)
    else:
        out_df = df.copy()

    engine = analyzer if analyzer is not None else FuzzySentimentAnalyzer("config/rules.yaml")
    pol_arr, amb_arr = engine.infer_batch(
        out_df["POS"].astype(float).values,
        out_df["NEG"].astype(float).values,
        out_df["INT"].astype(float).values,
    )

    out_df["fuzzy_polarity"] = pol_arr
    out_df["fuzzy_ambivalence"] = amb_arr
    out_df["is_mixed"] = out_df["fuzzy_ambivalence"] >= mix_thresh
    out_df["fuzzy_pred_3class"] = [
        polarity_to_3class(v, neg_thresh=neg_th, pos_thresh=pos_th) for v in pol_arr
    ]
    out_df["fuzzy_pred_5class"] = [polarity_to_5class(v) for v in pol_arr]
    return out_df


if __name__ == "__main__":
    w_curr = load_fusion_weight()
    dev_path = "data/processed/sst_dev_features.parquet"
    dev_df = pd.read_parquet(dev_path)
    scored = run_fuzzy_pipeline_df(dev_df, w=w_curr)
    print(f"Ran fuzzy pipeline on {len(scored)} dev rows (w={w_curr}).")
    print(scored[["fuzzy_polarity", "fuzzy_ambivalence", "is_mixed", "fuzzy_pred_3class"]].describe().round(4))