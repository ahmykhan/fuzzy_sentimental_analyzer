# Checkpoint 4 (CP04) — Tasks 4.1 & 4.2: Intensity clues (a, b, c, d) and POS, NEG, INT feature fusion

import os
import re
import yaml
import numpy as np
import pandas as pd
from src.preprocess import clean_brackets

BOOSTERS = {
    "very", "extremely", "really", "so", "super", "totally", "utterly",
    "completely", "absolutely", "awfully", "terribly", "incredibly",
    "remarkably", "exceptionally", "deeply", "highly", "pretty", "quite",
}


def load_fusion_weight(config_path: str = "config/config.yaml", default_w: float = 0.7) -> float:
    """Load fusion weight w from config/config.yaml."""
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        if "fusion" in cfg and "w" in cfg["fusion"]:
            return float(cfg["fusion"]["w"])
    return default_w


def intensity_clues(text: str, vpos: float = 0.0, vneg: float = 0.0, compound: float = 0.0) -> dict:
    """
    Compute the four intensity clues a, b, c, d in [0, 1] and their average INT:
      a: booster words share (capped at 1.0)
      b: exclamation mark intensity (scaled)
      c: ALL-CAPS word share (excluding single-letter 'I' and 'A')
      d: overall VADER strength (|compound| or vpos + vneg capped at 1.0)
    """
    cleaned = clean_brackets(text if isinstance(text, str) else "")
    words = re.findall(r"[A-Za-z']+", cleaned)
    word_count = len(words)

    # Clue a: booster words share
    booster_hits = sum(1 for w in words if w.lower() in BOOSTERS)
    a = min(1.0, (booster_hits / max(word_count, 1)) * 5.0) if booster_hits > 0 else 0.0

    # Clue b: exclamation marks intensity
    excl_count = cleaned.count("!")
    b = min(1.0, excl_count / 3.0)

    # Clue c: ALL-CAPS word share (ignore single letters like 'I' or 'A')
    caps_words = sum(1 for w in words if len(w) >= 2 and w.isupper())
    c = min(1.0, caps_words / max(word_count, 1))

    # Clue d: VADER overall strength
    d = min(1.0, max(abs(float(compound)), min(1.0, float(vpos) + float(vneg))))

    int_score = float(np.mean([a, b, c, d]))
    return {
        "clue_a": float(a),
        "clue_b": float(b),
        "clue_c": float(c),
        "clue_d": float(d),
        "INT": int_score,
    }


def fuse_features_df(roberta_df: pd.DataFrame, vader_df: pd.DataFrame, w: float = 0.7) -> pd.DataFrame:
    """Combine RoBERTa (p_pos, p_neg) and VADER (vpos, vneg, compound) into POS, NEG, INT."""
    df = roberta_df.copy()
    df["vpos"] = vader_df["vpos"].values
    df["vneg"] = vader_df["vneg"].values
    df["compound"] = vader_df["compound"].values

    df["POS"] = w * df["p_pos"] + (1.0 - w) * np.minimum(1.0, 2.0 * df["vpos"])
    df["NEG"] = w * df["p_neg"] + (1.0 - w) * np.minimum(1.0, 2.0 * df["vneg"])

    text_col = "sentence" if "sentence" in df.columns else "text"
    clues = [
        intensity_clues(txt, vp, vn, cmp_val)
        for txt, vp, vn, cmp_val in zip(df[text_col], df["vpos"], df["vneg"], df["compound"])
    ]
    clues_df = pd.DataFrame(clues)
    for col in clues_df.columns:
        df[col] = clues_df[col].values

    return df


def build_and_save_features(out_dir: str = "data/processed") -> None:
    """Build feature tables for SST train/dev/test and TweetEval test using w from config.yaml."""
    w = load_fusion_weight()
    print(f"Building POS, NEG, INT features with w={w}...", flush=True)

    splits = ["sst_train", "sst_dev", "sst_test", "tweet_test"]
    for name in splits:
        rob_path = os.path.join(out_dir, f"{name}_roberta.parquet")
        vad_path = os.path.join(out_dir, f"{name}_vader.parquet")
        out_path = os.path.join(out_dir, f"{name}_features.parquet")

        rob_df = pd.read_parquet(rob_path)
        vad_df = pd.read_parquet(vad_path)
        feat_df = fuse_features_df(rob_df, vad_df, w=w)
        feat_df.to_parquet(out_path, index=False)
        print(f"Saved {len(feat_df)} rows -> {out_path}", flush=True)


if __name__ == "__main__":
    build_and_save_features()