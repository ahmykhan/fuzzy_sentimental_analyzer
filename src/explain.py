# Checkpoint 7 (CP07) — Tasks 7.1, 7.2 & 7.3: Plain-English explanation generator and Flesch readability check

import re
import numpy as np
import pandas as pd
from src.fuzzy_pipeline import load_pipeline_config
from src.fuzzy_system import FuzzySentimentAnalyzer
from src.preprocess import for_explanation

RULE_PHRASES = {
    "strong_positive_intense": "strong upsides and high stress",
    "strong_positive_normal": "clear upsides",
    "moderate_positive": "some upsides",
    "strong_negative_intense": "strong flaws and high stress",
    "strong_negative_normal": "clear flaws",
    "moderate_negative": "some flaws",
    "neutral_low_polarity": "weak upsides and weak flaws",
    "conflicted_high_both": "high upsides and high flaws together",
    "conflicted_med_both": "a mix of upsides and flaws",
    "mixed_high_pos_med_neg": "strong upsides with some flaws",
    "mixed_med_pos_high_neg": "strong flaws with some upsides",
    "intense_neutral_fallback": "high stress with a flat tone",
}


def count_syllables(word: str) -> int:
    """Estimate syllable count in an English word for Flesch Reading Ease."""
    cleaned = re.sub(r"[^a-z]", "", word.lower())
    if not cleaned:
        return 0
    if len(cleaned) <= 3:
        return 1
    cleaned = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", cleaned)
    cleaned = re.sub(r"^y", "", cleaned)
    vowels = re.findall(r"[aeiouy]{1,2}", cleaned)
    return max(1, len(vowels))


def flesch_reading_ease(text: str) -> float:
    """Compute Flesch Reading Ease score (higher = easier to read; >= 60 is plain English)."""
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    words = re.findall(r"[A-Za-z]+", text)
    if not sentences or not words:
        return 100.0
    total_syllables = sum(count_syllables(w) for w in words)
    asl = len(words) / len(sentences)
    asw = total_syllables / len(words)
    score = 206.835 - 1.015 * asl - 84.6 * asw
    return round(float(score), 2)


def _polarity_label(polarity: float, neg_thresh: float, pos_thresh: float) -> str:
    if polarity <= -0.55:
        return "very bad"
    if polarity < neg_thresh:
        return "bad"
    if polarity >= 0.55:
        return "very good"
    if polarity > pos_thresh:
        return "good"
    return "flat"


def explain_prediction(
    text: str,
    pos: float,
    neg: float,
    intensity: float,
    analyzer: FuzzySentimentAnalyzer | None = None,
    clues: dict | None = None,
) -> dict:
    """
    Generate a short, plain-English explanation for a fuzzy sentiment prediction.
    Returns polarity, ambivalence, label, explanation text, active rules, and Flesch score.
    """
    cfg = load_pipeline_config()
    engine = analyzer if analyzer is not None else FuzzySentimentAnalyzer("config/rules.yaml")
    traced = engine.infer_with_trace(pos, neg, intensity)

    pol = traced["polarity"]
    amb = traced["ambivalence"]
    active = traced["active_rules"]

    label = _polarity_label(pol, cfg["pol_neg_thresh"], cfg["pol_pos_thresh"])
    readable_text = for_explanation(text)

    s1 = f"Tone is {label} ({pol:+.2f})."

    if active:
        top_rule = active[0]
        phrase = RULE_PHRASES.get(top_rule["name"], top_rule["name"].replace("_", " "))
        s2 = f"It shows {phrase}."
    else:
        s2 = f"Upside is {pos:.2f} and flaw is {neg:.2f}."

    if amb >= cfg["mixed_threshold"]:
        s3 = "Mixed sides lift the mix score."
    elif clues and clues.get("clue_b", 0.0) > 0.0:
        s3 = "Exclamation marks add punch."
    elif clues and clues.get("clue_c", 0.0) > 0.0:
        s3 = "Caps add punch."
    elif clues and clues.get("clue_a", 0.0) > 0.0:
        s3 = "Extra words add punch."
    elif intensity >= 0.6:
        s3 = "The punch is strong."
    else:
        s3 = "The tone stays calm."

    explanation = f"{s1} {s2} {s3}"
    readability = flesch_reading_ease(explanation)

    return {
        "clean_text": readable_text,
        "label": label,
        "polarity": pol,
        "ambivalence": amb,
        "explanation": explanation,
        "flesch_score": readability,
        "active_rules": active,
    }


def check_dev_readability(dev_path: str = "data/processed/sst_dev_features.parquet", sample_n: int = 50) -> None:
    """Verify explanation readability on a dev sample and print 5 sample explanations."""
    df = pd.read_parquet(dev_path).head(sample_n)
    engine = FuzzySentimentAnalyzer("config/rules.yaml")
    text_col = "sentence" if "sentence" in df.columns else "text"

    scores = []
    samples = []
    for _, row in df.iterrows():
        clues = {
            "clue_a": float(row.get("clue_a", 0.0)),
            "clue_b": float(row.get("clue_b", 0.0)),
            "clue_c": float(row.get("clue_c", 0.0)),
            "clue_d": float(row.get("clue_d", 0.0)),
        }
        res = explain_prediction(
            text=str(row[text_col]),
            pos=float(row["POS"]),
            neg=float(row["NEG"]),
            intensity=float(row["INT"]),
            analyzer=engine,
            clues=clues,
        )
        scores.append(res["flesch_score"])
        if len(samples) < 5:
            samples.append(res)

    print(
        f"Checked {len(scores)} dev explanations: mean Flesch = {np.mean(scores):.2f}, min Flesch = {np.min(scores):.2f}\n",
        flush=True,
    )
    for i, s in enumerate(samples, start=1):
        print(f"{i}. Text: {s['clean_text']}")
        print(f"   Explanation (Flesch {s['flesch_score']}): {s['explanation']}\n")


if __name__ == "__main__":
    check_dev_readability()