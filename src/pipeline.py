# Checkpoint 9 (CP09) — Task 9.1: Unified end-to-end analyse(text) function across all pipeline stages

import time
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.explain import explain_prediction
from src.features import intensity_clues
from src.fuzzy_pipeline import load_pipeline_config, polarity_to_3class, polarity_to_5class
from src.fuzzy_system import FuzzySentimentAnalyzer
from src.preprocess import for_explanation, for_roberta, for_vader

ROBERTA_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

_VADER_ENGINE = None
_ROBERTA_TOKENIZER = None
_ROBERTA_MODEL = None
_FUZZY_ENGINE = None


def _get_models():
    global _VADER_ENGINE, _ROBERTA_TOKENIZER, _ROBERTA_MODEL, _FUZZY_ENGINE
    if _VADER_ENGINE is None:
        _VADER_ENGINE = SentimentIntensityAnalyzer()
    if _ROBERTA_TOKENIZER is None or _ROBERTA_MODEL is None:
        _ROBERTA_TOKENIZER = AutoTokenizer.from_pretrained(ROBERTA_MODEL_NAME)
        _ROBERTA_MODEL = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL_NAME)
        _ROBERTA_MODEL.eval()
    if _FUZZY_ENGINE is None:
        _FUZZY_ENGINE = FuzzySentimentAnalyzer("config/rules.yaml")
    return _VADER_ENGINE, _ROBERTA_TOKENIZER, _ROBERTA_MODEL, _FUZZY_ENGINE


def validate_text_input(text: str, max_chars: int = 2000) -> str | None:
    """Return an error message if text is empty or exceeds max_chars, else None."""
    if text is None or not str(text).strip():
        return "Please enter a non-empty text before running analysis."
    if len(str(text)) > max_chars:
        return f"Text exceeds the {max_chars}-character limit ({len(str(text))} characters provided)."
    return None


def analyse(text: str, w: float | None = None) -> dict:
    """Run pre-processing, VADER, RoBERTa, feature fusion, Mamdani fuzzy trace, and explanation."""
    err = validate_text_input(text)
    if err:
        raise ValueError(err)

    cfg = load_pipeline_config()
    w_val = cfg["w"] if w is None else float(w)
    vader_eng, tok, model, fuzzy_eng = _get_models()

    t0 = time.perf_counter()

    t_vader = for_vader(text)
    t_rob = for_roberta(text)
    t_exp = for_explanation(text)

    vs = vader_eng.polarity_scores(t_vader)
    vpos = float(vs.get("pos", 0.0))
    vneg = float(vs.get("neg", 0.0))
    compound = float(vs.get("compound", 0.0))

    enc = tok(t_rob, return_tensors="pt", truncation=True, max_length=256, padding=True)
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

    p_neg = float(probs[0])
    p_neu = float(probs[1])
    p_pos = float(probs[2])

    clues = intensity_clues(text, vpos=vpos, vneg=vneg, compound=compound)
    pos_fused = float(np.clip(w_val * p_pos + (1.0 - w_val) * min(1.0, 2.0 * vpos), 0.0, 1.0))
    neg_fused = float(np.clip(w_val * p_neg + (1.0 - w_val) * min(1.0, 2.0 * vneg), 0.0, 1.0))
    int_fused = float(np.clip(clues["INT"], 0.0, 1.0))

    pos_degs = {k: round(fuzzy_eng._degree(fuzzy_eng.x_pos, mf, pos_fused), 3) for k, mf in fuzzy_eng.pos_mfs.items()}
    neg_degs = {k: round(fuzzy_eng._degree(fuzzy_eng.x_neg, mf, neg_fused), 3) for k, mf in fuzzy_eng.neg_mfs.items()}
    int_degs = {k: round(fuzzy_eng._degree(fuzzy_eng.x_int, mf, int_fused), 3) for k, mf in fuzzy_eng.int_mfs.items()}

    exp_res = explain_prediction(
        text=t_exp,
        pos=pos_fused,
        neg=neg_fused,
        intensity=int_fused,
        analyzer=fuzzy_eng,
        clues=clues,
    )

    pol = exp_res["polarity"]
    amb = exp_res["ambivalence"]
    pred_3 = polarity_to_3class(pol, neg_thresh=cfg["pol_neg_thresh"], pos_thresh=cfg["pol_pos_thresh"])
    pred_5 = polarity_to_5class(pol)
    elapsed_s = round(time.perf_counter() - t0, 4)

    return {
        "text": text,
        "clean_text": t_exp,
        "w": w_val,
        "vader": {"vpos": vpos, "vneg": vneg, "compound": compound},
        "roberta": {"p_neg": p_neg, "p_neu": p_neu, "p_pos": p_pos},
        "features": {"POS": round(pos_fused, 4), "NEG": round(neg_fused, 4), "INT": round(int_fused, 4)},
        "clues": clues,
        "degrees": {"POS": pos_degs, "NEG": neg_degs, "INT": int_degs},
        "polarity": round(pol, 4),
        "ambivalence": round(amb, 4),
        "is_mixed": bool(amb >= cfg["mixed_threshold"]),
        "pred_3class": pred_3,
        "pred_5class": pred_5,
        "verdict": exp_res["label"],
        "explanation": exp_res["explanation"],
        "flesch_score": exp_res["flesch_score"],
        "active_rules": exp_res["active_rules"],
        "latency_s": elapsed_s,
    }


def analyse_batch_df(df: pd.DataFrame, max_rows: int = 1000) -> pd.DataFrame:
    """Validate and run analyse() on a DataFrame containing a 'text' column (up to max_rows)."""
    if "text" not in df.columns:
        raise ValueError("Uploaded CSV must contain a 'text' column.")
    if len(df) > max_rows:
        raise ValueError(f"Batch upload supports up to {max_rows} rows ({len(df)} rows provided).")

    records = []
    for raw_txt in df["text"].fillna("").astype(str):
        err = validate_text_input(raw_txt)
        if err:
            records.append(
                {
                    "text": raw_txt,
                    "error": err,
                    "polarity": np.nan,
                    "ambivalence": np.nan,
                    "verdict": "error",
                    "explanation": "",
                }
            )
            continue
        res = analyse(raw_txt)
        records.append(
            {
                "text": raw_txt,
                "error": "",
                "polarity": res["polarity"],
                "ambivalence": res["ambivalence"],
                "is_mixed": res["is_mixed"],
                "pred_3class": res["pred_3class"],
                "pred_5class": res["pred_5class"],
                "verdict": res["verdict"],
                "flesch_score": res["flesch_score"],
                "explanation": res["explanation"],
            }
        )
    return pd.DataFrame(records)