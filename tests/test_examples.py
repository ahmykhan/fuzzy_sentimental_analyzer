# Checkpoint 5 (CP05) — Task 5.5: Example-based behavioral tests for positive, negative, neutral, intense & mixed inputs

from src.features import intensity_clues
from src.fuzzy_system import FuzzySentimentAnalyzer


def _run_case(text: str, pos: float, neg: float, vpos: float, vneg: float, compound: float):
    engine = FuzzySentimentAnalyzer("config/rules.yaml")
    clues = intensity_clues(text, vpos=vpos, vneg=vneg, compound=compound)
    out = engine.infer(pos, neg, clues["INT"])
    return clues, out


def test_positive_example():
    _, out = _run_case("A warm, funny, engaging film.", pos=0.92, neg=0.02, vpos=0.55, vneg=0.0, compound=0.72)
    assert out["polarity"] > 0.35
    assert out["ambivalence"] < 0.5


def test_negative_example():
    _, out = _run_case("A boring, terrible waste of time.", pos=0.02, neg=0.92, vpos=0.0, vneg=0.55, compound=-0.75)
    assert out["polarity"] < -0.35
    assert out["ambivalence"] < 0.5


def test_neutral_example():
    _, out = _run_case("The movie runs for two hours.", pos=0.10, neg=0.10, vpos=0.0, vneg=0.0, compound=0.0)
    assert -0.25 <= out["polarity"] <= 0.25


def test_intense_positive_example():
    clues_plain, out_plain = _run_case("Good movie.", pos=0.80, neg=0.05, vpos=0.40, vneg=0.0, compound=0.44)
    clues_intense, out_intense = _run_case(
        "EXTREMELY GOOD AND AMAZING MOVIE!!!", pos=0.80, neg=0.05, vpos=0.65, vneg=0.0, compound=0.85
    )
    assert clues_intense["INT"] > clues_plain["INT"]
    assert out_intense["polarity"] >= out_plain["polarity"] - 1e-6


def test_mixed_conflicted_example():
    _, out = _run_case(
        "Great acting and cinematography, yet terribly paced and boring.",
        pos=0.82,
        neg=0.80,
        vpos=0.35,
        vneg=0.35,
        compound=0.0,
    )
    assert out["ambivalence"] > 0.5
    