# Checkpoint 7 (CP07) — Task 7.4: Unit tests for explanation generation and Flesch readability

from src.explain import explain_prediction, flesch_reading_ease
from src.fuzzy_system import FuzzySentimentAnalyzer


def test_explain_returns_expected_fields():
    engine = FuzzySentimentAnalyzer("config/rules.yaml")
    res = explain_prediction("Great movie!", pos=0.90, neg=0.02, intensity=0.70, analyzer=engine)
    for key in ["clean_text", "label", "polarity", "ambivalence", "explanation", "flesch_score", "active_rules"]:
        assert key in res
    assert isinstance(res["explanation"], str) and len(res["explanation"]) > 10


def test_explain_mixed_mentions_mix():
    engine = FuzzySentimentAnalyzer("config/rules.yaml")
    res = explain_prediction(
        "Great cast, yet slow plot.",
        pos=0.85,
        neg=0.85,
        intensity=0.60,
        analyzer=engine,
    )
    assert "mix" in res["explanation"].lower()


def test_flesch_readability_passes_threshold():
    engine = FuzzySentimentAnalyzer("config/rules.yaml")
    res = explain_prediction(
        "It is a lovely film with nice acting.",
        pos=0.92,
        neg=0.02,
        intensity=0.40,
        analyzer=engine,
        clues={"clue_a": 0.0, "clue_b": 0.0, "clue_c": 0.0, "clue_d": 0.5},
    )
    assert res["flesch_score"] >= 60.0
    assert flesch_reading_ease("Short plain words work well here.") >= 60.0