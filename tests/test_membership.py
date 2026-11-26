# Checkpoint 5 (CP05) — Task 5.4: Unit tests for fuzzy memberships, rules, and defuzzification bounds

import numpy as np
from src.fuzzy_system import FuzzySentimentAnalyzer


def test_universes_and_output_bounds():
    engine = FuzzySentimentAnalyzer("config/rules.yaml")
    assert len(engine.x_pol) == 2001
    assert len(engine.x_amb) == 1001
    assert np.isclose(engine.x_pol.min(), -1.0) and np.isclose(engine.x_pol.max(), 1.0)
    assert np.isclose(engine.x_amb.min(), 0.0) and np.isclose(engine.x_amb.max(), 1.0)


def test_extreme_inputs_stay_within_bounds():
    engine = FuzzySentimentAnalyzer("config/rules.yaml")
    samples = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 1.0),
        (0.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (-0.5, 1.5, 2.0),  # out-of-range inputs should clip safely
    ]
    for pos, neg, intensity in samples:
        out = engine.infer(pos, neg, intensity)
        assert -1.0 <= out["polarity"] <= 1.0
        assert 0.0 <= out["ambivalence"] <= 1.0
        assert not np.isnan(out["polarity"])
        assert not np.isnan(out["ambivalence"])


def test_polarity_direction_matches_pos_vs_neg():
    engine = FuzzySentimentAnalyzer("config/rules.yaml")
    strong_pos = engine.infer(0.90, 0.05, 0.80)
    strong_neg = engine.infer(0.05, 0.90, 0.80)
    assert strong_pos["polarity"] > 0.3
    assert strong_neg["polarity"] < -0.3
    assert strong_pos["polarity"] > strong_neg["polarity"]


def test_ambivalence_higher_for_conflicted_than_one_sided():
    engine = FuzzySentimentAnalyzer("config/rules.yaml")
    one_sided = engine.infer(0.90, 0.05, 0.60)
    conflicted = engine.infer(0.85, 0.85, 0.60)
    assert conflicted["ambivalence"] > one_sided["ambivalence"]


def test_infer_batch_matches_single_infer():
    engine = FuzzySentimentAnalyzer("config/rules.yaml")
    pos_arr = [0.8, 0.1, 0.7]
    neg_arr = [0.1, 0.8, 0.7]
    int_arr = [0.7, 0.7, 0.5]
    pol_b, amb_b = engine.infer_batch(pos_arr, neg_arr, int_arr)
    for idx in range(len(pos_arr)):
        single = engine.infer(pos_arr[idx], neg_arr[idx], int_arr[idx])
        assert np.isclose(pol_b[idx], single["polarity"])
        assert np.isclose(amb_b[idx], single["ambivalence"])