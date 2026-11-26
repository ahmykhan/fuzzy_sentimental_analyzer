# Checkpoint 4 (CP04) — Task 4.6: Unit tests for intensity clues, INT, and POS/NEG feature fusion

import numpy as np
import pandas as pd
from src.features import intensity_clues, fuse_features_df, load_fusion_weight


def test_intensity_clues_range_and_keys():
    res = intensity_clues("This is a normal review.", vpos=0.2, vneg=0.0, compound=0.2)
    for key in ["clue_a", "clue_b", "clue_c", "clue_d", "INT"]:
        assert key in res
        assert 0.0 <= res[key] <= 1.0


def test_booster_increases_clue_a():
    plain = intensity_clues("The movie was good.", vpos=0.4, vneg=0.0, compound=0.44)
    boosted = intensity_clues("The movie was extremely good and very entertaining.", vpos=0.4, vneg=0.0, compound=0.44)
    assert boosted["clue_a"] > plain["clue_a"]


def test_exclamation_increases_clue_b():
    no_excl = intensity_clues("Great acting", vpos=0.5, vneg=0.0, compound=0.62)
    with_excl = intensity_clues("Great acting!!!", vpos=0.5, vneg=0.0, compound=0.62)
    assert with_excl["clue_b"] > no_excl["clue_b"]
    assert np.isclose(with_excl["clue_b"], 1.0)


def test_caps_increases_clue_c_without_single_i():
    only_i = intensity_clues("I liked the plot.", vpos=0.3, vneg=0.0, compound=0.42)
    with_caps = intensity_clues("I AWFULLY LOVED the plot.", vpos=0.3, vneg=0.0, compound=0.42)
    assert only_i["clue_c"] == 0.0
    assert with_caps["clue_c"] > 0.0


def test_vader_strength_sets_clue_d():
    res = intensity_clues("Neutral text", vpos=0.0, vneg=0.0, compound=-0.85)
    assert np.isclose(res["clue_d"], 0.85)


def test_fuse_features_df_output_columns_and_bounds():
    rob_df = pd.DataFrame({
        "sentence": ["Awesome film!", "Terrible waste of time."],
        "p_neg": [0.02, 0.90],
        "p_neu": [0.05, 0.07],
        "p_pos": [0.93, 0.03],
    })
    vad_df = pd.DataFrame({
        "vpos": [0.60, 0.00],
        "vneg": [0.00, 0.55],
        "compound": [0.75, -0.70],
    })
    fused = fuse_features_df(rob_df, vad_df, w=0.7)
    for col in ["POS", "NEG", "INT", "clue_a", "clue_b", "clue_c", "clue_d"]:
        assert col in fused.columns
        assert ((fused[col] >= 0.0) & (fused[col] <= 1.0)).all()
    assert fused.loc[0, "POS"] > fused.loc[1, "POS"]
    assert fused.loc[1, "NEG"] > fused.loc[0, "NEG"]


def test_load_fusion_weight_reads_config():
    w = load_fusion_weight("config/config.yaml")
    assert isinstance(w, float)
    assert 0.0 <= w <= 1.0