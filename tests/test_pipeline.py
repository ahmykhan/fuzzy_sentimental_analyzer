# Checkpoint 9 (CP09) — Task 9.6: Pipeline speed and validation tests

import pandas as pd
import pytest
from src.pipeline import analyse, analyse_batch_df, validate_text_input


def test_validation_rejects_empty_and_overlong():
    assert validate_text_input("") is not None
    assert validate_text_input("   ") is not None
    assert validate_text_input("a" * 2001) is not None
    assert validate_text_input("A short valid review.") is None


def test_batch_rejects_missing_text_column():
    bad_df = pd.DataFrame({"review": ["Good movie"]})
    with pytest.raises(ValueError, match="text"):
        analyse_batch_df(bad_df)


def test_analyse_speed_and_structure():
    res = analyse("The acting was brilliant, but the story was boring.")
    assert "polarity" in res and "ambivalence" in res and "explanation" in res
    assert res["latency_s"] <= 3.0