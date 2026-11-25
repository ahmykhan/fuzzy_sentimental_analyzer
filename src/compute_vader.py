# Checkpoint 3 (CP03) — Task 3.3: Compute VADER scores (vpos, vneg, compound) and save as *_vader.parquet

import os
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from src.preprocess import for_vader


def compute_vader_for_file(input_csv: str, output_parquet: str, analyzer: SentimentIntensityAnalyzer) -> None:
    """Read a CSV split, run VADER on text cleaned by for_vader, and save Parquet with vpos, vneg, compound."""
    df = pd.read_csv(input_csv)

    # SST splits use 'sentence' column; TweetEval uses 'text' column
    text_col = "sentence" if "sentence" in df.columns else "text"

    vpos_scores = []
    vneg_scores = []
    compound_scores = []

    for raw_text in df[text_col]:
        cleaned = for_vader(raw_text)
        scores = analyzer.polarity_scores(cleaned)
        vpos_scores.append(scores["pos"])
        vneg_scores.append(scores["neg"])
        compound_scores.append(scores["compound"])

    df["vpos"] = vpos_scores
    df["vneg"] = vneg_scores
    df["compound"] = compound_scores

    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    print(f"Saved {len(df)} rows -> {output_parquet}")


def main() -> None:
    analyzer = SentimentIntensityAnalyzer()

    datasets = [
        ("data/processed/sst_train.csv", "data/processed/sst_train_vader.parquet"),
        ("data/processed/sst_dev.csv", "data/processed/sst_dev_vader.parquet"),
        ("data/processed/sst_test.csv", "data/processed/sst_test_vader.parquet"),
        ("data/processed/tweet_test.csv", "data/processed/tweet_test_vader.parquet"),
    ]

    for input_csv, output_parquet in datasets:
        compute_vader_for_file(input_csv, output_parquet, analyzer)


if __name__ == "__main__":
    main()