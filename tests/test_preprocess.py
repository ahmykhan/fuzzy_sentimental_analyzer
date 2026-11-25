# Checkpoint 3 (CP03) — Task 3.4: Compute RoBERTa probabilities (p_neg, p_neu, p_pos) in batches of 32

import os
import time
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from src.preprocess import for_roberta

LOCAL_MODEL_DIR = "models/roberta_sentiment"
HUB_MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BATCH_SIZE = 32
MAX_LEN = 256


def load_roberta_model(device: torch.device):
    source = LOCAL_MODEL_DIR if os.path.isdir(LOCAL_MODEL_DIR) else HUB_MODEL_NAME
    print(f"Loading RoBERTa from: {source}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source).to(device)
    model.eval()
    return tokenizer, model


def compute_roberta_for_file(
    input_csv: str,
    output_parquet: str,
    tokenizer,
    model,
    device: torch.device,
) -> None:
    start_time = time.time()
    df = pd.read_csv(input_csv)
    text_col = "sentence" if "sentence" in df.columns else "text"
    cleaned_texts = [for_roberta(val) for val in df[text_col]]
    total = len(cleaned_texts)
    total_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    print(f"\nStarting {input_csv} ({total} rows, {total_batches} batches)...", flush=True)

    p_neg_scores = []
    p_neu_scores = []
    p_pos_scores = []

    with torch.no_grad():
        for batch_idx, start in enumerate(range(0, total, BATCH_SIZE), start=1):
            batch = cleaned_texts[start : start + BATCH_SIZE]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LEN,
                return_tensors="pt",
            ).to(device)

            outputs = model(**encoded)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

            for row in probs:
                p_neg_scores.append(float(row[0]))
                p_neu_scores.append(float(row[1]))
                p_pos_scores.append(float(row[2]))

            if batch_idx % 25 == 0 or batch_idx == total_batches:
                elapsed = time.time() - start_time
                print(f"  Batch {batch_idx}/{total_batches} ({elapsed:.1f}s)", flush=True)

    df["p_neg"] = p_neg_scores
    df["p_neu"] = p_neu_scores
    df["p_pos"] = p_pos_scores

    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    df.to_parquet(output_parquet, index=False)
    print(f"Saved {len(df)} rows -> {output_parquet} in {time.time() - start_time:.1f}s", flush=True)


def main() -> None:
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running RoBERTa on device: {device} (batch_size={BATCH_SIZE}, max_len={MAX_LEN})", flush=True)

    tokenizer, model = load_roberta_model(device)

    datasets = [
        ("data/processed/sst_train.csv", "data/processed/sst_train_roberta.parquet"),
        ("data/processed/sst_dev.csv", "data/processed/sst_dev_roberta.parquet"),
        ("data/processed/sst_test.csv", "data/processed/sst_test_roberta.parquet"),
        ("data/processed/tweet_test.csv", "data/processed/tweet_test_roberta.parquet"),
    ]

    for input_csv, output_parquet in datasets:
        compute_roberta_for_file(input_csv, output_parquet, tokenizer, model, device)


if __name__ == "__main__":
    main()