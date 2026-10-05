import os
import pandas as pd

def clean_sst_text(text: str) -> str:
    # Task 2.2: fix -LRB-, -RRB- and broken characters
    text = text.replace("-LRB-", "(").replace("-RRB-", ")")
    try:
        text = text.encode("latin1").decode("utf-8")
    except Exception:
        pass
    return text.strip()

def score_to_3class(score: float) -> str:
    if score <= 0.4:
        return "negative"
    elif score <= 0.6:
        return "neutral"
    return "positive"

def score_to_5class(score: float) -> str:
    if score <= 0.2:
        return "very negative"
    elif score <= 0.4:
        return "negative"
    elif score <= 0.6:
        return "neutral"
    elif score <= 0.8:
        return "positive"
    return "very positive"

def process_sst(raw_dir="data/raw", out_dir="data/processed"):
    os.makedirs(out_dir, exist_ok=True)
    
    sentences = pd.read_csv(f"{raw_dir}/datasetSentences.txt", sep="\t")
    sentences["sentence"] = sentences["sentence"].apply(clean_sst_text)
    
    splits = pd.read_csv(f"{raw_dir}/datasetSplit.txt", sep=",")
    df = sentences.merge(splits, on="sentence_index")
    
    dictionary = pd.read_csv(f"{raw_dir}/dictionary.txt", sep="|", names=["phrase", "phrase_id"])
    labels = pd.read_csv(f"{raw_dir}/sentiment_labels.txt", sep="|", header=0, names=["phrase_id", "score"])
    phrase_scores = dictionary.merge(labels, on="phrase_id")
    
    sst_full = df.merge(phrase_scores, left_on="sentence", right_on="phrase", how="inner")
    
    # Add 3-class and 5-class labels (Task 2.3)
    sst_full["label_3class"] = sst_full["score"].apply(score_to_3class)
    sst_full["label_5class"] = sst_full["score"].apply(score_to_5class)
    
    # 1=train (8,544), 2=test (2,210), 3=dev (1,101)
    train = sst_full[sst_full["splitset_label"] == 1]
    test  = sst_full[sst_full["splitset_label"] == 2]
    dev   = sst_full[sst_full["splitset_label"] == 3]
    
    train.to_csv(f"{out_dir}/sst_train.csv", index=False)
    dev.to_csv(f"{out_dir}/sst_dev.csv", index=False)
    test.to_csv(f"{out_dir}/sst_test.csv", index=False)
    
    print(f"SST Train: {len(train)} (Target: 8544)")
    print(f"SST Dev:   {len(dev)} (Target: 1101)")
    print(f"SST Test:  {len(test)} (Target: 2210)")

if __name__ == "__main__":
    process_sst()