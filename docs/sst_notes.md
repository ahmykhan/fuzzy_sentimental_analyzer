# Stanford Sentiment Treebank (SST) — Raw File Structure

The SST archive (`stanfordSentimentTreebank.zip`) splits the dataset across four relational text files in `data/raw/`:

1. **`datasetSentences.txt`** (tab-separated): Contains `sentence_index` and the raw review text (`sentence`). Contains Treebank bracket tokens (`-LRB-`, `-RRB-`) and latin1/UTF-8 encoding artifacts that are cleaned in `src/data_loader.py`.
2. **`datasetSplit.txt`** (comma-separated): Maps each `sentence_index` to `splitset_label` (`1` = train, `2` = test, `3` = dev).
3. **`dictionary.txt`** (pipe-separated `|`): Maps every sentence and sub-phrase (`phrase`) to a unique `phrase_id`.
4. **`sentiment_labels.txt`** (pipe-separated `|`): Maps each `phrase_id` to a continuous sentiment score between `0.0` (very negative) and `1.0` (very positive).

## How They Link
- `datasetSentences.txt` joins with `datasetSplit.txt` on `sentence_index`.
- `dictionary.txt` joins with `sentiment_labels.txt` on `phrase_id`.
- The two merged tables join on `sentence == phrase` to produce the final labeled splits: **8,544** train, **1,101** dev, and **2,210** test rows.