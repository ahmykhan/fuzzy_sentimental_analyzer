# Internship Worklog

## Checkpoint 1 — Setup and Understanding
- **Tasks done:** 1.1 to 1.7 (environment setup, repo structure, VADER/RoBERTa/scikit-fuzzy demos, `docs/C1-requirements.md`).
- **Hours:** 16h
- **Problems:** Large RoBERTa model weights (`model.safetensors`, ~447 MB) caused Git push timeout; resolved by adding `models/` to `.gitignore`.

## Checkpoint 2 — Data Preparation
- **Tasks done:** 2.1 to 2.7 (SST raw download & notes, `src/data_loader.py`, clean SST CSVs, TweetEval test download, 6 unit tests, `notebooks/01_eda.ipynb`, `docs/mixed_examples.md`).
- **Hours:** 16h
- **Problems:** Unrelated Git histories blocked the initial Pull Request; resolved by merging upstream history with `--allow-unrelated-histories`.
- **Plan for next checkpoint:** Tasks 3.1 to 3.5 (preprocessing functions, unit tests, VADER and RoBERTa Parquet score files).

## Checkpoint 3 — Pre-processing and Features (Part 1)
- **Tasks done:** 3.1 to 3.5 (`src/preprocess.py`, `tests/test_preprocess.py` with 8 passing tests, `src/compute_vader.py` generating `*_vader.parquet`, `src/compute_roberta.py` generating `*_roberta.parquet` in batches of 32, probability sum verification, 10 hand-checked sentences, and `docs/checkpoints/CP03.md`).
- **Hours:** 16h
- **Problems:** Fixed `don't` contraction expansion, ran pytest via `python -m pytest` for `src` imports, and added `flush=True` batch progress logs every 25 batches so CPU inference progress was visible.
- **Plan for next checkpoint:** Tasks 4.1 to 4.6 (`src/features.py`, `POS`/`NEG`/`INT` feature table, `notebooks/02_features.ipynb`, `src/evaluate.py`, `results/baselines_dev.csv`, `tests/test_features.py`, and CP04 report).