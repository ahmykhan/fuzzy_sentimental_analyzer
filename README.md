# Fuzzy Sentiment Analyser

Hybrid VADER + RoBERTa + Mamdani fuzzy inference pipeline (scikit-fuzzy) with polarity, ambivalence, plain-English explanations (Flesch >= 60), desktop GUI + batch CSV mode, 23 passing pytest tests, and locked test evaluation on sst_test (n=2,210) and tweet_test (n=12,284).

## Final Locked Test Results (w = 0.9, thresholds [-0.15, 0.15])

- **sst_test (`Fuzzy_System(w=0.9)`):** Pearson r = 0.7612, Spearman rho = 0.7533, MAE = 0.2985, 3-class Acc = 0.7000, 3-class Macro-F1 = 0.6306, 5-class Acc = 0.4466, 5-class Macro-F1 = 0.4454.
- **tweet_test (`Fuzzy_System(w=0.9)`):** Pearson r = 0.5975, Spearman rho = 0.6328, MAE = 0.7752, 3-class Acc = 0.2685, 3-class Macro-F1 = 0.2050, 5-class Acc = 0.1685, 5-class Macro-F1 = 0.0875.

## Quick Run

```powershell
python -m pytest
python -m app.desktop_app
python -m app.desktop_app --csv results\sample_input.csv --out results\batch_output.csv
```
