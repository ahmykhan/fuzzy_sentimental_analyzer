<!-- Checkpoint 10 (CP10) — Task 10.5: 10-slide final presentation outline -->

# Final Presentation Outline (10 Slides)

1. **Title & Overview:** Hybrid VADER + RoBERTa + Mamdani Fuzzy Sentiment Analyser.
2. **Problem & Motivation:** Capturing polarity, mixed feelings (ambivalence), and readable explanations instead of rigid labels alone.
3. **Datasets & Pre-processing (`CP02`–`CP03`):** SST (`8,544` train / `1,101` dev / `2,210` test) and TweetEval test (`12,284`) with mode-specific text cleaning.
4. **Feature Engineering (`CP04`):** Intensity clues (`a`, `b`, `c`, `d` -> `INT`) and fused `POS`/`NEG` inputs.
5. **Dev Baselines (`CP04`):** Comparing `B1_VADER` and `B2_RoBERTa` on `sst_dev`.
6. **Mamdani Fuzzy System (`CP05`):** Membership functions (`results/membership.png`), rule firing, and centroid defuzzification.
7. **Dev Tuning (`CP06`):** Sweeping `w` and thresholds on `sst_dev` (`results/tuning_dev.csv`), locking `w = 0.9` and `[-0.15, 0.15]`.
8. **Plain-English Explanations (`CP07`):** Rule-trace explanations with readability verified (`Flesch >= 60`).
9. **Locked Test Evaluation (`CP08`):** Results on `sst_test` and `tweet_test` (`results/test_metrics.csv`) and error analysis.
10. **Desktop App & Handover (`CP09`–`CP10`):** `tkinter` app + CSV batch mode, user test notes, documentation, and `v1.0` release.