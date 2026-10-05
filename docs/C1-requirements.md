# Fuzzy Sentiment Analyser - Requirements Summary

## 1. Project Goal
The objective is to develop a local, standalone sentiment analysis application that evaluates English sentences or short texts, determines polarity and ambivalence (mixed feelings), and produces an interpretable, plain-English explanation for its verdict.

## 2. Technical Architecture & Pipeline
The application runs across four main processing stages:
1. **Feature Extraction:**
   - **VADER:** Computes word-level lexical proportions (`vpos`, `vneg`) and punctuation/capitalization intensity clues.
   - **RoBERTa:** Uses a pre-trained transformer (`cardiffnlp/twitter-roberta-base-sentiment-latest`) to compute contextual class probabilities (`p_pos`, `p_neg`).
   - Combines both into three bounded inputs ($0.0$ to $1.0$): `POS`, `NEG`, and `INT` (intensity).
2. **Fuzzy Inference System:**
   - Fuzzifies inputs into `Low`, `Medium`, and `High` membership sets using triangular and trapezoidal functions.
   - Evaluates a Mamdani inference engine with 27 IF-THEN rules.
   - Computes crisp numerical outputs using centroid defuzzification:
     - `POLARITY`: ranges from -1.0 (strongly negative) to +1.0 (strongly positive).
     - `AMBIVALENCE`: ranges from 0.0 (unambiguous) to 1.0 (conflicting/mixed feelings).
3. **Natural Language Explanation:**
   - Fills pre-defined sentence templates (T1–T7) based on fuzzy rule activations.
   - Extracts key supporting words and handles negation (e.g., "not good").
   - Validates that readability achieves a Flesch Reading Ease score of $\ge 60$.
4. **Desktop User Interface:**
   - Provides a clean local desktop interface for single-sentence inputs with gauges/meters and explanation breakdowns.
   - Supports batch processing for CSV files containing up to 1,000 texts.

## 3. Scope Boundaries
- **In Scope:** Local offline execution, SST and TweetEval benchmark datasets, pre-trained model inference, rule-based explainability.
- **Out of Scope:** Model training or fine-tuning, multilingual support, aspect-based sentiment, sarcasm detectors, and web/cloud hosting.