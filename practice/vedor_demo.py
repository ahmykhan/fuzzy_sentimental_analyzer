from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

analyzer = SentimentIntensityAnalyzer()

# 10 test sentences covering positive, negative, mixed, and negated cases
sentences = [
    "I absolutely loved this movie! The cast was amazing.",              # Positive
    "The acting was brilliant, but the story was boring and too long.",  # Mixed
    "The food was not good.",                                            # Negated
    "The service was fast and the employees were helpful.",              # Positive
    "I hated the background score, it gave me a headache.",              # Negative
    "It is not bad, but I would not recommend it either.",               # Mixed / Negated
    "The laptop arrived on time and works properly.",                    # Neutral / Mild positive
    "Worst customer support experience ever.",                           # Negative
    "The camera is great, although battery life is disappointing.",      # Mixed
    "This was never going to work out anyway."                           # Negated / Negative
]

print(f"{'Sentence':<60} | {'Pos':<5} | {'Neg':<5} | {'Compound':<8}")
print("-" * 85)

for text in sentences:
    scores = analyzer.polarity_scores(text)
    print(f"{text[:58]:<60} | {scores['pos']:<5.2f} | {scores['neg']:<5.2f} | {scores['compound']:<8.2f}")