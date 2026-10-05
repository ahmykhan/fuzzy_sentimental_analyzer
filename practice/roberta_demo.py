import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Model identifier (Cardiff NLP Twitter RoBERTa sentiment model)
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
save_directory = "models/roberta_sentiment"

print(f"Downloading and saving model to {save_directory}...")
os.makedirs(save_directory, exist_ok=True)

# 2. Download and save locally inside models/ (as required by Task 1.6)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

tokenizer.save_pretrained(save_directory)
model.save_pretrained(save_directory)
print("Model and tokenizer saved successfully.")

# 3. Same 10 sentences from Task 1.4
sentences = [
    "I absolutely loved this movie! The cast was amazing.",
    "The acting was brilliant, but the story was boring and too long.",
    "The food was not good.",
    "The service was fast and the employees were helpful.",
    "I hated the background score, it gave me a headache.",
    "It is not bad, but I would not recommend it either.",
    "The laptop arrived on time and works properly.",
    "Worst customer support experience ever.",
    "The camera is great, although battery life is disappointing.",
    "This was never going to work out anyway."
]

print("\n" + "=" * 95)
print(f"{'Sentence':<55} | {'Neg (p_neg)':<11} | {'Neu (p_neu)':<11} | {'Pos (p_pos)':<11}")
print("=" * 95)

# Label mapping for this model: 0 -> Negative, 1 -> Neutral, 2 -> Positive
model.eval()
with torch.no_grad():
    for text in sentences:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=1).squeeze().tolist()
        
        p_neg, p_neu, p_pos = probs[0], probs[1], probs[2]
        print(f"{text[:52]:<55} | {p_neg:<11.4f} | {p_neu:<11.4f} | {p_pos:<11.4f}")