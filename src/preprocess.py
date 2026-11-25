import re
# Checkpoint 3 (CP03) — Task 3.1: Clean-up functions (for_vader, for_roberta, for_explanation)
import re

CONTRACTIONS = {
    r"\bcan't\b": "cannot",
    r"\bwon't\b": "will not",
    r"\bshan't\b": "shall not",
    r"\bdon't\b": "do not",
    r"\bdoesn't\b": "does not",
    r"\bdidn't\b": "did not",
    r"\bisn't\b": "is not",
    r"\baren't\b": "are not",
    r"\bwasn't\b": "was not",
    r"\bweren't\b": "were not",
    r"\bhaven't\b": "have not",
    r"\bhasn't\b": "has not",
    r"\bhadn't\b": "had not",
    r"\bcouldn't\b": "could not",
    r"\bshouldn't\b": "should not",
    r"\bwouldn't\b": "would not",
    r"\bi'm\b": "I am",
    r"\bit's\b": "it is",
    r"\bthat's\b": "that is",
    r"\bthere's\b": "there is",
    r"\bwe're\b": "we are",
    r"\bthey're\b": "they are",
    r"\byou're\b": "you are",
    r"\bi've\b": "I have",
    r"\bwe've\b": "we have",
    r"\bthey've\b": "they have",
    r"\bi'll\b": "I will",
    r"\bwe'll\b": "we will",
    r"\bthey'll\b": "they will",
    r"\blet's\b": "let us",
}

def clean_brackets(text: str) -> str:
    text = text.replace("-LRB-", "(").replace("-RRB-", ")")
    text = text.replace("-LCB-", "{").replace("-RCB-", "}")
    text = text.replace("-LSB-", "[").replace("-RSB-", "]")
    return text


def for_vader(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = clean_brackets(text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def for_roberta(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = clean_brackets(text)
    text = re.sub(r"https?://\S+|www\.\S+", "http", text)
    text = re.sub(r"@\w+", "@user", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def for_explanation(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = clean_brackets(text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    for pattern, replacement in CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text