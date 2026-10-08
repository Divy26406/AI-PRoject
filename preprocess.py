import re
import html
from typing import Optional, List

STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just",
    "don", "should", "now", "d", "ll", "m", "o", "re", "ve", "y", "ain", "aren",
    "couldn", "didn", "doesn", "hadn", "hasn", "haven", "isn", "ma", "mightn",
    "mustn", "needn", "shan", "shouldn", "wasn", "weren", "won", "wouldn"
}

HTML_TAG_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
NON_ALPHA_RE = re.compile(r"[^a-zA-Z\s]")
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: Optional[str]) -> str:
    if text is None:
        return ""

    text = str(text)
    text = html.unescape(text)
    text = HTML_TAG_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    text = EMAIL_RE.sub(" ", text)
    text = text.lower()
    text = NON_ALPHA_RE.sub(" ", text)
    tokens = text.split()
    filtered_tokens: List[str] = [
        token for token in tokens
        if token not in STOPWORDS and len(token) > 1
    ]

    return " ".join(filtered_tokens)


def combine_and_clean(subject: Optional[str] = "", message: Optional[str] = "") -> str:
    sub = "" if subject is None else str(subject)
    msg = "" if message is None else str(message)
    combined = f"{sub} {msg}".strip()
    return clean_text(combined)


if __name__ == "__main__":
    sample = "CONGRATULATIONS!!! You WON a FREE Cash Prize of $1,000,000! Click http://spam.xyz/now"
    cleaned = clean_text(sample)
    print("Original:", sample)
    print("Cleaned: ", cleaned)
