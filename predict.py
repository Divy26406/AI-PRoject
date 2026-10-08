import os
from typing import Dict, Any, Optional
import joblib
from preprocess import combine_and_clean

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "spam_model.pkl")
LOGREG_MODEL_PATH = os.path.join(MODEL_DIR, "logistic_regression_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

_CACHED_MODELS = {}
_CACHED_VECTORIZER = None


def load_artifacts(model_type: str = "naive_bayes"):
    global _CACHED_MODELS, _CACHED_VECTORIZER

    if _CACHED_VECTORIZER is None:
        if not os.path.exists(VECTORIZER_PATH):
            raise FileNotFoundError(
                f"TF-IDF vectorizer not found at {VECTORIZER_PATH}. "
                "Please run train_model.py first."
            )
        _CACHED_VECTORIZER = joblib.load(VECTORIZER_PATH)

    if model_type not in _CACHED_MODELS:
        path = LOGREG_MODEL_PATH if model_type == "logistic_regression" else DEFAULT_MODEL_PATH
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found at {path}. Please run train_model.py first."
            )
        _CACHED_MODELS[model_type] = joblib.load(path)

    return _CACHED_MODELS[model_type], _CACHED_VECTORIZER


def predict_email(
    message: str,
    subject: Optional[str] = "",
    model_type: str = "naive_bayes"
) -> Dict[str, Any]:
    model, vectorizer = load_artifacts(model_type=model_type)

    cleaned_text = combine_and_clean(subject, message)

    if not cleaned_text.strip():
        return {
            "label": "Not Spam",
            "is_spam": False,
            "confidence": 50.0,
            "spam_probability": 0.5,
            "ham_probability": 0.5,
            "cleaned_text": "",
            "warning": "Input contained no discriminative words after preprocessing."
        }

    tfidf_vec = vectorizer.transform([cleaned_text])

    probs = model.predict_proba(tfidf_vec)[0]
    ham_prob = float(probs[0])
    spam_prob = float(probs[1])

    is_spam = spam_prob >= 0.5
    label = "Spam" if is_spam else "Not Spam"
    confidence = (spam_prob if is_spam else ham_prob) * 100.0

    return {
        "label": label,
        "is_spam": is_spam,
        "confidence": round(confidence, 2),
        "spam_probability": round(spam_prob, 4),
        "ham_probability": round(ham_prob, 4),
        "cleaned_text": cleaned_text
    }


if __name__ == "__main__":
    test_msg = "Congratulations! You won a $1,000,000 cash prize! Claim your reward now."
    res = predict_email(test_msg, subject="Prize Winner Notice")
    print(res)
