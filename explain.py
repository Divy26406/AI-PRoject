from typing import List, Dict, Any, Tuple
import numpy as np
from predict import load_artifacts
from preprocess import combine_and_clean


def get_feature_explanations(
    message: str,
    subject: str = "",
    model_type: str = "naive_bayes",
    top_n: int = 8
) -> List[Dict[str, Any]]:
    model, vectorizer = load_artifacts(model_type=model_type)
    cleaned_text = combine_and_clean(subject, message)

    if not cleaned_text.strip():
        return []

    tfidf_vec = vectorizer.transform([cleaned_text])
    feature_names = np.array(vectorizer.get_feature_names_out())

    nonzero_indices = tfidf_vec.nonzero()[1]
    if len(nonzero_indices) == 0:
        return []

    if hasattr(model, "feature_log_prob_"):
        log_ratios = model.feature_log_prob_[1] - model.feature_log_prob_[0]
        term_weights = log_ratios
    elif hasattr(model, "coef_"):
        term_weights = model.coef_[0]
    else:
        return []

    tfidf_values = tfidf_vec.toarray()[0]
    contributions = []

    for idx in nonzero_indices:
        term = feature_names[idx]
        tfidf_val = tfidf_values[idx]
        weight = term_weights[idx]
        impact = tfidf_val * weight

        direction = "Spam" if impact > 0 else "Ham"
        contributions.append({
            "term": term,
            "score": round(float(impact), 4),
            "abs_score": round(float(abs(impact)), 4),
            "direction": direction,
            "tfidf": round(float(tfidf_val), 4)
        })

    contributions.sort(key=lambda x: x["abs_score"], reverse=True)

    return contributions[:top_n]


def get_top_terms_by_class(
    message: str,
    subject: str = "",
    model_type: str = "naive_bayes",
    predicted_is_spam: bool = True,
    top_n: int = 6
) -> List[Dict[str, Any]]:
    all_explanations = get_feature_explanations(
        message, subject, model_type=model_type, top_n=50
    )

    target_direction = "Spam" if predicted_is_spam else "Ham"
    filtered = [e for e in all_explanations if e["direction"] == target_direction]

    if not filtered:
        filtered = all_explanations

    return filtered[:top_n]


if __name__ == "__main__":
    sample = (
        "Congratulations! You are the winner of a free cash prize. "
        "Click here to claim your reward now."
    )
    terms = get_top_terms_by_class(sample, subject="Winner Notice", predicted_is_spam=True)
    print("Top Spam Explanatory Terms:")
    for t in terms:
        print(f" - {t['term']}: score={t['score']} ({t['direction']})")
