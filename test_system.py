import os
from preprocess import clean_text, combine_and_clean
from predict import predict_email
from explain import get_top_terms_by_class

def run_tests():
    print("=" * 60)
    print("RUNNING AUTOMATED VERIFICATION TESTS")
    print("=" * 60)

    print("\n[Test 1] NLP Preprocessing Pipeline:")
    raw_sample = "CONGRATULATIONS!!! You WON a FREE Prize of $1,000! Visit http://spam.net/win"
    cleaned = clean_text(raw_sample)
    print(f"  Input:   {raw_sample}")
    print(f"  Cleaned: {cleaned}")
    assert "congratulations" in cleaned
    assert "prize" in cleaned
    assert "http" not in cleaned
    print("  -> PASSED")

    print("\n[Test 2] Spam Email Prediction:")
    spam_sample = (
        "Congratulations! You are the winner of a free cash prize. "
        "Click here to claim your reward now."
    )
    res_spam = predict_email(spam_sample, subject="Winner Notice")
    print(f"  Label: {res_spam['label']}, Confidence: {res_spam['confidence']}%")
    assert res_spam["is_spam"] is True
    assert res_spam["confidence"] > 80.0
    print("  -> PASSED (Correctly flagged as Spam)")

    print("\n[Test 3] Ham (Legitimate) Email Prediction:")
    ham_sample = (
        "Dear students, your project presentation is scheduled for Monday at 10 AM. "
        "Please be present on time."
    )
    res_ham = predict_email(ham_sample, subject="Presentation Schedule")
    print(f"  Label: {res_ham['label']}, Confidence: {res_ham['confidence']}%")
    assert res_ham["is_spam"] is False
    assert res_ham["confidence"] > 80.0
    print("  -> PASSED (Correctly flagged as Not Spam)")

    print("\n[Test 4] Explainability Term Extraction:")
    spam_terms = get_top_terms_by_class(spam_sample, subject="Winner Notice", predicted_is_spam=True)
    print(f"  Top terms for Spam: {[t['term'] for t in spam_terms]}")
    assert len(spam_terms) > 0
    assert any(w in [t['term'] for t in spam_terms] for w in ["winner", "prize", "cash prize", "claim", "reward", "free"])

    ham_terms = get_top_terms_by_class(ham_sample, subject="Presentation Schedule", predicted_is_spam=False)
    print(f"  Top terms for Ham:  {[t['term'] for t in ham_terms]}")
    assert len(ham_terms) > 0
    assert any(w in [t['term'] for t in ham_terms] for w in ["presentation", "scheduled", "students", "project", "monday"])
    print("  -> PASSED")

    print("\n[Test 5] Sample .txt Files Verification:")
    samples_dir = os.path.join(os.path.dirname(__file__), "sample_emails")
    sample_files = os.listdir(samples_dir)
    print(f"  Found sample files: {sample_files}")
    assert len(sample_files) >= 4
    for sf in sample_files:
        path = os.path.join(samples_dir, sf)
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
            assert len(content.strip()) > 20
    print("  -> PASSED")

    print("\n[Test 6] Model Metrics File Verification:")
    metrics_path = os.path.join(os.path.dirname(__file__), "model", "model_metrics.json")
    assert os.path.exists(metrics_path)
    import json
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    print(f"  Accuracy (Naive Bayes):       {metrics['models']['Multinomial Naive Bayes']['accuracy']*100:.2f}%")
    print(f"  Accuracy (Logistic Regression): {metrics['models']['Logistic Regression']['accuracy']*100:.2f}%")
    assert metrics['models']['Multinomial Naive Bayes']['accuracy'] > 0.95
    print("  -> PASSED")

    print("\n" + "=" * 60)
    print("ALL 6 TESTS PASSED SUCCESSFULLY! SYSTEM IS FULLY OPERATIONAL.")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
