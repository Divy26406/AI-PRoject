import os
import json
import urllib.request
import zipfile
import time
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from preprocess import combine_and_clean

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
CSV_PATH = os.path.join(DATA_DIR, "enron_spam_data.csv")
ZIP_PATH = os.path.join(DATA_DIR, "enron_spam_data.zip")

DATASET_URL = (
    "https://raw.githubusercontent.com/MWiechmann/enron_spam_data/master/enron_spam_data.zip"
)


def ensure_dataset() -> str:
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(CSV_PATH):
        print(f"[Dataset] Found existing CSV at: {CSV_PATH}")
        return CSV_PATH

    print(f"[Dataset] Downloading Enron-Spam dataset from {DATASET_URL}...")
    urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)
    print("[Dataset] Extracting dataset archive...")
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall(DATA_DIR)
    print(f"[Dataset] Extracted successfully to: {CSV_PATH}")
    return CSV_PATH


def load_and_preprocess_data(csv_path: str):
    print("[1/5] Loading Enron-Spam dataset...")
    df = pd.read_csv(csv_path)
    total_raw = len(df)
    print(f"      Total records loaded: {total_raw:,}")
    print(f"      Columns: {df.columns.tolist()}")

    df["Subject"] = df["Subject"].fillna("")
    df["Message"] = df["Message"].fillna("")

    df["label"] = df["Spam/Ham"].str.lower().map({"spam": 1, "ham": 0})
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    print("[2/5] Cleaning and preprocessing email text with NLP pipeline...")
    start_time = time.time()

    cleaned_texts = [
        combine_and_clean(sub, msg)
        for sub, msg in zip(df["Subject"], df["Message"])
    ]
    df["cleaned_text"] = cleaned_texts

    initial_count = len(df)
    df = df[df["cleaned_text"].str.strip() != ""].reset_index(drop=True)
    dropped_empty = initial_count - len(df)
    elapsed = time.time() - start_time
    print(f"      NLP preprocessing finished in {elapsed:.1f}s")
    if dropped_empty > 0:
        print(f"      Dropped {dropped_empty} empty emails after cleaning.")

    spam_count = int((df["label"] == 1).sum())
    ham_count = int((df["label"] == 0).sum())
    print(f"      Dataset distribution: {spam_count:,} Spam ({spam_count/len(df)*100:.1f}%), "
          f"{ham_count:,} Ham ({ham_count/len(df)*100:.1f}%)")

    return df


def train_and_evaluate():
    os.makedirs(MODEL_DIR, exist_ok=True)
    csv_path = ensure_dataset()
    df = load_and_preprocess_data(csv_path)

    X = df["cleaned_text"].values
    y = df["label"].values

    print("\n[3/5] Splitting data into 80% Train and 20% Test (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"      Training set size: {len(X_train):,}")
    print(f"      Testing set size:  {len(X_test):,}")

    print("\n[4/5] Extracting TF-IDF Features (unigrams + bigrams, max_features=10,000)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
        min_df=2
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"      TF-IDF matrix shape: {X_train_tfidf.shape}")

    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    }

    metrics_results = {}
    fitted_models = {}

    print("\n[5/5] Training and Evaluating Models...")
    print("=" * 65)

    for name, clf in models.items():
        print(f"\n---> Training: {name}...")
        t0 = time.time()
        clf.fit(X_train_tfidf, y_train)
        train_duration = time.time() - t0

        y_pred = clf.predict(X_test_tfidf)
        y_prob = clf.predict_proba(X_test_tfidf)[:, 1] if hasattr(clf, "predict_proba") else None

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, pos_label=1))
        rec = float(recall_score(y_test, y_pred, pos_label=1))
        f1 = float(f1_score(y_test, y_pred, pos_label=1))
        cm = confusion_matrix(y_test, y_pred).tolist()

        tn, fp, fn, tp = int(cm[0][0]), int(cm[0][1]), int(cm[1][0]), int(cm[1][1])

        print(f"     Trained in: {train_duration:.2f}s")
        print(f"     Accuracy :  {acc * 100:.2f}%")
        print(f"     Precision:  {prec * 100:.2f}%")
        print(f"     Recall   :  {rec * 100:.2f}%")
        print(f"     F1-Score :  {f1 * 100:.2f}%")
        print(f"     Confusion Matrix: TP={tp}, FP={fp}, TN={tn}, FN={fn}")

        metrics_results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "confusion_matrix": {
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "matrix": cm
            },
            "train_duration_sec": round(train_duration, 2)
        }
        fitted_models[name] = clf

    print("\n" + "=" * 65)

    primary_model_name = "Multinomial Naive Bayes"
    best_clf = fitted_models[primary_model_name]

    model_save_path = os.path.join(MODEL_DIR, "spam_model.pkl")
    vectorizer_save_path = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
    logreg_save_path = os.path.join(MODEL_DIR, "logistic_regression_model.pkl")
    metrics_save_path = os.path.join(MODEL_DIR, "model_metrics.json")

    print(f"\n[Saving Artifacts]")
    print(f"Saving primary classifier ({primary_model_name}) to: {model_save_path}")
    joblib.dump(best_clf, model_save_path)

    print(f"Saving Logistic Regression classifier to: {logreg_save_path}")
    joblib.dump(fitted_models["Logistic Regression"], logreg_save_path)

    print(f"Saving TF-IDF vectorizer to: {vectorizer_save_path}")
    joblib.dump(vectorizer, vectorizer_save_path)

    overall_summary = {
        "dataset_name": "Enron-Spam Email Dataset",
        "dataset_source": "https://github.com/MWiechmann/enron_spam_data",
        "total_emails": int(len(df)),
        "spam_emails": int((df["label"] == 1).sum()),
        "ham_emails": int((df["label"] == 0).sum()),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "vocabulary_size": int(len(vectorizer.vocabulary_)),
        "primary_model": primary_model_name,
        "models": metrics_results,
        "trained_timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=4)
    print(f"Saved evaluation metrics to: {metrics_save_path}")

    print("\n Training and evaluation completed successfully!")
    return overall_summary


if __name__ == "__main__":
    train_and_evaluate()
