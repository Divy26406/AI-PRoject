# 🛡️ Spam Email Detection System Using NLP & Machine Learning

An end-to-end Machine Learning and Natural Language Processing (NLP) system that analyzes email text and accurately classifies it as **Spam** or **Not Spam (Ham)**, complete with confidence scoring, model explainability, and an interactive Streamlit web dashboard.

Trained and evaluated on the benchmark **Enron-Spam Dataset** containing **33,716 real-world emails**.

---

## 📌 Project Overview

- **Primary Goal**: Automatically identify spam and phishing emails before they can harm users or clutter inboxes.
- **Dataset**: Enron-Spam Email Dataset (17,109 Spam, 16,545 Ham).
- **Core Pipeline**: Text Preprocessing ➔ NLP Tokenization & Cleaning ➔ TF-IDF Feature Extraction ➔ ML Classification (Multinomial Naive Bayes & Logistic Regression) ➔ Evaluation ➔ Explainability ➔ Streamlit Dashboard.
- **Explainability**: Mathematically computes and highlights the exact words/features in each email that contributed most strongly to the prediction.

---

## 🏗️ Project Architecture & File Structure

```text
Spam_Email_Detection/
│
├── data/
│   ├── enron_spam_data.csv          # 33,716 labeled Enron emails
│   └── enron_spam_data.zip          # Compressed source archive
│
├── model/
│   ├── spam_model.pkl               # Primary trained Multinomial Naive Bayes classifier
│   ├── logistic_regression_model.pkl# Trained Logistic Regression classifier
│   ├── tfidf_vectorizer.pkl         # Trained TF-IDF vectorizer (10,000 features)
│   └── model_metrics.json           # Evaluation metrics on holdout test set
│
├── sample_emails/                   # Sample .txt test files for professor demonstration
│   ├── sample_spam_lottery.txt      # Lottery prize scam example
│   ├── sample_spam_phishing.txt     # Account suspension phishing example
│   ├── sample_ham_meeting.txt       # Team sprint meeting example
│   └── sample_ham_project.txt       # University project submission example
│
├── preprocess.py                    # NLP text cleaning, HTML stripping, tokenization
├── train_model.py                   # Data loading, training, evaluation & serialization
├── predict.py                       # Inference pipeline for new email predictions
├── explain.py                       # Mathematical explainability & top influential terms
├── app.py                           # Full-featured Streamlit web application
├── requirements.txt                 # Project dependencies
└── README.md                        # Documentation and guide
```

---

## 📊 Model Performance on Unseen Test Data (6,731 Emails)

The dataset was split into an **80% Training set (26,923 emails)** and a **20% Holdout Testing set (6,731 emails)** using stratified sampling.

| Metric | Multinomial Naive Bayes (Primary) | Logistic Regression (Comparison) |
|---|---|---|
| **Accuracy** | **98.66%** | **99.08%** |
| **Precision** | **98.46%** | **98.58%** |
| **Recall** | **98.92%** | **99.62%** |
| **F1-Score** | **98.69%** | **99.10%** |
| **True Positives (TP)** | 3,385 | 3,409 |
| **True Negatives (TN)** | 3,256 | 3,260 |
| **False Positives (FP)** | 53 | 49 |
| **False Negatives (FN)** | 37 | 13 |
| **Training Speed** | 0.01 seconds | 0.12 seconds |

---

## 🚀 How to Run the Project

### 1. Prerequisites
Make sure Python 3.10+ is installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Re-train the Models
The trained model files are already generated and saved in `model/`. If you wish to retrain from scratch:
```bash
python train_model.py
```

### 4. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Or with specific Python interpreter:
```bash
py -3.13 -m streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🖥️ Using the Streamlit Interface

1. **Input Modes**:
   - **Type / Paste Email**: Type or paste any custom email subject and message.
   - **Upload TXT File**: Upload any `.txt` email file.
2. **Quick Samples**:
   - In the sidebar, select from 4 pre-loaded test samples (Spam Lottery, Phishing, Ham Meeting, ML Project) to test in 1 click.
3. **Model Selection**:
   - Switch between **Multinomial Naive Bayes** (primary) and **Logistic Regression** directly from the sidebar.
4. **Prediction & Explainability Output**:
   - 🔴 **SPAM** or 🟢 **NOT SPAM (HAM)** badge.
   - Predicted **Confidence Score** and probability breakdown.
   - **Explainability**: Important words that triggered the prediction with visual importance chart.
5. **Tabs**:
   - **Tab 1: Email Classifier & Explainability**: Interactive classification interface.
   - **Tab 2: Model Evaluation & Benchmarks**: Real holdout evaluation metrics and confusion matrices.
   - **Tab 3: Architecture & Summary**: Detailed pipeline explanation and presentation summary for professors.

---

## 🎓 Short Explanation for Professor

> *“Our project is a Spam Email Detection system using NLP and Machine Learning. We use the Enron-Spam Email Dataset containing thousands of labeled spam and legitimate emails. First, we preprocess the email text using NLP techniques and convert the text into numerical features using TF-IDF. We then train a machine-learning classification model and evaluate it using unseen testing data. After training, we save the model and TF-IDF vectorizer. Through our Streamlit application, a user can enter or upload a new email. The email goes through the same NLP and TF-IDF process and is then classified by the trained model as Spam or Not Spam. The application also displays the prediction confidence and important model-derived terms to provide an explanation of the result.”*
