import os
import streamlit as st
from predict import predict_email
from explain import get_top_terms_by_class

st.set_page_config(
    page_title="Spam Email Detection",
    page_icon="📧",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 680px;
    }
    .header-title {
        text-align: center;
        font-size: 2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .header-subtitle {
        text-align: center;
        font-size: 0.95rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .analysis-container {
        background-color: #FFFFFF;
        border: 2px solid #E2E8F0;
        border-radius: 12px;
        padding: 24px 28px;
        margin-top: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace, sans-serif;
    }
    .analysis-title {
        text-align: center;
        font-size: 1.25rem;
        font-weight: 700;
        letter-spacing: 2px;
        color: #334155;
        border-top: 2px solid #E2E8F0;
        border-bottom: 2px solid #E2E8F0;
        padding: 8px 0;
        margin-bottom: 20px;
    }
    .prediction-spam {
        font-size: 1.4rem;
        font-weight: 800;
        color: #DC2626;
        margin-bottom: 12px;
    }
    .prediction-ham {
        font-size: 1.4rem;
        font-weight: 800;
        color: #16A34A;
        margin-bottom: 12px;
    }
    .confidence-text {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1E293B;
        margin-bottom: 16px;
    }
    .terms-heading {
        font-size: 1.1rem;
        font-weight: 700;
        color: #334155;
        margin-top: 16px;
        margin-bottom: 8px;
    }
    .term-item {
        font-size: 1rem;
        color: #475569;
        margin-left: 8px;
        line-height: 1.6;
    }
    .divider-bottom {
        border-bottom: 2px solid #E2E8F0;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="header-title">Spam Email Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">Enter an email below or upload a .txt file for instant classification</div>', unsafe_allow_html=True)

DEFAULT_EMAIL = (
    "Congratulations! You are the winner of a free cash prize. "
    "Click here to claim your reward now."
)

input_mode = st.radio(
    "Choose Input Method:",
    ["Type / Paste Email", "Upload .txt File"],
    horizontal=True,
    label_visibility="collapsed"
)

email_text = ""

if input_mode == "Type / Paste Email":
    email_text = st.text_area(
        label="Email Content:",
        value=DEFAULT_EMAIL,
        height=140,
        placeholder="Type or paste email content here..."
    )
else:
    uploaded_file = st.file_uploader("Upload an email (.txt file):", type=["txt"])
    if uploaded_file is not None:
        email_text = uploaded_file.read().decode("utf-8", errors="ignore")
        st.caption(f"Loaded `{uploaded_file.name}` ({len(email_text)} characters)")
    else:
        email_text = DEFAULT_EMAIL

col_btn, _ = st.columns([1, 2])
with col_btn:
    check_btn = st.button("CHECK EMAIL", type="primary", use_container_width=True)

if email_text.strip():
    result = predict_email(message=email_text)
    is_spam = result["is_spam"]
    conf = result["confidence"]

    top_terms = get_top_terms_by_class(
        message=email_text,
        predicted_is_spam=is_spam,
        top_n=6
    )

    pred_html = (
        '<div class="prediction-spam">Prediction : 🔴 SPAM</div>'
        if is_spam else
        '<div class="prediction-ham">Prediction : 🟢 NOT SPAM</div>'
    )

    terms_html = ""
    for t in top_terms:
        terms_html += f'<div class="term-item">• {t["term"]}</div>'

    if not terms_html:
        terms_html = '<div class="term-item">• (no common model keywords found)</div>'

    output_card = f"""
    <div class="analysis-container">
        <div class="analysis-title">EMAIL ANALYSIS</div>
        {pred_html}
        <div class="confidence-text">Confidence : {conf:.2f}%</div>
        <div class="terms-heading">Important Terms:</div>
        {terms_html}
        <div class="divider-bottom"></div>
    </div>
    """

    st.markdown(output_card, unsafe_allow_html=True)
else:
    st.warning("Please enter or upload an email to analyze.")
