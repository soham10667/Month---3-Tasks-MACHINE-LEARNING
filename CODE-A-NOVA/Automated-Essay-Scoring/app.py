"""
Automated Essay Scoring (AES) - Interactive Streamlit Application
Code-A-Nova Internship Project
"""

import os
import re
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from scipy.sparse import hstack

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="Automated Essay Scoring | Code-A-Nova",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for clean, professional software UI
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Clean, Professional Header */
    .main-header {
        background: #1e293b;
        padding: 1.75rem 2rem;
        border-radius: 12px;
        color: #ffffff;
        margin-bottom: 2rem;
        border: 1px solid #334155;
    }
    .main-header h1 {
        color: #f8fafc !important;
        font-weight: 700;
        font-size: 2rem;
        margin: 0 0 0.4rem 0;
        letter-spacing: -0.01em;
    }
    .main-header p {
        color: #94a3b8 !important;
        font-size: 0.95rem;
        margin: 0;
        max-width: 820px;
        line-height: 1.5;
    }

    /* Clean Metric Cards */
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 1.15rem 1.3rem;
        transition: border-color 0.15s ease, background-color 0.15s ease;
    }
    .metric-card:hover {
        border-color: #64748b;
        background: #243247;
    }
    .metric-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8 !important;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 1.75rem;
        font-weight: 700;
        color: #f8fafc !important;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #64748b !important;
        margin-top: 0.2rem;
    }

    /* Status Badges */
    .score-badge {
        display: inline-block;
        padding: 0.25rem 0.8rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-excellent { background-color: #14532d; color: #86efac; border: 1px solid #166534; }
    .badge-good { background-color: #1e3a8a; color: #93c5fd; border: 1px solid #1d4ed8; }
    .badge-average { background-color: #713f12; color: #fde047; border: 1px solid #854d0e; }
    .badge-poor { background-color: #7f1d1d; color: #fca5a5; border: 1px solid #991b1b; }

    .limitation-box {
        background-color: #1e293b;
        border-left: 4px solid #f59e0b;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin-top: 1.25rem;
        color: #cbd5e1;
        border-top: 1px solid #334155;
        border-bottom: 1px solid #334155;
        border-right: 1px solid #334155;
    }
    .limitation-box h4 {
        margin: 0 0 0.4rem 0;
        color: #f59e0b !important;
    }

    /* Solid Standard Primary Buttons */
    div.stButton > button {
        background: #2563eb !important;
        border: 1px solid #1d4ed8 !important;
        border-radius: 8px !important;
        padding: 0.45rem 1rem !important;
        transition: background-color 0.15s ease !important;
        box-shadow: none !important;
    }
    div.stButton > button p,
    div.stButton > button span,
    div.stButton > button div {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
    }
    div.stButton > button:hover {
        background: #1d4ed8 !important;
        border-color: #1e40af !important;
    }

    /* Clean Dark Sidebar */
    [data-testid="stSidebar"] {
        background: #0f172a !important;
        border-right: 1px solid #1e293b !important;
    }
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4 {
        color: #f8fafc !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
    }
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] label {
        color: #cbd5e1 !important;
        font-size: 0.9rem !important;
    }
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #64748b !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #1e293b !important;
    }

    /* Text Area & Labels */
    .stTextArea label p {
        color: #f8fafc !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    .stCaption, .stCaption p {
        color: #64748b !important;
    }

    /* Tabs Visibility */
    button[data-baseweb="tab"] p {
        color: #94a3b8 !important;
        font-weight: 500 !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #f8fafc !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Paths & Artifact Loading
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "essay_scoring_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "evaluation_metrics.json")
SAMPLES_PATH = os.path.join(BASE_DIR, "data", "sample_essays.json")


@st.cache_resource
def load_model_bundle():
    """Load trained ML pipeline artifacts."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics_and_samples():
    """Load benchmark metrics and sample essays."""
    metrics = None
    samples = {}
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            metrics = json.load(f)
    if os.path.exists(SAMPLES_PATH):
        with open(SAMPLES_PATH, "r", encoding="utf-8") as f:
            samples = json.load(f)
    return metrics, samples


# ---------------------------------------------------------
# Feature Extraction Routine
# ---------------------------------------------------------
def extract_single_essay_features(text: str):
    """Extract linguistic, readability, and structural features for input text."""
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    num_words = max(len(words), 1)
    
    sentences = [s.strip() for s in re.split(r"[\.\?!]+", text) if len(s.strip()) > 0]
    num_sentences = max(len(sentences), 1)
    
    paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 0]
    num_paragraphs = max(len(paragraphs), 1)
    
    char_count = len(text)
    avg_word_len = sum(len(w) for w in words) / num_words
    avg_sent_len = num_words / num_sentences
    lexical_diversity = len(set(words)) / num_words
    
    # Syllables estimation
    syllables = sum(max(1, len(re.findall(r"[aeiouy]+", w))) for w in words)
    
    # Flesch Reading Ease (0-100: Higher = Easier)
    flesch = 206.835 - (1.015 * avg_sent_len) - (84.6 * (syllables / num_words))
    flesch = max(0.0, min(100.0, flesch))
    
    # Automated Readability Index (ARI: Approx US grade level)
    ari = (4.71 * (char_count / num_words)) + (0.5 * avg_sent_len) - 21.43
    ari = max(1.0, min(18.0, ari))
    
    punct_count = len(re.findall(r"[,;:\(\)\"'\-]", text))
    emphasis_count = len(re.findall(r"[\?!]", text))
    
    feats = np.array([[
        float(num_words),
        float(char_count),
        float(num_sentences),
        float(num_paragraphs),
        float(avg_word_len),
        float(avg_sent_len),
        float(lexical_diversity),
        float(syllables),
        float(flesch),
        float(ari),
        float(punct_count),
        float(emphasis_count)
    ]])
    
    details = {
        "words": num_words,
        "chars": char_count,
        "sentences": num_sentences,
        "paragraphs": num_paragraphs,
        "avg_word_len": round(avg_word_len, 2),
        "avg_sent_len": round(avg_sent_len, 2),
        "lexical_diversity": round(lexical_diversity * 100, 1),
        "flesch": round(flesch, 1),
        "ari": round(ari, 1),
        "punct": punct_count
    }
    return feats, details


def get_grade_and_status(score: float):
    """Map numeric 1.0-10.0 score to letter grades and feedback status."""
    if score >= 8.5:
        return "A / Exemplary", "Proficient / Advanced", "badge-excellent", "#059669"
    elif score >= 7.0:
        return "B / Good", "Competent / Solid", "badge-good", "#0d9488"
    elif score >= 5.0:
        return "C / Fair", "Developing / Adequate", "badge-average", "#d97706"
    else:
        return "D-F / Needs Work", "Beginning / Basic", "badge-poor", "#e11d48"


# ---------------------------------------------------------
# Application Header
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>Automated Essay Scoring System</h1>
    <p>Machine Learning assessment evaluating essay content relevance, syntactic complexity, vocabulary richness, and readability using the Hewlett Foundation ASAP benchmark.</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar: Setup & Sample Loaders
# ---------------------------------------------------------
bundle = load_model_bundle()
metrics, samples = load_metrics_and_samples()

with st.sidebar:
    st.markdown("### ⚙️ Model Information")
    if bundle is not None:
        st.success(f"**Active Model**: {bundle.get('model_name', 'Gradient Boosting')}")
        st.caption("Trained on 12,970+ essays from ASAP-AES.")
    else:
        st.error("Model artifacts not found! Run `python train_model.py` first.")

    st.markdown("---")
    st.markdown("### 📝 Quick Sample Essays")
    st.caption("Click to load a benchmark essay with a known score:")

    if "essay_input" not in st.session_state:
        st.session_state["essay_input"] = ""

    for title, data in samples.items():
        if st.button(f"Load {title}", key=f"btn_{title}", use_container_width=True):
            st.session_state["essay_input"] = data["text"]
            st.session_state["active_sample_info"] = f"Loaded **{title}** (Expected Score: ~{data['actual_score']}/10)"

    if st.button("🧹 Clear Text Area", use_container_width=True):
        st.session_state["essay_input"] = ""
        st.session_state.pop("active_sample_info", None)

    st.markdown("---")
    st.markdown("### 📊 Benchmark Metrics")
    if metrics:
        best_name = metrics.get("best_model", "Gradient Boosting")
        best_stats = metrics["models_evaluated"].get(best_name, {})
        st.metric("Test MAE", f"{best_stats.get('MAE', 1.01):.3f}")
        st.metric("Test RMSE", f"{best_stats.get('RMSE', 1.31):.3f}")
        st.metric("Quadratic Weighted Kappa", f"{best_stats.get('QWK', 0.75):.3f}")
        st.caption(f"Evaluated on {metrics.get('test_samples', 2595):,} unseen test essays.")

# ---------------------------------------------------------
# Main Essay Input Section
# ---------------------------------------------------------
if "active_sample_info" in st.session_state:
    st.info(st.session_state["active_sample_info"])

user_essay = st.text_area(
    "Enter or paste the essay text below for analysis:",
    value=st.session_state["essay_input"],
    height=260,
    placeholder="Type or paste your complete essay here (minimum 25 words recommended for accurate scoring)..."
)

col_btn, col_info = st.columns([1, 4])
with col_btn:
    analyze_clicked = st.button("🚀 Score Essay", type="primary", use_container_width=True)

with col_info:
    raw_word_count = len(re.findall(r"\b[a-zA-Z]+\b", user_essay))
    st.caption(f"Current word count: **{raw_word_count} words** | Character count: **{len(user_essay)} characters**")

# ---------------------------------------------------------
# Scoring & Feedback Execution
# ---------------------------------------------------------
if analyze_clicked or (user_essay.strip() and "scored" in st.session_state):
    st.session_state["scored"] = True
    clean_text = user_essay.strip()
    
    if len(clean_text) < 30 or raw_word_count < 10:
        st.warning("⚠️ The submitted text is too short to produce a reliable evaluation. Please provide at least 25 words.")
    elif bundle is None:
        st.error("Model artifacts are not loaded. Please train the model using `python train_model.py`.")
    else:
        # Extract features
        num_feats, details = extract_single_essay_features(clean_text)
        
        # Scale numerical features
        num_scaled = bundle["scaler"].transform(num_feats)
        
        # Transform TF-IDF
        text_tfidf = bundle["tfidf"].transform([clean_text])
        
        # Combine
        full_features = hstack([num_scaled, text_tfidf])
        
        # Predict
        raw_pred = float(bundle["model"].predict(full_features)[0])
        score = float(np.clip(raw_pred, 1.0, 10.0))
        grade, level, badge_cls, badge_color = get_grade_and_status(score)
        
        st.markdown("---")
        st.markdown("## 🎯 Evaluation Results")
        
        # Top Score Display Card
        col_score, col_meter = st.columns([1, 2])
        
        with col_score:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center; border-top: 4px solid {badge_color};">
                <div class="metric-label">Predicted Essay Score</div>
                <div class="metric-value" style="font-size: 3rem; color: {badge_color};">{score:.1f} <span style="font-size: 1.5rem; color: #94a3b8;">/ 10</span></div>
                <div style="margin-top: 0.6rem;">
                    <span class="score-badge {badge_cls}">{grade}</span>
                </div>
                <div class="metric-sub" style="margin-top: 0.5rem; font-weight: 600;">{level}</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_meter:
            st.markdown("### Score Distribution & Interpretation")
            progress_pct = int(((score - 1.0) / 9.0) * 100)
            st.progress(progress_pct / 100.0)
            
            st.markdown(f"""
            * **Scale**: Standardized from 1.0 (lowest) to 10.0 (highest).
            * **Score Rating**: This essay achieved a **{score:.2f}**, situating it within the **{grade}** range.
            * **Model Confidence**: Evaluated using our best regression estimator with an average holdout error of ±1.01 points.
            """)
            
        # Detailed Linguistic Feature Cards
        st.markdown("### 📊 Extracted Linguistic & Writing Dimensions")
        m1, m2, m3, m4 = st.columns(4)
        
        with m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Volume & Structure</div>
                <div class="metric-value">{details['words']}</div>
                <div class="metric-sub">{details['sentences']} sentences, {details['paragraphs']} paragraphs</div>
            </div>
            """, unsafe_allow_html=True)
            
        with m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Sentence Length</div>
                <div class="metric-value">{details['avg_sent_len']}</div>
                <div class="metric-sub">avg words per sentence</div>
            </div>
            """, unsafe_allow_html=True)

        with m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Lexical Diversity</div>
                <div class="metric-value">{details['lexical_diversity']}%</div>
                <div class="metric-sub">unique words (Type-Token Ratio)</div>
            </div>
            """, unsafe_allow_html=True)

        with m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Flesch Reading Ease</div>
                <div class="metric-value">{details['flesch']}</div>
                <div class="metric-sub">ARI Grade Level: {details['ari']}</div>
            </div>
            """, unsafe_allow_html=True)

        # Actionable Writing Diagnostic Feedback
        st.markdown("### 💡 Writing Diagnostic Observations")
        obs_col1, obs_col2 = st.columns(2)
        
        with obs_col1:
            if details["words"] < 150:
                st.warning("⚠️ **Length**: The essay is relatively brief. Expanding upon core arguments and evidence will bolster the depth.")
            elif details["words"] > 450:
                st.success("✅ **Length**: Excellent elaboration and comprehensive development of ideas.")
            else:
                st.info("ℹ️ **Length**: Balanced essay length suitable for general analytical prompts.")
                
            if details["lexical_diversity"] < 40.0:
                st.warning("⚠️ **Vocabulary**: High degree of word repetition. Consider introducing richer synonyms.")
            else:
                st.success("✅ **Vocabulary**: Strong lexical richness with varied vocabulary choice.")

        with obs_col2:
            if details["avg_sent_len"] > 26.0:
                st.warning("⚠️ **Syntax**: Average sentence length is high (>26 words). Look for potential run-on sentences.")
            elif details["avg_sent_len"] < 10.0:
                st.warning("⚠️ **Syntax**: Sentences are very short. Combining ideas can improve narrative flow.")
            else:
                st.success("✅ **Syntax**: Well-balanced sentence structure and readability.")

            if details["flesch"] < 40.0:
                st.info("ℹ️ **Readability**: Dense academic prose. Ensure readability remains accessible.")
            elif details["flesch"] > 75.0:
                st.info("ℹ️ **Readability**: Highly conversational and easy to read.")
            else:
                st.success("✅ **Readability**: Standard educational reading level.")

# ---------------------------------------------------------
# Technical Tabs: Limitations & Benchmark
# ---------------------------------------------------------
st.markdown("---")
tab_limit, tab_eval, tab_dataset = st.tabs(["⚠️ System Transparency & Limitations", "📈 Benchmark & Models", "📚 Dataset Information"])

with tab_limit:
    st.markdown("""
    ### Honest Model Limitations & Ethical Disclosure
    As required by academic standards and internship guidelines, users and evaluators should note:
    
    1. **Not a Full Grammar Checker**:
       - This system evaluates **statistical surface features** (punctuation density, word/sentence lengths, vocabulary richness, readability indices) and **TF-IDF lexical patterns**.
       - It does **not** integrate a complete rule-based syntax dependency parser (such as LanguageTool or Grammarly) and should not be confused with one.
    
    2. **Content Semantics & Factual Accuracy**:
       - The regression model identifies thematic vocabulary relevance via n-gram TF-IDF representations.
       - It cannot verify factual truthfulness, mathematical validity, or logical fallacies within arguments.
    
    3. **Length Sensitivity**:
       - Machine learning models trained on essay scoring benchmarks exhibit an inherent positive correlation with essay length. A very long essay with repetitive statements may receive a slightly higher score than an exceptionally concise one.
    
    4. **Intended Usage**:
       - Designed as an **educational assistance tool** for practice and formative feedback. It must not be used for high-stakes standardized academic decisions without human review.
    """)

with tab_eval:
    if metrics:
        st.markdown("### Model Comparison on 20% Unseen Test Split")
        eval_df = pd.DataFrame(metrics["models_evaluated"]).T
        eval_df.rename(columns={
            "MAE": "Mean Absolute Error (MAE)",
            "RMSE": "Root Mean Squared Error (RMSE)",
            "R2_Score": "R² Score",
            "QWK": "Quadratic Weighted Kappa (QWK)"
        }, inplace=True)
        
        # High-contrast dark styling for best metric values
        styled_df = (
            eval_df.style
            .format("{:.4f}")
            .highlight_min(
                subset=["Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)"],
                props="background-color: #14532d; color: #86efac; font-weight: 700;"
            )
            .highlight_max(
                subset=["R² Score", "Quadratic Weighted Kappa (QWK)"],
                props="background-color: #14532d; color: #86efac; font-weight: 700;"
            )
        )
        st.dataframe(styled_df, use_container_width=True)
        st.caption(f"Winner model: **{metrics.get('best_model')}** with QWK score of **{metrics['models_evaluated'].get(metrics.get('best_model'), {}).get('QWK'):.4f}** (indicates strong agreement with human expert raters).")
    else:
        st.info("Run `python train_model.py` to generate the evaluation benchmark.")

with tab_dataset:
    st.markdown("""
    ### Dataset: The Hewlett Foundation ASAP-AES Benchmark
    * **Sponsor**: William and Flora Hewlett Foundation (ASAP Competition).
    * **Total Corpus**: 12,976 student essays scored by human expert raters.
    * **Prompt Types**: Persuasive, Expository, and Response-to-Literature essays.
    * **Score Normalization**: Prompt-specific scores (ranging from 0-3 up to 10-60) are standardized to a continuous 1.0 - 10.0 scale for uniform evaluation.
    """)
