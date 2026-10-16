"""
Automated Essay Scoring (AES) - Model Training Pipeline
Code-A-Nova Internship Project

This script handles:
1. Dataset acquisition (Hewlett Foundation ASAP Essay Scoring Benchmark).
2. Data preprocessing and score normalization to a standardized 1-10 scale.
3. Feature engineering:
   - Surface text statistics (word count, sentence count, char count, avg word length).
   - Lexical diversity (Type-Token Ratio).
   - Readability indices (Flesch Reading Ease & Automated Readability Index).
   - Punctuation & syntactic complexity proxies.
   - TF-IDF content representations.
4. Model comparison: Ridge Regression vs Random Forest vs Gradient Boosting.
5. Model evaluation (MAE, RMSE, R2, Quadratic Weighted Kappa).
6. Artifact persistence (model, preprocessors, evaluation metrics, and sample essays).
"""

import os
import re
import json
import requests
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, cohen_kappa_score
from scipy.sparse import hstack

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
ASAP_DATA_URL = "https://raw.githubusercontent.com/Turanga1/Automated-Essay-Scoring/master/training_set_rel3.tsv"
LOCAL_RAW_PATH = os.path.join(DATA_DIR, "training_set_rel3.tsv")
CLEAN_DATA_PATH = os.path.join(DATA_DIR, "essays_clean.csv")

# Known ASAP score ranges per prompt set
SCORE_RANGES = {
    1: (2, 12),
    2: (1, 6),
    3: (0, 3),
    4: (0, 3),
    5: (0, 4),
    6: (0, 4),
    7: (2, 24),
    8: (10, 60),
}


def download_dataset_if_needed():
    """Ensure raw dataset is available locally."""
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(LOCAL_RAW_PATH):
        print(f"[*] Downloading ASAP Essay dataset from {ASAP_DATA_URL}...")
        resp = requests.get(ASAP_DATA_URL, timeout=60)
        resp.raise_for_status()
        with open(LOCAL_RAW_PATH, "wb") as f:
            f.write(resp.content)
        print(f"[+] Download complete: saved to {LOCAL_RAW_PATH} ({len(resp.content) / (1024*1024):.2f} MB)")
    else:
        print(f"[i] Raw dataset found at {LOCAL_RAW_PATH}")


def normalize_score(row):
    """Normalize prompt-specific score to a universal 1.0 - 10.0 scale."""
    set_id = row["essay_set"]
    raw_score = row["domain1_score"]
    s_min, s_max = SCORE_RANGES.get(set_id, (0, 10))
    # Map [s_min, s_max] -> [1.0, 10.0]
    norm = ((raw_score - s_min) / (s_max - s_min)) * 9.0 + 1.0
    return round(float(norm), 2)


def load_and_preprocess_data():
    """Load, clean, and standardize the essay dataset."""
    download_dataset_if_needed()
    print("[*] Loading raw dataset with ISO-8859-1 encoding...")
    df = pd.read_csv(LOCAL_RAW_PATH, sep="\t", encoding="ISO-8859-1")
    
    # Filter valid essays and scores
    df = df.dropna(subset=["essay", "domain1_score"]).copy()
    df["essay"] = df["essay"].astype(str).str.strip()
    df = df[df["essay"].str.len() > 20].copy()
    
    # Compute normalized score
    df["normalized_score"] = df.apply(normalize_score, axis=1)
    
    # Select clean subset of columns
    clean_df = df[["essay_id", "essay_set", "essay", "domain1_score", "normalized_score"]].copy()
    clean_df.rename(columns={"domain1_score": "raw_score"}, inplace=True)
    
    # Save clean dataset for quick inspection
    clean_df.to_csv(CLEAN_DATA_PATH, index=False)
    print(f"[+] Processed {len(clean_df)} valid essays. Clean data saved to {CLEAN_DATA_PATH}")
    return clean_df


def extract_linguistic_features(texts):
    """
    Extract linguistic, structural, and readability features from essay texts.
    Features:
    1. Word Count
    2. Character Count
    3. Sentence Count
    4. Paragraph Count
    5. Average Word Length
    6. Average Sentence Length
    7. Lexical Diversity (Type-Token Ratio)
    8. Syllable Count Approximation
    9. Flesch Reading Ease
    10. Automated Readability Index (ARI)
    11. Syntactic Punctuation Count (commas, semicolons, quotes, dashes)
    12. Exclamation & Question Marks Count
    """
    feature_matrix = []
    
    for text in texts:
        # Tokenize words
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        num_words = max(len(words), 1)
        
        # Tokenize sentences
        sentences = [s.strip() for s in re.split(r"[\.\?!]+", text) if len(s.strip()) > 0]
        num_sentences = max(len(sentences), 1)
        
        # Paragraphs
        paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 0]
        num_paragraphs = max(len(paragraphs), 1)
        
        char_count = len(text)
        avg_word_len = sum(len(w) for w in words) / num_words
        avg_sent_len = num_words / num_sentences
        lexical_diversity = len(set(words)) / num_words
        
        # Syllable approximation (count vowel groups)
        syllables = sum(max(1, len(re.findall(r"[aeiouy]+", w))) for w in words)
        
        # Readability metrics
        flesch = 206.835 - (1.015 * avg_sent_len) - (84.6 * (syllables / num_words))
        ari = (4.71 * (char_count / num_words)) + (0.5 * avg_sent_len) - 21.43
        
        # Punctuation & complexity markers
        punct_count = len(re.findall(r"[,;:\(\)\"'\-]", text))
        emphasis_count = len(re.findall(r"[\?!]", text))
        
        feature_matrix.append([
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
        ])
        
    return np.array(feature_matrix)


FEATURE_NAMES = [
    "Word Count",
    "Character Count",
    "Sentence Count",
    "Paragraph Count",
    "Avg Word Length",
    "Avg Sentence Length",
    "Lexical Diversity (TTR)",
    "Estimated Syllables",
    "Flesch Reading Ease",
    "Automated Readability Index",
    "Punctuation Marks",
    "Emphasis Marks"
]


def train_and_evaluate():
    """Main training routine comparing models and saving the best artifact."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    df = load_and_preprocess_data()
    
    # Split train/test (80/20) with fixed random seed
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["essay"], df["normalized_score"], test_size=0.20, random_state=42
    )
    
    print(f"[*] Training Set: {len(X_train_text)} essays | Test Set: {len(X_test_text)} essays")
    
    print("[*] Extracting linguistic and structural features...")
    train_ling_feats = extract_linguistic_features(X_train_text)
    test_ling_feats = extract_linguistic_features(X_test_text)
    
    scaler = StandardScaler()
    train_ling_scaled = scaler.fit_transform(train_ling_feats)
    test_ling_scaled = scaler.transform(test_ling_feats)
    
    print("[*] Computing TF-IDF n-gram features (max 400 vocabulary)...")
    tfidf = TfidfVectorizer(max_features=400, ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    train_tfidf = tfidf.fit_transform(X_train_text)
    test_tfidf = tfidf.transform(X_test_text)
    
    # Combine engineered linguistic features + TF-IDF vectors
    X_train_full = hstack([train_ling_scaled, train_tfidf])
    X_test_full = hstack([test_ling_scaled, test_tfidf])
    
    models = {
        "Ridge Regression": Ridge(alpha=1.5, random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(n_estimators=120, learning_rate=0.08, max_depth=5, random_state=42)
    }
    
    results = {}
    best_model_name = None
    lowest_mae = float("inf")
    trained_estimators = {}
    
    print("\n" + "=" * 65)
    print("MODEL EVALUATION BENCHMARK ON TEST SPLIT (20% Holdout)")
    print("=" * 65)
    
    for name, model in models.items():
        print(f"[*] Training {name}...")
        model.fit(X_train_full, y_train)
        preds = model.predict(X_test_full)
        # Clip predictions to valid 1-10 range
        preds = np.clip(preds, 1.0, 10.0)
        
        mae = float(mean_absolute_error(y_test, preds))
        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))
        
        # Quadratic Weighted Kappa on integer-rounded scores
        y_test_discrete = np.round(y_test).astype(int)
        preds_discrete = np.round(preds).astype(int)
        qwk = float(cohen_kappa_score(y_test_discrete, preds_discrete, weights="quadratic"))
        
        results[name] = {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2_Score": round(r2, 4),
            "QWK": round(qwk, 4)
        }
        trained_estimators[name] = model
        
        print(f" -> {name}:")
        print(f"    MAE:  {mae:.4f}")
        print(f"    RMSE: {rmse:.4f}")
        print(f"    R²:   {r2:.4f}")
        print(f"    QWK:  {qwk:.4f}")
        
        if mae < lowest_mae:
            lowest_mae = mae
            best_model_name = name

    print("=" * 65)
    print(f"[+] Winner Model Selected: {best_model_name} (Lowest MAE: {lowest_mae:.4f})")
    print("=" * 65)
    
    best_model = trained_estimators[best_model_name]
    
    # Save artifacts bundle
    artifacts = {
        "model": best_model,
        "model_name": best_model_name,
        "scaler": scaler,
        "tfidf": tfidf,
        "feature_names": FEATURE_NAMES,
        "score_range": (1.0, 10.0),
        "evaluation_metrics": results
    }
    
    model_save_path = os.path.join(MODELS_DIR, "essay_scoring_model.pkl")
    joblib.dump(artifacts, model_save_path)
    print(f"[+] Model pipeline artifacts successfully saved to {model_save_path}")
    
    # Save metrics JSON
    metrics_path = os.path.join(MODELS_DIR, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            "best_model": best_model_name,
            "models_evaluated": results,
            "train_samples": len(X_train_text),
            "test_samples": len(X_test_text)
        }, f, indent=4)
    print(f"[+] Evaluation metrics saved to {metrics_path}")
    
    # Save sample essays for quick testing in UI
    save_sample_essays(df)


def save_sample_essays(df):
    """Save 3 representative sample essays (high, medium, low) for interactive testing."""
    samples = {}
    
    high = df[df["normalized_score"] >= 8.5].sort_values("normalized_score", ascending=False).iloc[0]
    med = df[(df["normalized_score"] >= 5.0) & (df["normalized_score"] <= 6.0)].iloc[0]
    low = df[df["normalized_score"] <= 3.0].sort_values("normalized_score").iloc[0]
    
    samples["High Scoring Essay"] = {
        "text": high["essay"],
        "actual_score": float(high["normalized_score"]),
        "level": "Proficient / Advanced (Score: 8.5+)"
    }
    samples["Medium Scoring Essay"] = {
        "text": med["essay"],
        "actual_score": float(med["normalized_score"]),
        "level": "Intermediate / Developing (Score: ~5.5)"
    }
    samples["Low Scoring Essay"] = {
        "text": low["essay"],
        "actual_score": float(low["normalized_score"]),
        "level": "Beginning / Basic (Score: <= 3.0)"
    }
    
    sample_path = os.path.join(DATA_DIR, "sample_essays.json")
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump(samples, f, indent=4)
    print(f"[+] Sample test essays saved to {sample_path}")


if __name__ == "__main__":
    train_and_evaluate()
