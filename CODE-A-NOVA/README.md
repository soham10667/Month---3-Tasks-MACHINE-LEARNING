# 🚀 CODE-A-NOVA — Machine Learning Projects

A comprehensive repository housing two independent, end-to-end Machine Learning applications developed for the **Code-A-Nova Internship**:

1. **✍️ Automated Essay Scoring (AES)**: Evaluates student essays based on structural linguistics, lexical diversity, readability metrics, and TF-IDF topical representations using the benchmark Hewlett Foundation ASAP dataset.
2. **🎬 Movie Recommendation System (CineMatch)**: Recommends movies through TF-IDF Content-Based Filtering, TruncatedSVD Matrix Factorization Collaborative Filtering, and IMDb Bayesian Weighted Rating on the MovieLens dataset.

Both projects feature dedicated data pipelines, machine learning training workflows, model persistence, evaluation benchmarks, and interactive **Streamlit** user interfaces.

---

## 📁 Repository Structure

```text
CODE-A-NOVA/
│
├── Automated-Essay-Scoring/
│   ├── data/
│   │   ├── essays_clean.csv            # Cleaned essay benchmark dataset (12,970+ essays)
│   │   └── sample_essays.json          # Curated test essays for UI evaluation
│   ├── models/
│   │   ├── essay_scoring_model.pkl     # Serialized best regression pipeline (526 KB)
│   │   └── evaluation_metrics.json     # Quantitative test benchmark (MAE, RMSE, QWK)
│   ├── app.py                          # Interactive Streamlit application
│   ├── train_model.py                  # End-to-end training & evaluation pipeline
│   ├── requirements.txt                # Project-specific dependencies
│   ├── README.md                       # Comprehensive documentation
│   └── .gitignore                      # Project-level ignore rules
│
├── Movie-Recommendation-System/
│   ├── data/
│   │   ├── movies.csv                  # Movie metadata (9,742 titles)
│   │   ├── ratings.csv                 # 100,836 user ratings
│   │   ├── tags.csv                    # User-generated genre tags
│   │   └── links.csv                   # IMDb/TMDb identifiers
│   ├── models/
│   │   ├── movie_recommender.pkl       # Serialized models & latent factors (1.51 MB)
│   │   └── evaluation_metrics.json     # Matrix factorization evaluation metrics
│   ├── app.py                          # Interactive Streamlit application
│   ├── train_model.py                  # Multi-engine training & evaluation pipeline
│   ├── requirements.txt                # Project-specific dependencies
│   ├── README.md                       # Comprehensive documentation
│   └── .gitignore                      # Project-level ignore rules
│
├── README.md                           # Master repository documentation
└── .gitignore                          # Global repository ignore rules
```

---

## 🛠️ Quick Start & Installation

Ensure you have Python 3.10+ installed. Both projects are completely self-contained.

### Project 1: Automated Essay Scoring
```bash
# 1. Navigate to the project directory
cd Automated-Essay-Scoring

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Re-train and evaluate models
python train_model.py

# 4. Launch the Streamlit application
streamlit run app.py
```

### Project 2: Movie Recommendation System
```bash
# 1. Navigate to the project directory
cd Movie-Recommendation-System

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Re-run data pipeline and model training
python train_model.py

# 4. Launch the Streamlit application
streamlit run app.py
```

---

## 📊 Summary of Evaluation & Results

### 1. Automated Essay Scoring
Evaluated on **2,595 unseen test essays (20% holdout split)** on a normalized 1.0–10.0 scale:

| Model | MAE | RMSE | $R^2$ Score | Quadratic Weighted Kappa (QWK) |
|---|---|---|---|---|
| Ridge Regression | 1.0934 | 1.4048 | 0.5743 | 0.7123 |
| Random Forest Regressor | 1.0281 | 1.3568 | 0.6029 | 0.7349 |
| **Gradient Boosting Regressor (Selected)** | **1.0112** | **1.3126** | **0.6283** | **0.7563** |

* **Key Takeaway**: Gradient Boosting achieved the lowest error and a Quadratic Weighted Kappa of **0.7563**, confirming strong alignment with human expert scoring standards.

### 2. Movie Recommendation System
Evaluated on a **20% holdout split** across 100,836 ratings:

| Metric | Measured Value | Notes |
|---|---|---|
| **SVD Latent Components** | 25 | Dimensions in reduced latent space |
| **SVD Explained Variance** | 38.19% | Captured from user-movie interaction matrix |
| **Holdout Rating MAE** | 2.5420 | Holdout rating prediction absolute error |
| **Holdout Rating RMSE** | 2.8232 | Holdout root mean squared error |
| **Catalog Matrix Sparsity** | 98.3% | Rating matrix sparsity percentage |
| **Artifact Footprint** | 1.51 MB | Compact, sub-second inference runtime |

---

## ⚠️ Academic Transparency & Ethical Limitations

* **Automated Essay Scoring**:
  * Utilizes statistical surface linguistics, readability algorithms (Flesch Reading Ease, ARI), and TF-IDF representations.
  * Does not claim exhaustive rule-based syntax dependency grammar parsing or deep semantic fact-checking. Intended for educational feedback and practice.
* **Movie Recommendation System**:
  * Collaborative filtering is subject to cold-start limitations for unrated movies and new users (content-based fallback is provided).
  * 98.3% rating matrix sparsity means recommendations for niche indie titles rely more on content metadata than user co-rating patterns.

---

## 👨‍💻 Author & Internship Details

* **Program**: Code-A-Nova Internship
* **Repository**: `CODE-A-NOVA`
* **Language**: Python 3.10+
* **Frameworks**: scikit-learn, Streamlit, pandas, numpy, scipy, joblib
