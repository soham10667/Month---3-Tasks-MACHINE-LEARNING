# 🚀 Machine Learning Projects — Month 3 Tasks

A comprehensive, production-grade repository housing two independent, end-to-end Machine Learning applications:

1. **✍️ Automated Essay Scoring (AES)**: Evaluates student essays using structural linguistics, lexical diversity, readability indices (Flesch Reading Ease & ARI), and n-gram TF-IDF representations trained on the Hewlett Foundation ASAP benchmark dataset.
2. **🎬 Movie Recommendation System (CineMatch)**: A multi-strategy movie recommendation engine combining TF-IDF Content-Based Filtering, TruncatedSVD Matrix Factorization Collaborative Filtering, and IMDb Bayesian Weighted Rating on the MovieLens dataset.

Both applications feature complete end-to-end data preprocessing pipelines, model training and serialization, offline benchmark evaluations, and modern interactive **Streamlit** user interfaces.

---

## 📁 Repository Structure

```text
.
├── CODE-A-NOVA/
│   ├── Automated-Essay-Scoring/
│   │   ├── data/
│   │   │   ├── essays_clean.csv            # Cleaned ASAP benchmark dataset (12,970+ essays)
│   │   │   └── sample_essays.json          # Curated test essays for interactive scoring
│   │   ├── models/
│   │   │   ├── essay_scoring_model.pkl     # Serialized best regression pipeline (Gradient Boosting)
│   │   │   └── evaluation_metrics.json     # Quantitative test benchmark (MAE, RMSE, R², QWK)
│   │   ├── app.py                          # Interactive Streamlit application
│   │   ├── train_model.py                  # End-to-end training & evaluation pipeline
│   │   ├── requirements.txt                # Project-specific dependencies
│   │   ├── README.md                       # Sub-project documentation
│   │   └── .gitignore                      # Project-level ignore rules
│   │
│   ├── Movie-Recommendation-System/
│   │   ├── data/
│   │   │   ├── movies.csv                  # Movie catalog metadata (9,742 titles)
│   │   │   ├── ratings.csv                 # 100,836 user ratings
│   │   │   ├── tags.csv                    # User-generated genre tags
│   │   │   └── links.csv                   # IMDb/TMDb identifiers
│   │   ├── models/
│   │   │   ├── movie_recommender.pkl       # Serialized models, TF-IDF matrix & SVD factors
│   │   │   └── evaluation_metrics.json     # Matrix factorization evaluation metrics
│   │   ├── app.py                          # Interactive Streamlit application
│   │   ├── train_model.py                  # Multi-engine training & evaluation pipeline
│   │   ├── requirements.txt                # Project-specific dependencies
│   │   ├── README.md                       # Sub-project documentation
│   │   └── .gitignore                      # Project-level ignore rules
│   │
│   └── README.md                           # CODE-A-NOVA sub-directory documentation
│
├── run_all.bat                             # One-click Windows launcher for both applications
├── run_essay_scorer.bat                    # One-click Windows launcher for Essay Scorer (Port 8501)
├── run_movie_recommender.bat               # One-click Windows launcher for Movie Recommender (Port 8502)
├── README.md                               # Master repository documentation
└── .gitignore                              # Global repository ignore rules
```

---

## 🛠️ Quick Start & Installation

Ensure you have **Python 3.10+** installed. Both sub-projects are fully self-contained.

### ⚡ One-Click Windows Launch

To launch both applications in separate terminal windows with a single command:

```cmd
.\run_all.bat
```

Or run individual apps:
- `.\run_essay_scorer.bat` $\rightarrow$ Launches Essay Scorer on **http://localhost:8501**
- `.\run_movie_recommender.bat` $\rightarrow$ Launches Movie Recommender on **http://localhost:8502**

---

### 💻 Manual Execution (Terminal / PowerShell)

#### 1. Automated Essay Scoring (Port 8501)

```powershell
# 1. Navigate to directory
cd CODE-A-NOVA/Automated-Essay-Scoring

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Re-train and evaluate models
python train_model.py

# 4. Launch Streamlit application
python -m streamlit run app.py --server.port 8501
```

#### 2. Movie Recommendation System (Port 8502)

```powershell
# 1. Navigate to directory
cd CODE-A-NOVA/Movie-Recommendation-System

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Re-run data pipeline & matrix factorization
python train_model.py

# 4. Launch Streamlit application
python -m streamlit run app.py --server.port 8502
```

---

## 📊 Summary of Evaluation & Results

### 1. Automated Essay Scoring (AES)
Evaluated on **2,595 unseen test essays (20% holdout split)** across human rater benchmarks on a standardized continuous 1.0–10.0 scale:

| Model | MAE | RMSE | $R^2$ Score | Quadratic Weighted Kappa (QWK) |
|---|:---:|:---:|:---:|:---:|
| Ridge Regression | 1.0934 | 1.4048 | 0.5743 | 0.7123 |
| Random Forest Regressor | 1.0281 | 1.3568 | 0.6029 | 0.7352 |
| **Gradient Boosting Regressor (Selected)** | **1.0112** | **1.3126** | **0.6283** | **0.7563** |

* **Key Finding**: Gradient Boosting achieved the lowest error and a Quadratic Weighted Kappa of **0.7563**, confirming strong alignment with human expert scoring standards.

---

### 2. Movie Recommendation System (CineMatch)
Evaluated across **100,836 ratings** and **9,742 movies** from the MovieLens benchmark:

| Metric | Measured Value | Notes |
|---|:---:|---|
| **SVD Latent Components** | 25 | Dimensions in reduced latent factor space |
| **SVD Explained Variance** | 38.19% | Captured from user-movie interaction matrix |
| **Catalog Matrix Sparsity** | 98.3% | Rating matrix sparsity percentage |
| **Inference Latency** | < 50ms | Real-time cosine similarity & dot-product search |
| **Artifact Footprint** | 1.51 MB | Compact, sub-second loading footprint |

---

## 🎨 UI & User Experience Highlights

- **Clean, Professional Software UI**: Designed with modern, restrained dark-mode aesthetics (inspired by GitHub, Linear, and Tailwind UI) avoiding loud neon gradients or artificial styling.
- **High Contrast & Dark Mode Optimized**: Fully tested in dark mode with high contrast typography, accessible status pills, and clear metric tables.
- **Actionable Diagnostics**: The Essay Scorer breaks down essay performance by volume, syntactic complexity, vocabulary richness, and readability (Flesch Reading Ease & ARI).
- **Multi-Strategy Recommender**: CineMatch lets users toggle between Content-Based lexical similarity, Collaborative latent factor user taste matching, and Bayesian top-rated genre exploration.

---

## ⚠️ Academic Transparency & Ethical Limitations

* **Automated Essay Scoring**:
  * Employs statistical surface linguistics, readability formulas (Flesch, ARI), and TF-IDF topical representations.
  * Does not claim exhaustive rule-based syntax dependency grammar parsing or deep semantic fact-checking. Designed as an educational feedback tool for practice and drafting.
* **Movie Recommendation System**:
  * Collaborative filtering is subject to cold-start conditions for unrated movies or new users (content-based fallback is provided automatically).
  * 98.3% rating matrix sparsity means recommendations for niche indie titles rely more on content metadata than user co-rating patterns.

---

## 👨‍💻 Repository Information

* **Repository**: [soham10667/Month---3-Tasks-MACHINE-LEARNING](https://github.com/soham10667/Month---3-Tasks-MACHINE-LEARNING)
* **Language**: Python 3.10+
* **Core Libraries**: `scikit-learn`, `Streamlit`, `pandas`, `numpy`, `scipy`, `joblib`
