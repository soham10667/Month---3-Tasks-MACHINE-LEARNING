# 🎬 CineMatch — Movie Recommendation System

An intelligent, multi-engine Movie Recommendation System built with Python, scikit-learn, and Streamlit using the benchmark **MovieLens Latest Small** dataset curated by GroupLens Research.

---

## 📌 Project Overview

With thousands of movies available across modern streaming platforms, users face choice overload. This project implements and compares two core recommendation paradigms:
1. **Content-Based Filtering**: Suggests films based on genre tags and lexical metadata using TF-IDF and vector-space Cosine Similarity.
2. **Collaborative Filtering**: Suggests films based on user consumption patterns and latent tastes using **TruncatedSVD Matrix Factorization**.
3. **IMDb Bayesian Weighted Rating**: Provides a statistically balanced baseline for discovering top-rated films by genre, eliminating small-sample review distortion.

---

## 🗂️ Dataset Details

* **Dataset**: MovieLens Latest Small (`ml-latest-small.zip`)
* **Source**: GroupLens Research, University of Minnesota ([Grouplens Dataset](https://files.grouplens.org/datasets/movielens/ml-latest-small.zip))
* **Total Movies**: 9,742 titles spanning 20 distinct genres.
* **Total Ratings**: 100,836 validated user ratings across 610 registered users.
* **Rating Scale**: 0.5 to 5.0 stars.
* **Auto-download**: The training script `train_model.py` automatically downloads, extracts, and caches the dataset into `data/`.

---

## 🔬 Algorithmic Architecture

### 1. Content-Based Filtering
* Combines cleaned titles and pipe-separated genres into a textual corpus per film.
* Computes **TF-IDF (Term Frequency - Inverse Document Frequency)** feature representations with unigrams and bigrams.
* Incurs sub-second inference using vector dot products over normalized sparse arrays to produce match percentages ($0\% - 100\%$).

### 2. Collaborative Filtering (Latent Factor Model)
* Formulates a sparse User-Item interaction matrix $R \in \mathbb{R}^{610 \times 9724}$.
* Applies **Truncated Singular Value Decomposition (SVD)** to decompose ratings into latent preference factors:
  $$R \approx U \cdot \Sigma \cdot V^T$$
* Extracts 25 latent dimensions capturing underlying aesthetic preferences (e.g. 90s indie cinema, gritty crime dramas).
* Computes cosine similarity between normalized movie latent factors $V^T$ to recommend titles liked by users with similar profiles.

### 3. IMDb Bayesian Weighted Rating
* Solves the issue where a movie with a single 5.0 rating outranks a masterpiece with thousands of 4.8 ratings:
  $$\text{WR} = \left(\frac{v}{v+m}\right) R + \left(\frac{m}{v+m}\right) C$$
  where $v$ is rating count, $m$ is the 65th percentile rating threshold, $R$ is movie average rating, and $C$ is the catalog global average ($\approx 3.50$).

---

## 📊 Evaluation & Dataset Metrics

Evaluated on a **20% holdout test split** of user rating records:

| Metric | Result | Description |
|---|---|---|
| **SVD Latent Components** | 25 | Dimensions in decomposed feature space |
| **SVD Explained Variance** | 38.19% | Variance captured from user interaction matrix |
| **Collaborative Holdout MAE** | 2.5420 | Holdout rating prediction absolute error |
| **Collaborative Holdout RMSE** | 2.8232 | Holdout root mean squared rating error |
| **Matrix Sparsity** | 98.3% | Sparsity percentage of the user-movie grid |
| **Model Size** | 1.51 MB | Compact, GitHub-safe artifact |

---

## 🚀 Installation & Running

### 1. Prerequisites
Ensure Python 3.10+ is installed.

### 2. Install Dependencies
Navigate to this project directory:
```bash
cd Movie-Recommendation-System
pip install -r requirements.txt
```

### 3. Train / Preprocess Models (Optional - Pretrained model included)
To re-run the training pipeline and generate fresh artifacts:
```bash
python train_model.py
```
This will:
* Download and extract MovieLens data into `data/`.
* Compute IMDb weighted ratings and build TF-IDF matrices.
* Train TruncatedSVD and evaluate collaborative filtering on a holdout split.
* Save the compact model artifact to `models/movie_recommender.pkl` (1.51 MB).

### 4. Launch the Streamlit Application
```bash
streamlit run app.py
```
The app will open automatically in your browser at `http://localhost:8501`.

---

## 💡 Application Features

* **Multi-Strategy Recommendations**: Toggle seamlessly between Content-Based, Collaborative, and Top-Rated modes.
* **Search & Genre Filter**: Quickly find any baseline movie using keyword searches or genre dropdowns.
* **Ranked Result Cards**: Rich movie cards displaying Release Year, Genre Badges, Match Percentage, and Star Ratings.
* **Adjustable Slider**: Customize the number of recommendations from 5 to 25.
* **Transparent Architecture Tabs**: View detailed explanations of algorithms and system limitations.

---

## ⚠️ System Limitations & Trade-Offs

1. **Cold-Start Problem**: Newly added movies or users without prior ratings cannot be analyzed by Collaborative Filtering. Content-based filtering is recommended as a fallback.
2. **Matrix Sparsity**: With 98.3% of the rating matrix unobserved, niche films may have weaker latent representations.
3. **Popularity Bias**: Heavily rated blockbusters naturally appear more often in user co-occurrence patterns.
4. **No Deep Narrative Synopses**: Relies on metadata tokens and genres; does not inspect full plot scripts or cinematography nuances.
