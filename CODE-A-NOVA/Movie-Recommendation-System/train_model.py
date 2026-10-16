"""
Movie Recommendation System - Model Training & Preprocessing Pipeline
Code-A-Nova Internship Project

Dataset Source: MovieLens Latest Small Dataset (GroupLens Research, University of Minnesota)
URL: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip

Algorithms Implemented:
1. Content-Based Filtering (TF-IDF Vectorization & Vector Space Cosine Similarity on genres + title metadata).
2. Collaborative Filtering (User-Item Matrix Factorization via TruncatedSVD & Item Latent Factor Similarity).
3. Bayesian Weighted Rating (IMDb Top Rated formula for popularity baseline).
"""

import os
import re
import io
import zipfile
import json
import requests
import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
MOVIELENS_URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"


def download_and_extract_movielens():
    """Download MovieLens latest small dataset and extract movies and ratings."""
    os.makedirs(DATA_DIR, exist_ok=True)
    movies_path = os.path.join(DATA_DIR, "movies.csv")
    ratings_path = os.path.join(DATA_DIR, "ratings.csv")

    if not os.path.exists(movies_path) or not os.path.exists(ratings_path):
        print(f"[*] Downloading MovieLens dataset from {MOVIELENS_URL}...")
        resp = requests.get(MOVIELENS_URL, timeout=60)
        resp.raise_for_status()
        print(f"[+] Download complete ({len(resp.content)/(1024*1024):.2f} MB). Unzipping files...")

        with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
            for member in z.namelist():
                filename = os.path.basename(member)
                if filename in ["movies.csv", "ratings.csv", "tags.csv", "links.csv"]:
                    target_file = os.path.join(DATA_DIR, filename)
                    with open(target_file, "wb") as f_out:
                        f_out.write(z.read(member))
                    print(f" -> Extracted: {filename}")
    else:
        print(f"[i] Dataset files already present in {DATA_DIR}")

    movies_df = pd.read_csv(movies_path)
    ratings_df = pd.read_csv(ratings_path)
    return movies_df, ratings_df


def extract_year(title):
    """Extract release year from movie title e.g., 'Toy Story (1995)' -> 1995."""
    match = re.search(r"\((\d{4})\)", str(title))
    return int(match.group(1)) if match else None


def clean_title(title):
    """Clean movie title by removing year suffix."""
    return re.sub(r"\s*\(\d{4}\)", "", str(title)).strip()


def compute_weighted_ratings(movies_df, ratings_df):
    """
    Compute IMDb-style Bayesian Weighted Rating:
    WR = (v / (v + m)) * R + (m / (v + m)) * C
    v: number of ratings for movie
    m: minimum ratings threshold
    R: average rating of movie
    C: global mean rating across all movies
    """
    stats = ratings_df.groupby("movieId").agg(
        vote_count=("rating", "count"),
        vote_average=("rating", "mean")
    ).reset_index()

    movies_with_stats = pd.merge(movies_df, stats, on="movieId", how="left")
    movies_with_stats["vote_count"] = movies_with_stats["vote_count"].fillna(0).astype(int)
    movies_with_stats["vote_average"] = movies_with_stats["vote_average"].fillna(0.0).round(2)

    C = ratings_df["rating"].mean()
    m = stats["vote_count"].quantile(0.65)  # 65th percentile threshold

    def weighted_rating(x):
        v = x["vote_count"]
        R = x["vote_average"]
        if v == 0:
            return 0.0
        return round((v / (v + m) * R) + (m / (v + m) * C), 2)

    movies_with_stats["weighted_rating"] = movies_with_stats.apply(weighted_rating, axis=1)
    return movies_with_stats, float(C), float(m)


def train_content_based_model(movies_df):
    """
    Build TF-IDF content representations on combined genres and title tokens.
    Saves the compact sparse matrix directly for ultra-fast, memory-efficient on-the-fly similarity.
    """
    formatted_genres = movies_df["genres"].str.replace("|", " ", regex=False)
    corpus = movies_df["clean_title"] + " " + formatted_genres

    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(corpus)
    return tfidf, tfidf_matrix


def train_collaborative_filtering_model(ratings_df, movies_df, n_components=25):
    """
    Train TruncatedSVD Matrix Factorization on the User-Item rating matrix.
    Stores compact normalized item factors for instant cosine similarity querying.
    """
    train_ratings, test_ratings = train_test_split(ratings_df, test_size=0.20, random_state=42)

    user_item_matrix = train_ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    user_factors = svd.fit_transform(user_item_matrix)
    item_factors = svd.components_.T  # Shape: (num_movies, n_components)

    # Normalize item factors so dot product equals cosine similarity
    norms = np.linalg.norm(item_factors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    norm_item_factors = item_factors / norms

    movie_ids_in_matrix = list(user_item_matrix.columns)
    movie_id_to_idx = {mid: idx for idx, mid in enumerate(movie_ids_in_matrix)}

    # Evaluate reconstructed ratings on holdout test set
    reconstructed_matrix = np.dot(user_factors, svd.components_)
    reconstructed_df = pd.DataFrame(reconstructed_matrix, index=user_item_matrix.index, columns=user_item_matrix.columns)

    actuals = []
    predictions = []
    global_mean = float(train_ratings["rating"].mean())

    for _, row in test_ratings.iterrows():
        u = row["userId"]
        m = row["movieId"]
        if u in reconstructed_df.index and m in reconstructed_df.columns:
            pred = reconstructed_df.loc[u, m]
            if pred <= 0:
                pred = global_mean
            actuals.append(row["rating"])
            predictions.append(pred)

    mae = float(mean_absolute_error(actuals, predictions))
    rmse = float(np.sqrt(mean_squared_error(actuals, predictions)))
    explained_var = float(svd.explained_variance_ratio_.sum())

    cf_metrics = {
        "SVD_Components": n_components,
        "Explained_Variance_Ratio": round(explained_var, 4),
        "Test_MAE": round(mae, 4),
        "Test_RMSE": round(rmse, 4),
        "Total_Users": int(ratings_df["userId"].nunique()),
        "Total_Movies_Rated": int(ratings_df["movieId"].nunique()),
        "Total_Ratings": len(ratings_df),
        "Matrix_Sparsity_Percent": round(100.0 * (1 - len(ratings_df) / (ratings_df["userId"].nunique() * ratings_df["movieId"].nunique())), 2)
    }

    return norm_item_factors, movie_ids_in_matrix, movie_id_to_idx, cf_metrics


def build_and_save_pipeline():
    """Execute full recommendation training pipeline and save compact artifacts."""
    os.makedirs(MODELS_DIR, exist_ok=True)
    movies_df, ratings_df = download_and_extract_movielens()

    print("[*] Preprocessing movie metadata...")
    movies_df["year"] = movies_df["title"].apply(extract_year)
    movies_df["clean_title"] = movies_df["title"].apply(clean_title)

    print("[*] Computing IMDb Bayesian weighted ratings...")
    movies_with_stats, global_mean, min_votes = compute_weighted_ratings(movies_df, ratings_df)

    print("[*] Training Content-Based Filtering engine (TF-IDF matrix)...")
    tfidf_vectorizer, tfidf_matrix = train_content_based_model(movies_with_stats)

    print("[*] Training Collaborative Filtering engine (TruncatedSVD Matrix Factorization)...")
    norm_item_factors, cf_movie_ids, cf_movie_idx, cf_metrics = train_collaborative_filtering_model(ratings_df, movies_with_stats)

    print("\n" + "=" * 65)
    print("COLLABORATIVE FILTERING EVALUATION & DATASET METRICS")
    print("=" * 65)
    for k, v in cf_metrics.items():
        print(f" {k:<28}: {v}")
    print("=" * 65)

    all_genres = sorted(list(set(
        g for genres in movies_with_stats["genres"].dropna() for g in genres.split("|") if g != "(no genres listed)"
    )))

    # Save compact artifacts bundle (optimized for GitHub and sub-second inference)
    artifacts = {
        "movies_df": movies_with_stats,
        "tfidf_matrix": tfidf_matrix,
        "norm_item_factors": norm_item_factors.astype(np.float32),
        "cf_movie_ids": cf_movie_ids,
        "cf_movie_idx": cf_movie_idx,
        "all_genres": all_genres,
        "global_mean_rating": global_mean,
        "min_votes_threshold": min_votes,
        "metrics": cf_metrics
    }

    model_path = os.path.join(MODELS_DIR, "movie_recommender.pkl")
    print(f"[*] Saving optimized model artifacts to {model_path}...")
    joblib.dump(artifacts, model_path, compress=5)
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"[+] Model saved successfully: {size_mb:.2f} MB (GitHub safe, < 100MB)")

    # Save metrics JSON
    metrics_path = os.path.join(MODELS_DIR, "evaluation_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(cf_metrics, f, indent=4)
    print(f"[+] Metrics saved to {metrics_path}")


if __name__ == "__main__":
    build_and_save_pipeline()
