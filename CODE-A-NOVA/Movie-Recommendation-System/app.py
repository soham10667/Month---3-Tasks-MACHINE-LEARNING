"""
Movie Recommendation System - Interactive Streamlit Application
Code-A-Nova Internship Project
"""

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ---------------------------------------------------------
# Page Configuration & Styling
# ---------------------------------------------------------
st.set_page_config(
    page_title="CineMatch | Movie Recommendation System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .hero-banner {
        background: linear-gradient(135deg, #090d16 0%, #111827 50%, #1e1b4b 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 18px;
        color: #ffffff;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-banner h1 {
        color: #ffffff !important;
        font-weight: 800;
        font-size: 2.3rem;
        margin: 0 0 0.5rem 0;
        letter-spacing: -0.02em;
    }
    .hero-banner p {
        color: #94a3b8;
        font-size: 1.05rem;
        margin: 0;
        max-width: 820px;
    }

    .movie-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 1.25rem 1.4rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .movie-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 18px rgba(0, 0, 0, 0.08);
        border-color: #cbd5e1;
    }

    .badge-rank {
        background-color: #3b82f6;
        color: white;
        padding: 0.25rem 0.65rem;
        border-radius: 8px;
        font-weight: 800;
        font-size: 0.85rem;
        display: inline-block;
    }
    .badge-genre {
        background-color: #f1f5f9;
        color: #334155;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin-right: 0.35rem;
        margin-bottom: 0.25rem;
    }
    .badge-match {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.25rem 0.7rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        display: inline-block;
    }
    .badge-rating {
        background-color: #fef3c7;
        color: #b45309;
        padding: 0.25rem 0.6rem;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Artifact Loading
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "movie_recommender.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "models", "evaluation_metrics.json")


@st.cache_resource
def load_recommender_bundle():
    """Load cached recommender bundle."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    """Load evaluation metrics."""
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r") as f:
            return json.load(f)
    return None


bundle = load_recommender_bundle()
metrics = load_metrics()

# ---------------------------------------------------------
# Recommendation Helper Algorithms
# ---------------------------------------------------------
def get_content_recommendations(movie_idx: int, top_n: int = 10):
    """Compute on-the-fly cosine similarity using the sparse TF-IDF feature matrix."""
    tfidf_matrix = bundle["tfidf_matrix"]
    movies_df = bundle["movies_df"]

    # Target vector (1, vocab_size)
    target_vec = tfidf_matrix[movie_idx]
    
    # Cosine similarity is dot product of normalized sparse vectors
    sim_scores = (tfidf_matrix * target_vec.T).toarray().ravel()
    
    # Sort descending (exclude the target movie itself)
    sorted_indices = np.argsort(sim_scores)[::-1]
    sorted_indices = [idx for idx in sorted_indices if idx != movie_idx][:top_n]

    results = []
    for idx in sorted_indices:
        row = movies_df.iloc[idx]
        sim = float(sim_scores[idx])
        results.append({
            "movieId": row["movieId"],
            "title": row["title"],
            "clean_title": row["clean_title"],
            "year": row["year"],
            "genres": row["genres"],
            "vote_average": row["vote_average"],
            "vote_count": row["vote_count"],
            "weighted_rating": row["weighted_rating"],
            "score": sim,
            "score_label": f"{int(sim * 100)}% Content Match"
        })
    return results


def get_collaborative_recommendations(movie_idx: int, top_n: int = 10):
    """Compute similarity in TruncatedSVD latent factor space."""
    movies_df = bundle["movies_df"]
    target_mid = movies_df.iloc[movie_idx]["movieId"]
    
    cf_movie_idx = bundle["cf_movie_idx"]
    norm_factors = bundle["norm_item_factors"]
    cf_movie_ids = bundle["cf_movie_ids"]

    if target_mid not in cf_movie_idx:
        return None  # Cold start: unrated or not in rating matrix

    matrix_pos = cf_movie_idx[target_mid]
    target_vector = norm_factors[matrix_pos]  # shape (25,)

    # Dot product with all normalized item factors gives cosine similarity
    sim_scores = np.dot(norm_factors, target_vector)

    sorted_indices = np.argsort(sim_scores)[::-1]
    # Filter out target movie
    sorted_indices = [idx for idx in sorted_indices if idx != matrix_pos][:top_n]

    mid_to_df_row = {row["movieId"]: row for _, row in movies_df.iterrows()}

    results = []
    for pos in sorted_indices:
        mid = cf_movie_ids[pos]
        sim = float(sim_scores[pos])
        # scale similarity (-1 to 1) into percentage (0 to 100%)
        match_pct = max(0, min(100, int((sim + 1.0) / 2.0 * 100)))
        if mid in mid_to_df_row:
            row = mid_to_df_row[mid]
            results.append({
                "movieId": row["movieId"],
                "title": row["title"],
                "clean_title": row["clean_title"],
                "year": row["year"],
                "genres": row["genres"],
                "vote_average": row["vote_average"],
                "vote_count": row["vote_count"],
                "weighted_rating": row["weighted_rating"],
                "score": sim,
                "score_label": f"{match_pct}% User Taste Match"
            })
    return results


def get_top_rated_by_genre(selected_genre: str, top_n: int = 10, min_votes: int = 15):
    """Rank movies by IMDb Bayesian Weighted Rating within a selected genre."""
    movies_df = bundle["movies_df"]
    
    if selected_genre and selected_genre != "All Genres":
        filtered = movies_df[movies_df["genres"].str.contains(selected_genre, na=False, regex=False)]
    else:
        filtered = movies_df

    filtered = filtered[filtered["vote_count"] >= min_votes]
    ranked = filtered.sort_values("weighted_rating", ascending=False).head(top_n)

    results = []
    for _, row in ranked.iterrows():
        results.append({
            "movieId": row["movieId"],
            "title": row["title"],
            "clean_title": row["clean_title"],
            "year": row["year"],
            "genres": row["genres"],
            "vote_average": row["vote_average"],
            "vote_count": row["vote_count"],
            "weighted_rating": row["weighted_rating"],
            "score": row["weighted_rating"],
            "score_label": f"★ {row['weighted_rating']:.2f} Weighted Score"
        })
    return results


# ---------------------------------------------------------
# Header Display
# ---------------------------------------------------------
st.markdown("""
<div class="hero-banner">
    <h1>🎬 CineMatch Movie Recommendation System</h1>
    <p>Intelligent recommendation engine combining TF-IDF Content-Based Filtering, SVD Matrix Factorization Collaborative Filtering, and IMDb Bayesian Weighted Rating on the MovieLens benchmark.</p>
</div>
""", unsafe_allow_html=True)

if bundle is None:
    st.error("Model artifacts not found! Please run `python train_model.py` to generate models.")
    st.stop()

movies_df = bundle["movies_df"]
all_genres = bundle["all_genres"]

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎛️ Recommendation Strategy")
    mode = st.radio(
        "Choose Recommendation Approach:",
        [
            "🎯 Content-Based (Genre & Title Similarity)",
            "👥 Collaborative (User Rating Patterns & SVD)",
            "🏆 Top-Rated by Genre (IMDb Formula)"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("### ⚙️ Search & Parameters")
    num_recs = st.slider("Number of Recommendations:", min_value=5, max_value=25, value=10, step=1)

    st.markdown("---")
    st.markdown("### 📊 Dataset & Model Stats")
    if metrics:
        st.metric("Total Movies in Catalog", f"{len(movies_df):,}")
        st.metric("Ratings in Dataset", f"{metrics.get('Total_Ratings', 100836):,}")
        st.metric("SVD Explained Variance", f"{metrics.get('Explained_Variance_Ratio', 0.38)*100:.1f}%")
        st.caption(f"Matrix Sparsity: **{metrics.get('Matrix_Sparsity_Percent', 98.3)}%**")

# ---------------------------------------------------------
# Recommendation Modes Interface
# ---------------------------------------------------------
if "Content-Based" in mode or "Collaborative" in mode:
    st.markdown("### 🔍 Select a Baseline Movie")
    
    # Filter helpers
    col_genre_filter, col_search = st.columns([1, 2])
    with col_genre_filter:
        genre_filter = st.selectbox("Filter movie list by Genre:", ["All"] + all_genres)
    
    with col_search:
        search_kw = st.text_input("Filter by Keyword in Title:", placeholder="e.g. Matrix, Toy Story, Batman")

    # Filter available movies for dropdown
    filtered_list = movies_df
    if genre_filter != "All":
        filtered_list = filtered_list[filtered_list["genres"].str.contains(genre_filter, na=False, regex=False)]
    if search_kw.strip():
        filtered_list = filtered_list[filtered_list["clean_title"].str.contains(search_kw.strip(), case=False, na=False)]

    if len(filtered_list) == 0:
        st.warning("No movies matched your search filter. Showing complete catalog.")
        filtered_list = movies_df

    # Dropdown with index mapping
    movie_options = filtered_list["title"].tolist()
    selected_movie_title = st.selectbox("Choose a Movie You Like:", movie_options, index=0)
    
    # Find original index in movies_df
    selected_idx = movies_df[movies_df["title"] == selected_movie_title].index[0]
    selected_row = movies_df.iloc[selected_idx]

    # Display selected movie details
    col_sel_info, col_sel_stats = st.columns([2, 1])
    with col_sel_info:
        genres_badges = " ".join([f"<span class='badge-genre'>{g}</span>" for g in selected_row['genres'].split('|')])
        st.markdown(f"""
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 1.5rem;">
            <div style="font-size: 0.8rem; color: #64748b; font-weight: 700; text-transform: uppercase;">Selected Movie</div>
            <div style="font-size: 1.35rem; font-weight: 800; color: #0f172a; margin: 0.2rem 0;">{selected_row['clean_title']} <span style="color: #64748b; font-size: 1.1rem;">({selected_row['year'] or 'N/A'})</span></div>
            <div style="margin-top: 0.4rem;">{genres_badges}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_sel_stats:
        st.markdown(f"""
        <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1rem 1.2rem; margin-bottom: 1.5rem; text-align: center;">
            <div style="font-size: 0.8rem; color: #64748b; font-weight: 700; text-transform: uppercase;">Community Rating</div>
            <div style="font-size: 1.6rem; font-weight: 800; color: #b45309; margin: 0.2rem 0;">★ {selected_row['vote_average']} <span style="font-size: 0.9rem; color: #64748b;">/ 5.0</span></div>
            <div style="font-size: 0.8rem; color: #64748b;">({selected_row['vote_count']} user ratings)</div>
        </div>
        """, unsafe_allow_html=True)

    # Compute Recommendations
    if "Content-Based" in mode:
        recommendations = get_content_recommendations(selected_idx, top_n=num_recs)
        strategy_desc = "Content-Based Filtering: Recommended based on lexical similarity in genres and title concepts using TF-IDF and Cosine Similarity."
    else:
        recommendations = get_collaborative_recommendations(selected_idx, top_n=num_recs)
        strategy_desc = "Collaborative Filtering: Recommended based on latent factor user-rating similarity discovered via TruncatedSVD Matrix Factorization."

    st.markdown(f"### 📋 Ranked Recommendations ({len(recommendations) if recommendations else 0} results)")
    st.caption(strategy_desc)

    if recommendations is None:
        st.warning("⚠️ This movie has very few user ratings, triggering a **cold-start** condition for collaborative filtering. Switch to **Content-Based Filtering** to view recommendations based on genres and themes.")
    else:
        for rank, item in enumerate(recommendations, start=1):
            genre_tags = " ".join([f"<span class='badge-genre'>{g}</span>" for g in str(item['genres']).split('|')])
            st.markdown(f"""
            <div class="movie-card">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 0.5rem;">
                    <div>
                        <span class="badge-rank">#{rank}</span>
                        <span style="font-size: 1.2rem; font-weight: 800; color: #0f172a; margin-left: 0.5rem;">
                            {item['clean_title']}
                        </span>
                        <span style="color: #64748b; font-size: 0.95rem; font-weight: 600;">({item['year'] or 'N/A'})</span>
                    </div>
                    <div>
                        <span class="badge-match">{item['score_label']}</span>
                        <span class="badge-rating" style="margin-left: 0.4rem;">★ {item['vote_average']} ({item['vote_count']} votes)</span>
                    </div>
                </div>
                <div style="margin-top: 0.6rem;">{genre_tags}</div>
            </div>
            """, unsafe_allow_html=True)

else:
    # Top-Rated by Genre mode
    st.markdown("### 🏆 Top-Rated Movies by Genre (IMDb Formula)")
    selected_genre = st.selectbox("Select a Genre to Explore:", ["All Genres"] + all_genres, index=0)
    min_votes_slider = st.slider("Minimum Number of User Ratings:", min_value=5, max_value=80, value=25, step=5)

    recommendations = get_top_rated_by_genre(selected_genre, top_n=num_recs, min_votes=min_votes_slider)

    st.caption(f"Ranked by Bayesian Weighted Rating with minimum {min_votes_slider} reviews.")
    for rank, item in enumerate(recommendations, start=1):
        genre_tags = " ".join([f"<span class='badge-genre'>{g}</span>" for g in str(item['genres']).split('|')])
        st.markdown(f"""
        <div class="movie-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 0.5rem;">
                <div>
                    <span class="badge-rank">#{rank}</span>
                    <span style="font-size: 1.2rem; font-weight: 800; color: #0f172a; margin-left: 0.5rem;">
                        {item['clean_title']}
                    </span>
                    <span style="color: #64748b; font-size: 0.95rem; font-weight: 600;">({item['year'] or 'N/A'})</span>
                </div>
                <div>
                    <span class="badge-match">{item['score_label']}</span>
                    <span class="badge-rating" style="margin-left: 0.4rem;">Avg: ★ {item['vote_average']} ({item['vote_count']} votes)</span>
                </div>
            </div>
            <div style="margin-top: 0.6rem;">{genre_tags}</div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Educational & Technical Tabs
# ---------------------------------------------------------
st.markdown("---")
tab_algo, tab_limit, tab_data = st.tabs(["📐 Algorithmic Foundations", "⚠️ Limitations & Edge Cases", "📚 Dataset & Benchmark"])

with tab_algo:
    st.markdown("""
    ### Recommendation Algorithms Implemented
    
    #### 1. Content-Based Filtering
    * **Representation**: Constructs a combined text profile from movie titles and genre tags.
    * **Feature Extraction**: TF-IDF (Term Frequency - Inverse Document Frequency) maps documents into vector space $\mathbb{R}^{V}$.
    * **Similarity Metric**: Cosine Similarity between vector $u$ and $v$:
      $$\text{Cosine Similarity}(u, v) = \frac{u \cdot v}{\|u\| \|v\|}$$
    
    #### 2. Collaborative Filtering (Matrix Factorization)
    * **Concept**: Discovers latent representations of movies based solely on how users rate them.
    * **Decomposition**: Truncated Singular Value Decomposition (SVD):
      $$R \approx U \cdot \Sigma \cdot V^T$$
      where $V^T$ yields a compact 25-dimensional latent feature vector for each movie.
    * **Inference**: High cosine similarity between movie latent vectors indicates that users who liked movie A consistently liked movie B, enabling cross-genre discoveries.

    #### 3. IMDb Bayesian Weighted Rating
    * Mitigates small-sample bias (e.g. 1 review of 5.0 vs 200 reviews of 4.5):
      $$\text{WR} = \left(\frac{v}{v+m}\right) R + \left(\frac{m}{v+m}\right) C$$
      where $v$ is vote count, $m$ is the threshold percentile, $R$ is movie average, and $C$ is catalog mean.
    """)

with tab_limit:
    st.markdown("""
    ### Honest System Limitations & Trade-Offs
    
    1. **Cold-Start Problem**:
       - *New Movies*: Movies with zero or few ratings cannot be recommended by Collaborative Filtering. They require Content-Based Fallback.
       - *New Users*: A brand-new user with no rating history must rely on genre selection or popular top-rated titles.
    
    2. **Extreme Matrix Sparsity**:
       - In MovieLens, 98.3% of all possible user-movie interaction pairs are empty. While SVD mitigates this, sparse rating records can limit precision for niche indie films.
    
    3. **Popularity Bias**:
       - Blockbuster movies (e.g. *Pulp Fiction*, *The Shawshank Redemption*, *Forrest Gump*) have thousands of ratings and appear more frequently in latent factor neighborhoods than lesser-known films.
    
    4. **Absence of Deep Narrative Semantics**:
       - The current dataset relies on title tokens and genre tags. Detailed plot synopses, directors, actors, and cinematography styles are not included in `ml-latest-small`.
    """)

with tab_data:
    st.markdown("""
    ### MovieLens Latest Small Dataset
    * **Curated by**: GroupLens Research, Department of Computer Science and Engineering at University of Minnesota.
    * **Users**: 610 registered users.
    * **Movies**: 9,742 cataloged films.
    * **Ratings**: 100,836 individual ratings on a 0.5 to 5.0 star scale.
    * **License**: Open research and educational use.
    """)
