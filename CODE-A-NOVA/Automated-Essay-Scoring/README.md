# ✍️ Automated Essay Scoring (AES) System

An end-to-end Machine Learning application that automatically evaluates student essays based on linguistic structure, lexical diversity, readability indices, and topical content representations using the benchmark **Hewlett Foundation Automated Student Assessment Prize (ASAP-AES)** dataset.

---

## 📌 Project Overview

Manual essay scoring is labor-intensive, time-consuming, and prone to human subjective variance. This project implements a regression-based Machine Learning pipeline to predict essay scores on a standardized **1.0 to 10.0 scale** with high agreement with human raters, paired with an interactive **Streamlit** user interface offering actionable diagnostic writing feedback.

---

## 🗂️ Dataset Details

* **Dataset**: Automated Student Assessment Prize (ASAP-AES)
* **Source**: The William and Flora Hewlett Foundation (Kaggle AES Benchmark)
* **Total Samples**: 12,976 essays across 8 diverse prompt domains (persuasive, expository, narrative).
* **Target Normalization**: Original prompt-specific raw scores (e.g. 2–12, 1–6, 10–60) are mathematically normalized to a uniform continuous scale:
  $$\text{Normalized Score} = \left(\frac{\text{raw\_score} - s_{\min}}{s_{\max} - s_{\min}}\right) \times 9.0 + 1.0$$
* **Auto-download**: The training script `train_model.py` automatically fetches the benchmark dataset if not already present locally.

---

## 🔬 Feature Engineering Pipeline

The model combines domain-specific linguistic features with statistical NLP representations:

1. **Surface Structure & Volume**:
   * Total Word Count, Character Count, Sentence Count, Paragraph Count.
   * Average Word Length and Average Sentence Length.
2. **Lexical Diversity**:
   * Type-Token Ratio (TTR): Ratio of unique vocabulary to total words.
3. **Readability & Complexity**:
   * **Flesch Reading Ease**: Measures text accessibility based on syllables per word and sentence length:
     $$\text{FRE} = 206.835 - 1.015 \left(\frac{\text{words}}{\text{sentences}}\right) - 84.6 \left(\frac{\text{syllables}}{\text{words}}\right)$$
   * **Automated Readability Index (ARI)**: Approximates US educational grade level.
4. **Syntactic Complexity Proxies**:
   * Density of structural punctuation (commas, semicolons, dashes, quotes).
   * Emphasis markers (exclamations, question marks).
5. **Content & Vocabulary (TF-IDF)**:
   * Sublinear TF-IDF vectorizer (max 400 vocabulary features, unigrams and bigrams, English stopword filtration).

All numerical features are standardized using `StandardScaler` and stacked with the sparse TF-IDF feature matrix.

---

## 📊 Model Evaluation & Results

Evaluated on a **20% holdout test split (2,595 unseen essays)** using standard regression metrics and the industry-standard **Quadratic Weighted Kappa (QWK)**:

| Model | MAE (Mean Absolute Error) | RMSE (Root Mean Squared Error) | $R^2$ Score | QWK (Quadratic Weighted Kappa) |
|---|---|---|---|---|
| **Ridge Regression** | 1.0934 | 1.4048 | 0.5743 | 0.7123 |
| **Random Forest Regressor** | 1.0281 | 1.3568 | 0.6029 | 0.7349 |
| **Gradient Boosting Regressor (Winner)** | **1.0112** | **1.3126** | **0.6283** | **0.7563** |

* **Winner**: `Gradient Boosting Regressor` achieved the lowest MAE (**1.0112**) and a QWK of **0.7563**, representing substantial agreement with human expert scorers.

---

## 🚀 Installation & Running

### 1. Prerequisites
Ensure Python 3.10+ is installed.

### 2. Install Dependencies
Navigate to this project directory:
```bash
cd Automated-Essay-Scoring
pip install -r requirements.txt
```

### 3. Train the Model (Optional - Pretrained weights included)
To re-run the full training, feature extraction, and benchmark generation:
```bash
python train_model.py
```
This will:
* Download and clean the ASAP dataset in `data/`.
* Train and evaluate all 3 regression models on a holdout split.
* Save the best model artifact to `models/essay_scoring_model.pkl`.
* Output `models/evaluation_metrics.json` and `data/sample_essays.json`.

### 4. Launch the Streamlit Application
```bash
streamlit run app.py
```
The app will open automatically in your browser at `http://localhost:8501`.

---

## 💡 Application Features

* **Instant Essay Scoring**: Enter any essay text or load high/medium/low samples from the sidebar.
* **Predicted Score Display**: Score on a 1.0–10.0 scale, accompanied by letter grade (A+, A, B, C, D-F) and proficiency tier.
* **Writing Dimensions Breakdown**: Cards visualizing Word Count, Lexical Diversity (TTR), Flesch Reading Ease, and ARI Grade Level.
* **Actionable Diagnostic Feedback**: Automatic warnings for run-on sentences, repetitive vocabulary, or insufficient elaboration.
* **Academic Transparency Tab**: Fully discloses system limitations.

---

## ⚠️ System Limitations & Ethical Disclosure

1. **No Rule-Based Grammar Parser**: The system extracts structural, lexical, and readability heuristics. It does not integrate a comprehensive rule-based syntax dependency grammar checker.
2. **Semantic & Fact-Checking Limitations**: The model recognizes vocabulary relevance via TF-IDF n-grams; it cannot verify logical fallacies or factual assertions.
3. **Length Sensitivity**: Like most statistical NLP models, word length correlates positively with predicted score.
4. **Intended Use**: For formative educational feedback; not intended for unmonitored high-stakes grading.
