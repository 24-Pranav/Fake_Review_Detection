# Fake Review Detection System — Implementation Plan

Build an end-to-end AI-based Fake Review Detection System with ML/DL models, explainability, a Flask web app, and an analytics dashboard.

## Proposed Changes

### Project Structure

```
f:\Fake_Review_Detection\
├── data/
│   └── generate_dataset.py        # Synthetic dataset generator
├── models/                         # Saved trained models (auto-created)
├── modules/
│   ├── preprocessing.py            # NLP text preprocessing
│   ├── feature_extraction.py       # TF-IDF, n-grams, length, sentiment
│   ├── sentiment_analysis.py       # VADER sentiment analysis
│   ├── behavior_analysis.py        # Reviewer behavior patterns
│   └── explainability.py           # SHAP-based feature explanations
├── training/
│   └── train_models.py             # Train all 5 models & save them
├── app/
│   ├── app.py                      # Flask application
│   ├── static/
│   │   └── style.css               # Dashboard & form styling
│   └── templates/
│       ├── index.html              # Review input form
│       ├── result.html             # Prediction + explanation
│       └── dashboard.html          # Analytics dashboard
├── requirements.txt
└── README.md
```

---

### Dataset Generation

#### [NEW] [generate_dataset.py](file:///f:/Fake_Review_Detection/data/generate_dataset.py)
- Generate a ~2000-row CSV with columns: `review_text`, `rating`, `label` (0=genuine, 1=fake), `reviewer_id`, `timestamp`
- Use realistic templates— genuine reviews vary in tone and length; fake reviews use hyperbolic phrases, repetition, and extreme ratings

---

### Core Modules

#### [NEW] [preprocessing.py](file:///f:/Fake_Review_Detection/modules/preprocessing.py)
- `preprocess_text(text)` — lowercase, strip punctuation, tokenize, remove stopwords, lemmatize
- Uses NLTK (`WordNetLemmatizer`, `stopwords`, `word_tokenize`)

#### [NEW] [feature_extraction.py](file:///f:/Fake_Review_Detection/modules/feature_extraction.py)
- `build_features(df)` — returns combined feature matrix:
  - TF-IDF (unigrams + bigrams)
  - Review length (char count)
  - Sentiment score (compound via VADER)
- Saves fitted `TfidfVectorizer` to `models/tfidf_vectorizer.pkl`

#### [NEW] [sentiment_analysis.py](file:///f:/Fake_Review_Detection/modules/sentiment_analysis.py)
- `get_sentiment(text)` → dict with `neg`, `neu`, `pos`, `compound`
- `get_sentiment_label(text)` → "Positive" / "Negative" / "Neutral"
- Uses `vaderSentiment.SentimentIntensityAnalyzer`

#### [NEW] [behavior_analysis.py](file:///f:/Fake_Review_Detection/modules/behavior_analysis.py)
- `analyze_reviewer_behavior(df)` → DataFrame with per-reviewer flags:
  - `high_frequency` — more reviews than mean + 2 std
  - `repeated_text` — duplicate review texts
  - `extreme_ratings` — >70% of reviews are rating 1 or 5

#### [NEW] [explainability.py](file:///f:/Fake_Review_Detection/modules/explainability.py)
- `explain_prediction(model, vectorizer, text)` → dict of top feature words and their SHAP values
- Uses `shap.LinearExplainer` (for Logistic Regression) for fast explanations

---

### Training Pipeline

#### [NEW] [train_models.py](file:///f:/Fake_Review_Detection/training/train_models.py)
- Load CSV, preprocess, extract features
- 80/20 train-test split (`sklearn.model_selection.train_test_split`)
- Train: Logistic Regression, Random Forest, SVM, LSTM, BERT
- Evaluate each with accuracy, precision, recall, F1, confusion matrix
- Print metrics table
- Save models to `models/` dir (`.pkl` for sklearn, `.h5`/`.pt` for DL)

> [!IMPORTANT]
> LSTM will use a Keras `Sequential` model on padded TF-IDF vectors. BERT will use `transformers` `BertForSequenceClassification` fine-tuned for 2 epochs (lightweight, to keep training feasible).

---

### Flask Web Application

#### [NEW] [app.py](file:///f:/Fake_Review_Detection/app/app.py)
- Loads Logistic Regression model + TF-IDF vectorizer on startup
- Routes:
  - `GET /` — render input form
  - `POST /predict` — preprocess input, predict, explain, render result
  - `GET /dashboard` — render analytics with charts
- Passes: prediction label, confidence %, top SHAP features, sentiment

#### [NEW] [index.html](file:///f:/Fake_Review_Detection/app/templates/index.html)
- Text area for review input, submit button

#### [NEW] [result.html](file:///f:/Fake_Review_Detection/app/templates/result.html)
- Shows: Fake/Genuine badge, confidence score bar, sentiment, SHAP feature table

#### [NEW] [dashboard.html](file:///f:/Fake_Review_Detection/app/templates/dashboard.html)
- Chart.js charts: fake vs genuine distribution (pie), sentiment breakdown (bar), top words in fake reviews (horizontal bar)

#### [NEW] [style.css](file:///f:/Fake_Review_Detection/app/static/style.css)
- Dark-themed, modern UI with gradient accents, cards, and animations

---

### Supporting Files

#### [NEW] [requirements.txt](file:///f:/Fake_Review_Detection/requirements.txt)
- All dependencies: flask, scikit-learn, nltk, vaderSentiment, shap, tensorflow, transformers, torch, pandas, numpy, matplotlib, joblib

#### [NEW] [README.md](file:///f:/Fake_Review_Detection/README.md)
- Setup and usage instructions

---

## Verification Plan

### Automated Tests
1. **Training pipeline**: Run `python training/train_models.py` — verify it completes without errors, prints evaluation metrics, and saves model files to `models/`
2. **Flask app**: Run `python app/app.py` — navigate to `http://localhost:5000`, submit a test review, verify prediction and dashboard pages render

### Manual Verification
- After training, confirm `models/` directory contains: `logistic_regression.pkl`, `random_forest.pkl`, `svm.pkl`, `lstm_model.h5`, `bert_model/`, `tfidf_vectorizer.pkl`
- Submit both genuine-sounding and fake-sounding reviews in the web app and verify reasonable predictions
