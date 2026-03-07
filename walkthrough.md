# Fake Review Detection System - Walkthrough

## Project Structure

```
f:\Fake_Review_Detection\
+-- data/
|   +-- __init__.py
|   +-- generate_dataset.py         # Generates 2000 synthetic reviews
+-- models/                          # Created after training
+-- modules/
|   +-- __init__.py
|   +-- preprocessing.py             # NLP text preprocessing (NLTK w/ fallback)
|   +-- feature_extraction.py        # TF-IDF + length + sentiment features
|   +-- sentiment_analysis.py        # VADER sentiment analysis
|   +-- behavior_analysis.py         # Suspicious reviewer detection
|   +-- explainability.py            # SHAP-based explanations
+-- training/
|   +-- __init__.py
|   +-- train_models.py              # Trains 5 models, auto-selects best
+-- app/
|   +-- app.py                       # Flask web application
|   +-- static/style.css             # Premium dark theme
|   +-- templates/
|       +-- index.html               # Review input form
|       +-- result.html              # Prediction + explanation
|       +-- dashboard.html           # Analytics dashboard (4 charts)
+-- requirements.txt
+-- README.md
```

## What Was Built

| Component | Details |
|---|---|
| **Preprocessing** | Lowercase, punctuation removal, tokenization, stopwords, lemmatization (NLTK with graceful fallback) |
| **Features** | TF-IDF (unigrams+bigrams), review length, VADER sentiment score |
| **Models** | Logistic Regression, Random Forest, SVM, LSTM, BERT |
| **Best Model Selection** | Auto-selects by F1 score, saves as `model.pkl` |
| **Sentiment** | VADER analyzer with compound/pos/neg/neu scores |
| **Behavior Analysis** | Frequency, text duplication, extreme rating detection |
| **Explainability** | SHAP LinearExplainer for top feature attribution |
| **Web App** | Flask with prediction, confidence, SHAP table, sentiment breakdown |
| **Dashboard** | Fake/genuine distribution, sentiment chart, top words, review length distribution, model comparison table |

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train models (run in your local terminal)
```bash
cd f:\Fake_Review_Detection
python training/train_models.py
```

This generates the dataset, trains all 5 models, auto-selects the best, and saves:
- `models/model.pkl` (best model)
- `models/vectorizer.pkl` (TF-IDF vectorizer)
- Individual model files (logistic_regression.pkl, random_forest.pkl, svm.pkl, etc.)

### 3. Run the Flask app
```bash
python app/app.py
```
Visit **http://localhost:5000**

## Key Design Decisions

- **Auto-select best model**: Training compares all models by F1 score and saves the winner as `model.pkl`
- **NLTK fallback**: If NLTK data can't download, preprocessing falls back to `str.split()` tokenization and a built-in stopword list
- **SVM confidence**: Since LinearSVC lacks `predict_proba`, the app converts its `decision_function` output via sigmoid for a confidence percentage
- **LSTM/BERT graceful skip**: If TensorFlow or PyTorch aren't installed, those models are skipped without failing the pipeline
