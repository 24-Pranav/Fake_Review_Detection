# Fake Review Detection System - Walkthrough

## Project Structure

```
f:\Fake_Review_Detection\
+-- data/
|   +-- __init__.py
|   +-- generate_dataset.py         # Generates 2000 synthetic reviews
+-- models/                          # Created after training
|   +-- model.pkl                    # Best sklearn model (auto-selected)
|   +-- vectorizer.pkl               # TF-IDF vectorizer
|   +-- lstm_model.h5                # LSTM neural network
|   +-- lstm_tokenizer.pkl           # Keras Tokenizer + padding config
+-- modules/
|   +-- __init__.py
|   +-- preprocessing.py             # NLP text preprocessing (NLTK w/ fallback)
|   +-- feature_extraction.py        # TF-IDF + length + sentiment features
|   +-- sentiment_analysis.py        # VADER sentiment analysis
|   +-- behavior_analysis.py         # Suspicious reviewer + readability detection
|   +-- explainability.py            # SHAP-based explanations
+-- training/
|   +-- __init__.py
|   +-- train_models.py              # Trains 5 models, auto-selects best
+-- app/
|   +-- app.py                       # Flask web application
|   +-- static/style.css             # Premium dark theme
|   +-- templates/
|       +-- index.html               # Review input form
|       +-- result.html              # Prediction + explanation + LSTM
|       +-- dashboard.html           # Analytics dashboard (4 charts)
|       +-- setup.html               # Model-missing setup instructions
+-- requirements.txt
+-- README.md
```

## What Was Built

| Component | Details |
|---|---|
| **Preprocessing** | Lowercase, punctuation removal, tokenization, stopwords, lemmatization (NLTK with graceful fallback) |
| **Features** | TF-IDF (unigrams+bigrams), review length, VADER sentiment score |
| **Models** | Logistic Regression, Random Forest, SVM, LSTM, BERT |
| **LSTM Pipeline** | Keras Tokenizer → pad_sequences → Embedding → LSTM; tokenizer saved as `lstm_tokenizer.pkl` |
| **Best Model Selection** | Auto-selects by F1 score, saves as `model.pkl` |
| **Sentiment** | VADER analyzer with compound/pos/neg/neu scores |
| **Behavior Analysis** | Frequency, text duplication, extreme rating, Flesch-Kincaid readability detection |
| **Explainability** | SHAP LinearExplainer for top feature attribution (sklearn models only) |
| **Web App** | Flask with ensemble prediction, Trust Gauge, SHAP table, LSTM analysis, and Expert Recommendation |
| **Error Handling** | Global model-loading via `safe_load()` try-except; redirects to setup page if models are missing |
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
- `models/lstm_model.h5` (LSTM neural network)
- `models/lstm_tokenizer.pkl` (Keras tokenizer + padding config)
- Individual model files (logistic_regression.pkl, random_forest.pkl, svm.pkl)

### 3. Run the Flask app
```bash
python app/app.py
```
Visit **http://localhost:5000**

## LSTM ↔ SHAP Integration

The SHAP `LinearExplainer` only works with sklearn linear models (LogisticRegression, SVM). Since the LSTM uses a different feature space (token embeddings), it provides its own independent prediction displayed as a secondary signal on the results page. If the LSTM and sklearn models disagree, the UI shows a warning to prompt closer review.

## Key Design Decisions

- **Auto-select best model**: Training compares all models by F1 score and saves the winner as `model.pkl`
- **NLTK fallback**: If NLTK data can't download, preprocessing falls back to `str.split()` tokenization and a built-in stopword list
- **SVM confidence**: Since LinearSVC lacks `predict_proba`, the app converts its `decision_function` output via sigmoid for a confidence percentage
- **LSTM/BERT graceful skip**: If TensorFlow or PyTorch aren't installed, those models are skipped without failing the pipeline
- **Error handling**: If `model.pkl` or `vectorizer.pkl` is missing, the app redirects to a setup page with step-by-step instructions instead of crashing

## Troubleshooting

### Regenerating Models (Corrupted `.pkl` Files)

If model files are corrupted or produce unexpected errors:

```bash
# Step 1: Delete all existing model files
cd f:\Fake_Review_Detection
rmdir /s /q models
mkdir models

# Step 2: Retrain everything from scratch
python training/train_models.py

# Step 3: Restart the Flask app
python app/app.py
```

### Common Errors

| Error | Cause | Fix |
|---|---|---|
| `FileNotFoundError: model.pkl` | Models not trained yet | Run `python training/train_models.py` |
| App redirects to `/setup` page | `model.pkl` or `vectorizer.pkl` missing/corrupted | Delete `models/` folder and retrain |
| `ModuleNotFoundError: tensorflow` | TensorFlow not installed | Run `pip install tensorflow==2.15.0` (LSTM will be skipped if missing) |
| `ModuleNotFoundError: textstat` | textstat not installed | Run `pip install textstat==0.7.3` |
| NLTK download errors | Network issues during preprocessing | NLTK falls back to built-in tokenizer automatically |
| `UnpicklingError` on any `.pkl` file | File corrupted (partial write, version mismatch) | Delete the specific `.pkl` file and retrain |
| LSTM results missing from dashboard | TensorFlow wasn't available during training | Install TensorFlow, delete `models/`, and retrain |

### Verifying Model Integrity

To check if model files are loadable:

```python
import joblib, pickle

# Test sklearn model
model = joblib.load("models/model.pkl")
print(f"Model type: {type(model).__name__}")

# Test vectorizer
vec = joblib.load("models/vectorizer.pkl")
print(f"Vectorizer features: {len(vec.get_feature_names_out())}")

# Test LSTM tokenizer
with open("models/lstm_tokenizer.pkl", "rb") as f:
    config = pickle.load(f)
print(f"LSTM vocab size: {config['vocab_size']}, max_len: {config['max_len']}")
```

