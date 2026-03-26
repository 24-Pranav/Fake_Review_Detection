"""
Flask Web Application - Fake Review Detection
----------------------------------------------
Routes:
  GET  /           -> Review input form
  POST /predict    -> Prediction result + explanation
  GET  /dashboard  -> Analytics dashboard
  GET  /setup      -> Setup instructions (when models missing)
"""

import os
import sys

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
from collections import Counter
from flask import Flask, render_template, request, redirect, url_for

from modules.preprocessing import preprocess_text
from modules.sentiment_analysis import get_sentiment, get_sentiment_label
from modules.explainability import explain_prediction

app = Flask(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# ── Global model state ────────────────────────────────────────────────────
sklearn_model = None
vectorizer = None
lstm_model = None
lstm_tokenizer_config = None
all_sklearn_models = {}      # name -> model object for ensemble
model_loaded = False
lstm_loaded = False
missing_files = []


# ──────────────────────────────────────────────────────────────────────────
#  safe_load() — prevents crashes when .pkl / .h5 files are missing
# ──────────────────────────────────────────────────────────────────────────
def safe_load(path, loader="joblib"):
    """
    Safely load a model file. Returns None instead of crashing.

    Parameters
    ----------
    path : str – absolute path to the model file
    loader : str – 'joblib', 'pickle', or 'keras'
    """
    try:
        if not os.path.exists(path):
            print(f"[!] File not found: {os.path.basename(path)}")
            return None

        if loader == "joblib":
            return joblib.load(path)
        elif loader == "pickle":
            import pickle
            with open(path, "rb") as f:
                return pickle.load(f)
        elif loader == "keras":
            os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
            import tensorflow as tf
            tf.get_logger().setLevel("ERROR")
            return tf.keras.models.load_model(path)
        else:
            raise ValueError(f"Unknown loader: {loader}")
    except ImportError:
        print(f"[!] Required library not installed for {os.path.basename(path)}")
        return None
    except Exception as e:
        print(f"[!] Error loading {os.path.basename(path)}: {e}")
        return None


def load_models():
    """Load all models using safe_load(). Sets global flags."""
    global sklearn_model, vectorizer, lstm_model, lstm_tokenizer_config
    global all_sklearn_models, model_loaded, lstm_loaded, missing_files
    missing_files = []

    # ── Primary sklearn model + vectorizer ────────────────────────────
    sklearn_model = safe_load(os.path.join(MODELS_DIR, "model.pkl"))
    vectorizer = safe_load(os.path.join(MODELS_DIR, "vectorizer.pkl"))

    if sklearn_model is None:
        missing_files.append("model.pkl")
    if vectorizer is None:
        missing_files.append("vectorizer.pkl")

    model_loaded = sklearn_model is not None and vectorizer is not None
    if model_loaded:
        print("[OK] Model (model.pkl) and vectorizer (vectorizer.pkl) loaded")

    # ── Load ALL sklearn models for ensemble ──────────────────────────
    for name, fname in [("Logistic Regression", "logistic_regression.pkl"),
                        ("Random Forest", "random_forest.pkl"),
                        ("SVM", "svm.pkl")]:
        m = safe_load(os.path.join(MODELS_DIR, fname))
        if m is not None:
            all_sklearn_models[name] = m

    if all_sklearn_models:
        print(f"[OK] Loaded {len(all_sklearn_models)} sklearn models for ensemble")

    # ── LSTM model + tokenizer ────────────────────────────────────────
    lstm_path = os.path.join(MODELS_DIR, "lstm_model.h5")
    if not os.path.exists(lstm_path):
        lstm_path = os.path.join(MODELS_DIR, "lstm_model.keras")

    lstm_model = safe_load(lstm_path, loader="keras")
    lstm_tokenizer_config = safe_load(
        os.path.join(MODELS_DIR, "lstm_tokenizer.pkl"), loader="pickle"
    )

    lstm_loaded = lstm_model is not None and lstm_tokenizer_config is not None
    if lstm_loaded:
        print("[OK] LSTM model and tokenizer loaded")
    else:
        print("[!] LSTM inference disabled (model or tokenizer missing)")


# Load models on startup
load_models()


# ──────────────────────────────────────────────────────────────────────────
#  Routes
# ──────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    if not model_loaded:
        return redirect(url_for("setup"))
    return render_template("index.html")


@app.route("/setup")
def setup():
    return render_template(
        "setup.html",
        missing_files=missing_files,
        project_root=PROJECT_ROOT,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if not model_loaded:
        return redirect(url_for("setup"))

    review_text = request.form.get("review_text", "").strip()
    if not review_text:
        return render_template("index.html", error="Please enter a review.")

    # Preprocess
    clean = preprocess_text(review_text)

    # ── Feature vector (shared across sklearn models) ─────────────────
    from modules.feature_extraction import build_features
    X, _ = build_features([clean], fit=False, vectorizer=vectorizer)

    # ── Primary sklearn prediction ────────────────────────────────────
    prediction = sklearn_model.predict(X)[0]
    label = "FAKE" if prediction == 1 else "GENUINE"

    # Confidence
    if hasattr(sklearn_model, "predict_proba"):
        proba = sklearn_model.predict_proba(X)[0]
        confidence = round(max(proba) * 100, 1)
    elif hasattr(sklearn_model, "decision_function"):
        decision = sklearn_model.decision_function(X)[0]
        prob = 1 / (1 + np.exp(-decision))
        confidence = round(max(prob, 1 - prob) * 100, 1)
    else:
        confidence = None

    # Sentiment
    sentiment_scores = get_sentiment(review_text)
    sentiment_label = get_sentiment_label(review_text)

    # SHAP explanation
    shap_features = explain_prediction(sklearn_model, vectorizer, clean, top_n=8)

    # ── Ensemble predictions (all sklearn models) ─────────────────────
    ensemble_votes = {}  # model_name -> "FAKE" | "GENUINE"
    for name, mdl in all_sklearn_models.items():
        try:
            pred = mdl.predict(X)[0]
            ensemble_votes[name] = "FAKE" if pred == 1 else "GENUINE"
        except Exception:
            pass

    # ── LSTM prediction ───────────────────────────────────────────────
    lstm_label = None
    lstm_confidence = None
    if lstm_loaded and lstm_model is not None and lstm_tokenizer_config is not None:
        try:
            from tensorflow.keras.preprocessing.sequence import pad_sequences
            tok = lstm_tokenizer_config["tokenizer"]
            max_len = lstm_tokenizer_config["max_len"]

            seq = pad_sequences(
                tok.texts_to_sequences([clean]),
                maxlen=max_len,
                padding="post",
            )
            lstm_prob = float(lstm_model.predict(seq, verbose=0)[0][0])
            lstm_label = "FAKE" if lstm_prob > 0.5 else "GENUINE"
            lstm_confidence = round(max(lstm_prob, 1 - lstm_prob) * 100, 1)
            ensemble_votes["LSTM"] = lstm_label
        except Exception as e:
            print(f"[!] LSTM inference error: {e}")

    # ── Ensemble Trust Gauge (consensus score) ────────────────────────
    total_models = len(ensemble_votes)
    if total_models > 0:
        fake_votes = sum(1 for v in ensemble_votes.values() if v == "FAKE")
        genuine_votes = total_models - fake_votes
        consensus_pct = round(max(fake_votes, genuine_votes) / total_models * 100)
        consensus_label = "FAKE" if fake_votes > genuine_votes else "GENUINE"
    else:
        consensus_pct = confidence if confidence else 50
        consensus_label = label

    # ── Readability flag ──────────────────────────────────────────────
    readability_flag = None
    try:
        from modules.behavior_analysis import calculate_readability
        r = calculate_readability(review_text)
        if r.get("is_bot_generated"):
            readability_flag = r.get("bot_reason", "Suspicious readability pattern detected.")
        elif r.get("is_unnaturally_consistent"):
            readability_flag = "Readability is unnaturally consistent — possible AI-generated text."
    except Exception:
        pass

    # ── Expert Recommendation (AI Advisory Layer) ─────────────────────
    try:
        results_df = pd.read_csv(os.path.join(DATA_DIR, "model_results.csv"))
        top_model_name = results_df.sort_values(by="f1", ascending=False).iloc[0]["model"]
    except Exception:
        top_model_name = "Primary Model"

    word_count = len(review_text.split())
    if word_count < 10:
        recommendation = f"For short reviews, SVM excels at precision. Trust {top_model_name} for this case."
        rec_model = "SVM"
    elif word_count > 50:
        recommendation = "For longer reviews, the LSTM model captures sequential context best. Cross-reference its verdict."
        rec_model = "LSTM"
    elif sentiment_scores["compound"] > 0.8 or sentiment_scores["compound"] < -0.8:
        recommendation = "This review has extreme sentiment — the LSTM model is best for detecting emotional manipulation patterns."
        rec_model = "LSTM"
    else:
        recommendation = f"The {top_model_name} is the most balanced expert for this type of review (F1-Score leader)."
        rec_model = top_model_name

    return render_template(
        "result.html",
        review_text=review_text,
        label=label,
        confidence=confidence,
        sentiment_scores=sentiment_scores,
        sentiment_label=sentiment_label,
        shap_features=shap_features,
        lstm_label=lstm_label,
        lstm_confidence=lstm_confidence,
        ensemble_votes=ensemble_votes,
        consensus_pct=consensus_pct,
        consensus_label=consensus_label,
        readability_flag=readability_flag,
        recommendation=recommendation,
        rec_model=rec_model,
    )


@app.route("/dashboard")
def dashboard():
    csv_path = os.path.join(DATA_DIR, "reviews.csv")
    if not os.path.exists(csv_path):
        return render_template("dashboard.html", error="No dataset found. Run training first.")

    df = pd.read_csv(csv_path)

    # ── Fake vs Genuine distribution ──────────────────────────────────
    genuine_count = int((df["label"] == 0).sum())
    fake_count = int((df["label"] == 1).sum())

    # ── Sentiment distribution ────────────────────────────────────────
    df["sentiment"] = df["review_text"].apply(get_sentiment_label)
    sentiment_counts = df["sentiment"].value_counts().to_dict()
    sent_labels = list(sentiment_counts.keys())
    sent_values = list(sentiment_counts.values())

    # ── Top words in fake reviews ─────────────────────────────────────
    fake_reviews = df[df["label"] == 1]["review_text"]
    words = " ".join(fake_reviews).lower().split()
    stop = {"the", "a", "an", "is", "it", "to", "and", "of", "in", "for",
            "this", "i", "my", "was", "that", "not", "on", "with", "have",
            "be", "are", "so", "but", "you", "do", "at", "or", "if", "has"}
    words = [w for w in words if w not in stop and len(w) > 2]
    top_words = Counter(words).most_common(15)
    tw_labels = [w[0] for w in top_words]
    tw_values = [w[1] for w in top_words]

    # ── Review length distribution ────────────────────────────────────
    df["review_length"] = df["review_text"].apply(len)
    bins = [0, 50, 100, 150, 200, 300, 500, 1000]
    bin_labels = ["0-50", "51-100", "101-150", "151-200", "201-300", "301-500", "500+"]
    df["len_bin"] = pd.cut(df["review_length"], bins=bins, labels=bin_labels, right=True)
    len_counts = df["len_bin"].value_counts().sort_index().to_dict()
    rl_labels = list(len_counts.keys())
    rl_values = [int(v) for v in len_counts.values()]

    # ── Model results ─────────────────────────────────────────────────
    results_path = os.path.join(DATA_DIR, "model_results.csv")
    model_results = []
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        model_results = results_df.to_dict("records")

    return render_template(
        "dashboard.html",
        genuine_count=genuine_count,
        fake_count=fake_count,
        sent_labels=sent_labels,
        sent_values=sent_values,
        tw_labels=tw_labels,
        tw_values=tw_values,
        rl_labels=rl_labels,
        rl_values=rl_values,
        model_results=model_results,
        total_reviews=len(df),
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
