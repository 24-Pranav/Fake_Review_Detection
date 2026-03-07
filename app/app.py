"""
Flask Web Application - Fake Review Detection
----------------------------------------------
Routes:
  GET  /           -> Review input form
  POST /predict    -> Prediction result + explanation
  GET  /dashboard  -> Analytics dashboard
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
from flask import Flask, render_template, request

from modules.preprocessing import preprocess_text
from modules.sentiment_analysis import get_sentiment, get_sentiment_label
from modules.explainability import explain_prediction

app = Flask(__name__)

# ── Load Model & Vectorizer ───────────────────────────────────────────────
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

model = joblib.load(os.path.join(MODELS_DIR, "model.pkl"))
vectorizer = joblib.load(os.path.join(MODELS_DIR, "vectorizer.pkl"))
print("[OK] Model (model.pkl) and vectorizer (vectorizer.pkl) loaded")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    review_text = request.form.get("review_text", "").strip()
    if not review_text:
        return render_template("index.html", error="Please enter a review.")

    # Preprocess
    clean = preprocess_text(review_text)

    # Feature vector
    from modules.feature_extraction import build_features
    X, _ = build_features([clean], fit=False, vectorizer=vectorizer)

    # Predict
    prediction = model.predict(X)[0]
    label = "FAKE" if prediction == 1 else "GENUINE"

    # Confidence
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = round(max(proba) * 100, 1)
    elif hasattr(model, "decision_function"):
        decision = model.decision_function(X)[0]
        # Convert SVM decision function to pseudo-probability using sigmoid
        prob = 1 / (1 + np.exp(-decision))
        confidence = round(max(prob, 1 - prob) * 100, 1)
    else:
        confidence = None

    # Sentiment
    sentiment_scores = get_sentiment(review_text)
    sentiment_label = get_sentiment_label(review_text)

    # SHAP explanation
    shap_features = explain_prediction(model, vectorizer, clean, top_n=8)

    return render_template(
        "result.html",
        review_text=review_text,
        label=label,
        confidence=confidence,
        sentiment_scores=sentiment_scores,
        sentiment_label=sentiment_label,
        shap_features=shap_features,
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
