"""
Feature Extraction Module
─────────────────────────
Combines:
  • TF-IDF (unigrams + bigrams)
  • Review length (character count)
  • Sentiment score (VADER compound)
"""

import os
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from modules.sentiment_analysis import get_sentiment_compound

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")


def build_features(texts, fit=True, vectorizer=None):
    """
    Build a combined feature matrix from preprocessed text.

    Parameters
    ----------
    texts : array-like of preprocessed strings
    fit : bool – if True, fit a new TfidfVectorizer; if False, use `vectorizer`
    vectorizer : fitted TfidfVectorizer (required when fit=False)

    Returns
    -------
    (feature_matrix, vectorizer)
    """
    os.makedirs(MODELS_DIR, exist_ok=True)

    # ── TF-IDF (unigrams + bigrams) ───────────────────────────────────
    if fit:
        vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        tfidf_matrix = vectorizer.fit_transform(texts)
        joblib.dump(vectorizer, os.path.join(MODELS_DIR, "vectorizer.pkl"))
        print("[OK] TF-IDF vectorizer saved as vectorizer.pkl")
    else:
        tfidf_matrix = vectorizer.transform(texts)

    # ── Review length ──────────────────────────────────────────────────
    lengths = np.array([len(str(t)) for t in texts]).reshape(-1, 1)

    # ── Sentiment score ────────────────────────────────────────────────
    sentiments = np.array([get_sentiment_compound(str(t)) for t in texts]).reshape(-1, 1)

    # ── Combine features ───────────────────────────────────────────────
    extra = csr_matrix(np.hstack([lengths, sentiments]))
    feature_matrix = hstack([tfidf_matrix, extra])

    return feature_matrix, vectorizer
