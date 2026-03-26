"""
Explainable AI Module (SHAP)
----------------------------
Uses SHAP to explain which features/words contributed
most to the fake-review prediction.
"""

import numpy as np
import shap
from scipy.sparse import hstack, csr_matrix
from modules.sentiment_analysis import get_sentiment_compound


def explain_prediction(model, vectorizer, preprocessed_text: str, top_n: int = 10) -> list:
    """
    Explain a single prediction using SHAP LinearExplainer.

    Parameters
    ----------
    model : trained sklearn linear model (e.g. LogisticRegression)
    vectorizer : fitted TfidfVectorizer
    preprocessed_text : already-preprocessed review text
    top_n : number of top features to return

    Returns
    -------
    list of dicts: [{"feature": word, "shap_value": float}, ...]
    """
    try:
        # Build the FULL feature vector (TF-IDF + length + sentiment)
        # to match what the model was trained on
        tfidf_vector = vectorizer.transform([preprocessed_text])
        length = np.array([[len(preprocessed_text)]])
        sentiment = np.array([[get_sentiment_compound(preprocessed_text)]])
        extra = csr_matrix(np.hstack([length, sentiment]))
        full_vector = hstack([tfidf_vector, extra])

        # Use SHAP LinearExplainer
        explainer = shap.LinearExplainer(model, full_vector, feature_perturbation="interventional")
        shap_values = explainer.shap_values(full_vector)

        # Get feature names (TF-IDF names + extra feature names)
        tfidf_names = list(vectorizer.get_feature_names_out())
        all_names = tfidf_names + ["review_length", "sentiment_score"]

        # shap_values shape: (1, n_features)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]  # class 1 = fake
        else:
            sv = shap_values[0]

        # Top features by absolute SHAP value
        top_indices = np.argsort(np.abs(sv))[-top_n:][::-1]

        results = []
        for idx in top_indices:
            if sv[idx] != 0 and idx < len(all_names):
                results.append({
                    "feature": all_names[idx],
                    "shap_value": round(float(sv[idx]), 4),
                })

        return results

    except Exception as e:
        print(f"[!] SHAP explanation error: {e}")
        return []
