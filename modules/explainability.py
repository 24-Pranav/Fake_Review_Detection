"""
Explainable AI Module (SHAP)
────────────────────────────
Uses SHAP to explain which features/words contributed
most to the fake-review prediction.
"""

import numpy as np
import shap


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
        tfidf_vector = vectorizer.transform([preprocessed_text])

        # Use a SHAP LinearExplainer for speed
        explainer = shap.LinearExplainer(model, tfidf_vector, feature_perturbation="interventional")
        shap_values = explainer.shap_values(tfidf_vector)

        # Get feature names
        feature_names = vectorizer.get_feature_names_out()

        # shap_values shape: (1, n_features)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]  # class 1 = fake
        else:
            sv = shap_values[0]

        # Only consider TF-IDF features (not length/sentiment appended later)
        n_tfidf = len(feature_names)
        sv = sv[:n_tfidf]

        # Top features by absolute SHAP value
        top_indices = np.argsort(np.abs(sv))[-top_n:][::-1]

        results = []
        for idx in top_indices:
            if sv[idx] != 0:
                results.append({
                    "feature": feature_names[idx],
                    "shap_value": round(float(sv[idx]), 4),
                })

        return results

    except Exception as e:
        print(f"[!] SHAP explanation error: {e}")
        return []
