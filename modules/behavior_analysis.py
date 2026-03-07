"""
Reviewer Behaviour Analysis Module
───────────────────────────────────
Detects suspicious patterns:
  • High review frequency
  • Repeated / duplicate review text
  • Extreme rating patterns
"""

import pandas as pd
import numpy as np


def analyze_reviewer_behavior(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze reviewer behavior and return a per-reviewer summary
    with suspicious-pattern flags.

    Parameters
    ----------
    df : DataFrame with columns [reviewer_id, review_text, rating]

    Returns
    -------
    DataFrame with columns:
        reviewer_id, review_count, high_frequency,
        duplicate_count, has_repeated_text,
        extreme_ratio, extreme_ratings,
        is_suspicious
    """
    results = []

    grouped = df.groupby("reviewer_id")
    review_counts = grouped.size()
    mean_count = review_counts.mean()
    std_count = review_counts.std()
    freq_threshold = mean_count + 2 * std_count

    for reviewer_id, group in grouped:
        review_count = len(group)

        # ── High frequency ─────────────────────────────────────────────
        high_frequency = review_count > freq_threshold

        # ── Repeated text ──────────────────────────────────────────────
        texts = group["review_text"].tolist()
        unique_texts = set(texts)
        duplicate_count = review_count - len(unique_texts)
        has_repeated_text = duplicate_count > 0

        # ── Extreme ratings (1 or 5) ──────────────────────────────────
        extreme_count = ((group["rating"] == 1) | (group["rating"] == 5)).sum()
        extreme_ratio = extreme_count / review_count if review_count > 0 else 0
        extreme_ratings = extreme_ratio > 0.7

        # ── Overall suspicion flag ─────────────────────────────────────
        is_suspicious = sum([high_frequency, has_repeated_text, extreme_ratings]) >= 2

        results.append({
            "reviewer_id": reviewer_id,
            "review_count": review_count,
            "high_frequency": high_frequency,
            "duplicate_count": duplicate_count,
            "has_repeated_text": has_repeated_text,
            "extreme_ratio": round(extreme_ratio, 2),
            "extreme_ratings": extreme_ratings,
            "is_suspicious": is_suspicious,
        })

    return pd.DataFrame(results).sort_values("review_count", ascending=False).reset_index(drop=True)
