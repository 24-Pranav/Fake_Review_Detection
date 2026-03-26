"""
Reviewer Behaviour Analysis Module
───────────────────────────────────
Detects suspicious patterns:
  • High review frequency
  • Repeated / duplicate review text
  • Extreme rating patterns
  • Readability consistency (Flesch-Kincaid)
"""

import pandas as pd
import numpy as np


def calculate_readability(text: str) -> dict:
    """
    Calculate readability metrics for a single review using Flesch-Kincaid.

    Flags reviews that are suspiciously "perfect" or overly simple,
    which is typical of bot-generated or template-based fake reviews.

    Parameters
    ----------
    text : str — the review text to analyze

    Returns
    -------
    dict with keys:
        fk_grade : float — Flesch-Kincaid grade level
        reading_ease : float — Flesch Reading Ease score (0–100)
        is_unnaturally_consistent : bool — True if readability falls in
            the suspicious "AI-generated" band (grade 6–8, ease 60–70)
        is_bot_generated : bool — True if text is suspiciously simple
            or suspiciously "perfect" in structure
        bot_reason : str | None — Explanation if flagged
    """
    try:
        import textstat

        fk_grade = textstat.flesch_kincaid_grade(text)
        reading_ease = textstat.flesch_reading_ease(text)

        # AI-generated reviews tend to have very consistent readability
        # in the "accessible but not too simple" range
        is_unnaturally_consistent = (6.0 <= fk_grade <= 8.0) and (60.0 <= reading_ease <= 70.0)

        # Bot-detection: suspiciously simple or suspiciously polished
        is_bot = False
        bot_reason = None

        if reading_ease > 80 and fk_grade < 4:
            is_bot = True
            bot_reason = (
                f"Suspiciously simple text (Reading Ease: {reading_ease:.0f}, "
                f"Grade: {fk_grade:.1f}). Typical of bot-generated reviews."
            )
        elif reading_ease < 30 and fk_grade > 12:
            is_bot = True
            bot_reason = (
                f"Suspiciously complex text (Reading Ease: {reading_ease:.0f}, "
                f"Grade: {fk_grade:.1f}). May be auto-generated academic-style spam."
            )
        elif is_unnaturally_consistent:
            is_bot = True
            bot_reason = (
                f"Readability is unnaturally consistent (Grade: {fk_grade:.1f}, "
                f"Ease: {reading_ease:.0f}) — common in AI-generated template reviews."
            )

        return {
            "fk_grade": round(fk_grade, 2),
            "reading_ease": round(reading_ease, 2),
            "is_unnaturally_consistent": is_unnaturally_consistent,
            "is_bot_generated": is_bot,
            "bot_reason": bot_reason,
        }
    except ImportError:
        print("[!] textstat not installed - readability analysis unavailable")
        return {"fk_grade": None, "reading_ease": None, "is_unnaturally_consistent": False,
                "is_bot_generated": False, "bot_reason": None}
    except Exception as e:
        print(f"[!] Readability calculation error: {e}")
        return {"fk_grade": None, "reading_ease": None, "is_unnaturally_consistent": False,
                "is_bot_generated": False, "bot_reason": None}


def analyze_readability_consistency(texts: list) -> dict:
    """
    Analyze whether a batch of reviews (e.g. from a single reviewer)
    has unnaturally consistent readability scores.

    Low standard deviation in FK grade across many reviews suggests
    AI-generated or template-based content.

    Parameters
    ----------
    texts : list of str — review texts from the same reviewer

    Returns
    -------
    dict with keys:
        mean_fk_grade : float
        std_fk_grade : float
        mean_reading_ease : float
        std_reading_ease : float
        is_suspiciously_consistent : bool — True if std_fk_grade < 1.0
            across 3+ reviews
    """
    if len(texts) < 2:
        return {
            "mean_fk_grade": None, "std_fk_grade": None,
            "mean_reading_ease": None, "std_reading_ease": None,
            "is_suspiciously_consistent": False,
        }

    metrics = [calculate_readability(t) for t in texts]
    grades = [m["fk_grade"] for m in metrics if m["fk_grade"] is not None]
    eases = [m["reading_ease"] for m in metrics if m["reading_ease"] is not None]

    if len(grades) < 2:
        return {
            "mean_fk_grade": None, "std_fk_grade": None,
            "mean_reading_ease": None, "std_reading_ease": None,
            "is_suspiciously_consistent": False,
        }

    mean_grade = float(np.mean(grades))
    std_grade = float(np.std(grades))
    mean_ease = float(np.mean(eases))
    std_ease = float(np.std(eases))

    # Suspiciously consistent: low variation across 3+ reviews
    is_suspicious = len(grades) >= 3 and std_grade < 1.0

    return {
        "mean_fk_grade": round(mean_grade, 2),
        "std_fk_grade": round(std_grade, 2),
        "mean_reading_ease": round(mean_ease, 2),
        "std_reading_ease": round(std_ease, 2),
        "is_suspiciously_consistent": is_suspicious,
    }


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
