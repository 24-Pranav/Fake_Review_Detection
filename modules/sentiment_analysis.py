"""
Sentiment Analysis Module (VADER)
─────────────────────────────────
Uses VADER (Valence Aware Dictionary and sEntiment Reasoner)
to compute sentiment scores for review text.
"""

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_analyzer = SentimentIntensityAnalyzer()


def get_sentiment(text: str) -> dict:
    """Return VADER sentiment scores: neg, neu, pos, compound."""
    return _analyzer.polarity_scores(text)


def get_sentiment_compound(text: str) -> float:
    """Return only the compound sentiment score (-1 to +1)."""
    return _analyzer.polarity_scores(text)["compound"]


def get_sentiment_label(text: str) -> str:
    """Classify text sentiment as Positive / Negative / Neutral."""
    compound = get_sentiment_compound(text)
    if compound >= 0.05:
        return "Positive"
    elif compound <= -0.05:
        return "Negative"
    else:
        return "Neutral"
