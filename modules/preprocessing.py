"""
NLP Text Preprocessing Module
- Lowercase conversion
- Punctuation removal
- Tokenization
- Stopword removal
- Lemmatization
"""

import re
import string
import nltk

# Try downloading NLTK data; if network unavailable, use fallback
_NLTK_AVAILABLE = True
try:
    for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
        nltk.download(resource, quiet=True)
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    _lemmatizer = WordNetLemmatizer()
    _stop_words = set(stopwords.words("english"))
except Exception:
    _NLTK_AVAILABLE = False
    # Fallback stopwords (most common English stopwords)
    _stop_words = {
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you",
        "your", "yours", "yourself", "yourselves", "he", "him", "his",
        "himself", "she", "her", "hers", "herself", "it", "its", "itself",
        "they", "them", "their", "theirs", "themselves", "what", "which",
        "who", "whom", "this", "that", "these", "those", "am", "is", "are",
        "was", "were", "be", "been", "being", "have", "has", "had", "having",
        "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if",
        "or", "because", "as", "until", "while", "of", "at", "by", "for",
        "with", "about", "against", "between", "through", "during", "before",
        "after", "above", "below", "to", "from", "up", "down", "in", "out",
        "on", "off", "over", "under", "again", "further", "then", "once",
        "here", "there", "when", "where", "why", "how", "all", "both",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "s", "t",
        "can", "will", "just", "don", "should", "now", "d", "ll", "m", "o",
        "re", "ve", "y", "ain", "aren", "couldn", "didn", "doesn", "hadn",
        "hasn", "haven", "isn", "ma", "mightn", "mustn", "needn", "shan",
        "shouldn", "wasn", "weren", "won", "wouldn",
    }
    _lemmatizer = None


def preprocess_text(text: str) -> str:
    """Full NLP preprocessing pipeline on a single text string."""
    # 1. Lowercase
    text = text.lower()
    # 2. Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # 3. Remove extra whitespace & digits
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    # 4. Tokenize
    if _NLTK_AVAILABLE:
        from nltk.tokenize import word_tokenize
        tokens = word_tokenize(text)
    else:
        tokens = text.split()
    # 5. Remove stopwords & lemmatize
    if _lemmatizer is not None:
        tokens = [
            _lemmatizer.lemmatize(tok)
            for tok in tokens
            if tok not in _stop_words and len(tok) > 1
        ]
    else:
        tokens = [tok for tok in tokens if tok not in _stop_words and len(tok) > 1]
    return " ".join(tokens)


def preprocess_series(series):
    """Apply preprocessing to a pandas Series."""
    return series.astype(str).apply(preprocess_text)
