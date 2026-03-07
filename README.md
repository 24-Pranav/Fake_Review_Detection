# 🛡️ FakeGuard AI — Fake Review Detection System

An advanced AI-based Fake Review Detection System using Python, Machine Learning, and Deep Learning.

## Features

- **NLP Preprocessing** — lowercase, stopword removal, tokenization, lemmatization, punctuation removal
- **Feature Extraction** — TF-IDF, n-gram features, review length, sentiment score
- **5 ML/DL Models** — Logistic Regression, Random Forest, SVM, LSTM, BERT
- **Sentiment Analysis** — VADER-based sentiment scoring
- **Reviewer Behavior Analysis** — frequency, duplication, extreme rating detection
- **Explainable AI** — SHAP feature attribution
- **Flask Web App** — prediction form, confidence scores, explanation dashboard
- **Analytics Dashboard** — distribution charts, sentiment breakdown, top words

## Project Structure

```
├── data/              → Dataset generation
├── models/            → Saved trained models
├── modules/           → Core NLP & ML modules
├── training/          → Model training pipeline
├── app/               → Flask web application
└── requirements.txt
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models
```bash
python training/train_models.py
```

### 3. Run the Web App
```bash
python app/app.py
```
Then visit **http://localhost:5000**

## Models Trained

| Model | Type |
|---|---|
| Logistic Regression | Scikit-learn |
| Random Forest | Scikit-learn |
| SVM (LinearSVC) | Scikit-learn |
| LSTM | TensorFlow/Keras |
| BERT | HuggingFace Transformers |

## Evaluation Metrics

Each model is evaluated on: **Accuracy**, **Precision**, **Recall**, **F1 Score**, and **Confusion Matrix**.

## Tech Stack

Python, Flask, scikit-learn, TensorFlow, PyTorch, Transformers, NLTK, VADER, SHAP, Chart.js
