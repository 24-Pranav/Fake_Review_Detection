"""
Model Training Pipeline
-----------------------
Trains 5 models on the synthetic review dataset:
  1. Logistic Regression
  2. Random Forest
  3. Support Vector Machine
  4. LSTM Neural Network
  5. BERT Transformer

Evaluates each model and saves them to the models/ directory.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from modules.preprocessing import preprocess_series
from modules.feature_extraction import build_features
from modules.behavior_analysis import analyze_reviewer_behavior
from data.generate_dataset import generate_dataset

MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
os.makedirs(MODELS_DIR, exist_ok=True)


def evaluate_model(name, y_true, y_pred):
    """Print evaluation metrics for a model."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Confusion Matrix:")
    print(f"    {cm[0]}")
    print(f"    {cm[1]}")

    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def train_sklearn_models(X_train, X_test, y_train, y_test):
    """Train and evaluate Logistic Regression, Random Forest, SVM."""
    results = []

    # ── Logistic Regression ────────────────────────────────────────────
    print("\n[*] Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    results.append(evaluate_model("Logistic Regression", y_test, y_pred))
    joblib.dump(lr, os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    print("[OK] Logistic Regression saved")

    # ── Random Forest ──────────────────────────────────────────────────
    print("\n[*] Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    results.append(evaluate_model("Random Forest", y_test, y_pred))
    joblib.dump(rf, os.path.join(MODELS_DIR, "random_forest.pkl"))
    print("[OK] Random Forest saved")

    # ── SVM ────────────────────────────────────────────────────────────
    print("\n[*] Training Support Vector Machine...")
    svm = LinearSVC(max_iter=2000, random_state=42)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    results.append(evaluate_model("SVM", y_test, y_pred))
    joblib.dump(svm, os.path.join(MODELS_DIR, "svm.pkl"))
    print("[OK] SVM saved")

    return results


def train_lstm_model(X_train, X_test, y_train, y_test):
    """Train an LSTM model on TF-IDF features."""
    print("\n[*] Training LSTM Neural Network...")
    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout, Reshape
        from tensorflow.keras.callbacks import EarlyStopping

        # Convert sparse to dense and reshape for LSTM (samples, timesteps, features)
        X_tr = X_train.toarray() if hasattr(X_train, "toarray") else np.array(X_train)
        X_te = X_test.toarray() if hasattr(X_test, "toarray") else np.array(X_test)
        n_features = X_tr.shape[1]

        # Reshape: (samples, 1, features) — treat each sample as 1 timestep
        X_tr = X_tr.reshape((X_tr.shape[0], 1, n_features))
        X_te = X_te.reshape((X_te.shape[0], 1, n_features))

        model = Sequential([
            LSTM(64, input_shape=(1, n_features), return_sequences=False),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
        model.fit(
            X_tr, y_train,
            epochs=15,
            batch_size=32,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=0,
        )

        y_pred = (model.predict(X_te, verbose=0) > 0.5).astype(int).flatten()
        result = evaluate_model("LSTM", y_test, y_pred)

        model.save(os.path.join(MODELS_DIR, "lstm_model.h5"))
        print("[OK] LSTM model saved")
        return result

    except ImportError:
        print("[!] TensorFlow not available - skipping LSTM")
        return None
    except Exception as e:
        print(f"[!] LSTM training error: {e}")
        return None


def train_bert_model(texts_train, texts_test, y_train, y_test):
    """Fine-tune a BERT model for fake review classification."""
    print("\n[*] Training BERT Transformer...")
    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        from transformers import BertTokenizer, BertForSequenceClassification, AdamW

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"    Using device: {device}")

        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        # Tokenize
        train_enc = tokenizer(
            list(texts_train), truncation=True, padding=True, max_length=128, return_tensors="pt"
        )
        test_enc = tokenizer(
            list(texts_test), truncation=True, padding=True, max_length=128, return_tensors="pt"
        )

        train_dataset = TensorDataset(
            train_enc["input_ids"], train_enc["attention_mask"],
            torch.tensor(y_train.values, dtype=torch.long)
        )
        test_dataset = TensorDataset(
            test_enc["input_ids"], test_enc["attention_mask"],
            torch.tensor(y_test.values, dtype=torch.long)
        )

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16)

        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
        model.to(device)
        optimizer = AdamW(model.parameters(), lr=2e-5)

        # ── Train 2 epochs ─────────────────────────────────────────────
        model.train()
        for epoch in range(2):
            total_loss = 0
            for batch in train_loader:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            print(f"    Epoch {epoch + 1}/2  Loss: {total_loss / len(train_loader):.4f}")

        # ── Evaluate ───────────────────────────────────────────────────
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        result = evaluate_model("BERT", np.array(all_labels), np.array(all_preds))

        # Save model
        bert_dir = os.path.join(MODELS_DIR, "bert_model")
        model.save_pretrained(bert_dir)
        tokenizer.save_pretrained(bert_dir)
        print("[OK] BERT model saved")
        return result

    except ImportError:
        print("[!] Transformers / PyTorch not available - skipping BERT")
        return None
    except Exception as e:
        print(f"[!] BERT training error: {e}")
        return None


def main():
    print("=" * 60)
    print("  FAKE REVIEW DETECTION - MODEL TRAINING PIPELINE")
    print("=" * 60)

    # ── 1. Generate / Load Dataset ────────────────────────────────────
    csv_path = os.path.join(DATA_DIR, "reviews.csv")
    if not os.path.exists(csv_path):
        print("\n[*] Generating synthetic dataset...")
        df = generate_dataset(n_samples=2000, output_path=csv_path)
    else:
        print(f"\n[*] Loading existing dataset from {csv_path}")
        df = pd.read_csv(csv_path)

    print(f"    Total reviews: {len(df)}")
    print(f"    Genuine: {(df['label'] == 0).sum()}  |  Fake: {(df['label'] == 1).sum()}")

    # ── 2. Reviewer Behavior Analysis ─────────────────────────────────
    print("\n[*] Running reviewer behavior analysis...")
    behavior_df = analyze_reviewer_behavior(df)
    suspicious = behavior_df[behavior_df["is_suspicious"]]
    print(f"    Suspicious reviewers: {len(suspicious)} / {len(behavior_df)}")
    behavior_df.to_csv(os.path.join(DATA_DIR, "behavior_analysis.csv"), index=False)
    print("[OK] Behavior analysis saved")

    # ── 3. Preprocess ─────────────────────────────────────────────────
    print("\n[*] Preprocessing text...")
    df["clean_text"] = preprocess_series(df["review_text"])

    # ── 4. Feature Extraction ─────────────────────────────────────────
    print("\n[*] Extracting features...")
    X, vectorizer = build_features(df["clean_text"], fit=True)
    y = df["label"]

    # ── 5. Train-Test Split (80/20) ───────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n    Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    # Keep raw texts for BERT
    texts_train_idx, texts_test_idx = train_test_split(
        df.index, test_size=0.2, random_state=42, stratify=y
    )

    # ── 6. Train Models ───────────────────────────────────────────────
    all_results = []
    trained_models = {}  # name -> model object

    # Sklearn models
    sklearn_results = train_sklearn_models(X_train, X_test, y_train, y_test)
    all_results.extend(sklearn_results)
    # Load back the saved sklearn models for best-model selection
    trained_models["Logistic Regression"] = joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl"))
    trained_models["Random Forest"] = joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl"))
    trained_models["SVM"] = joblib.load(os.path.join(MODELS_DIR, "svm.pkl"))

    # LSTM
    lstm_result = train_lstm_model(X_train, X_test, y_train, y_test)
    if lstm_result:
        all_results.append(lstm_result)

    # BERT
    bert_result = train_bert_model(
        df.loc[texts_train_idx, "review_text"],
        df.loc[texts_test_idx, "review_text"],
        df.loc[texts_train_idx, "label"],
        df.loc[texts_test_idx, "label"],
    )
    if bert_result:
        all_results.append(bert_result)

    # ── 7. Summary ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 60)
    results_df = pd.DataFrame(all_results)
    print(results_df.to_string(index=False))

    results_df.to_csv(os.path.join(DATA_DIR, "model_results.csv"), index=False)
    print(f"\n[OK] Results saved to {os.path.join(DATA_DIR, 'model_results.csv')}")

    # ── 8. Auto-select best model and save as model.pkl ───────────────
    # Only consider sklearn models for model.pkl (LSTM/BERT have separate formats)
    sklearn_results_df = results_df[results_df["model"].isin(trained_models.keys())]
    if not sklearn_results_df.empty:
        best_row = sklearn_results_df.loc[sklearn_results_df["f1"].idxmax()]
        best_name = best_row["model"]
        best_model = trained_models[best_name]
        joblib.dump(best_model, os.path.join(MODELS_DIR, "model.pkl"))
        print(f"\n[OK] Best model selected: {best_name} (F1={best_row['f1']:.4f})")
        print(f"[OK] Saved as model.pkl")
    else:
        print("\n[!] No sklearn models to select from")

    print("[OK] All models saved to models/ directory")
    print("\nTraining complete! Run the Flask app with: python app/app.py")


if __name__ == "__main__":
    main()

