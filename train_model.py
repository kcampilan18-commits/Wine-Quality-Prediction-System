"""
Wine Quality Prediction System - model training script
Downloads the UCI red-wine dataset, prepares a binary target, trains a
Random Forest pipeline, evaluates it, and saves the trained model.
"""

from pathlib import Path
import json
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

DATA_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "wine-quality/winequality-red.csv"
)
DATA_FILE = Path("wine_quality.csv")
MODEL_FILE = Path("wine_model.pkl")
METRICS_FILE = Path("evaluation_results.json")
RANDOM_STATE = 42


def main():
    print("Downloading the UCI red-wine dataset...")
    df = pd.read_csv(DATA_URL, sep=";")
    df.to_csv(DATA_FILE, index=False)

    print(f"Dataset shape before cleaning: {df.shape}")
    print(df.head())
    print("\nMissing values:\n", df.isnull().sum())
    print("\nDuplicate rows:", int(df.duplicated().sum()))

    # Remove duplicate records and create the binary classification target.
    df = df.drop_duplicates().copy()
    y = (df["quality"] >= 6).astype(int)
    X = df.drop(columns=["quality"])  # Prevent target leakage.

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        )),
    ])

    print("\nTraining Random Forest model...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision_good_quality": float(
            precision_score(y_test, predictions, pos_label=1, zero_division=0)
        ),
        "recall_good_quality": float(
            recall_score(y_test, predictions, pos_label=1, zero_division=0)
        ),
        "f1_good_quality": float(
            f1_score(y_test, predictions, pos_label=1, zero_division=0)
        ),
        "precision_needs_improvement": float(
            precision_score(y_test, predictions, pos_label=0, zero_division=0)
        ),
        "recall_needs_improvement": float(
            recall_score(y_test, predictions, pos_label=0, zero_division=0)
        ),
        "f1_needs_improvement": float(
            f1_score(y_test, predictions, pos_label=0, zero_division=0)
        ),
        "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
        "test_records": int(len(y_test)),
        "training_records": int(len(y_train)),
        "class_labels": {"0": "NEEDS IMPROVEMENT", "1": "GOOD QUALITY"},
    }

    print("\nMODEL EVALUATION")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(
        "GOOD QUALITY - "
        f"Precision: {metrics['precision_good_quality']:.4f}, "
        f"Recall: {metrics['recall_good_quality']:.4f}, "
        f"F1-score: {metrics['f1_good_quality']:.4f}"
    )
    print(
        "NEEDS IMPROVEMENT - "
        f"Precision: {metrics['precision_needs_improvement']:.4f}, "
        f"Recall: {metrics['recall_needs_improvement']:.4f}, "
        f"F1-score: {metrics['f1_needs_improvement']:.4f}"
    )
    print("\nConfusion matrix:\n", confusion_matrix(y_test, predictions))
    print("\nClassification report:\n", classification_report(
        y_test,
        predictions,
        target_names=["Needs improvement", "Good quality"],
        zero_division=0,
    ))

    joblib.dump(model, MODEL_FILE)
    METRICS_FILE.write_text(json.dumps(metrics, indent=4), encoding="utf-8")
    print(f"\nSaved: {MODEL_FILE}")
    print(f"Saved: {METRICS_FILE}")
    print(f"Saved: {DATA_FILE}")


if __name__ == "__main__":
    main()
