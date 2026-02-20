# src/modeling.py
import os
import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve

# Local imports (module-friendly)
from .preprocessing import load_data, split_data, get_preprocessor

RANDOM_STATE = 42
TARGET_PRECISION = 0.95
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "artifacts")


def build_pipeline(preprocessor):
    """Build the Logistic Regression pipeline"""
    pipeline = Pipeline([
        ("prep", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ])
    return pipeline


def tune_threshold(model, X_train, y_train, target_precision=TARGET_PRECISION):
    """Tune decision threshold on training data to achieve target precision"""
    y_scores = model.predict_proba(X_train)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)

    precisions = precisions[:-1]  # align threshold array
    valid_idxs = np.where(precisions >= target_precision)[0]

    if len(valid_idxs) == 0:
        raise ValueError("Target precision not achievable with this model.")

    best_idx = valid_idxs[0]
    best_threshold = thresholds[best_idx]
    return best_threshold


def main():
    # Load dataset
    data_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw", "adult.csv")
    df = load_data(data_path)

    # Map target
    df["class"] = df["class"].map({
        "<=50K": 1,
        ">50K": 0
    })

    # Split data
    X_train, X_test, y_train, y_test = split_data(df)

    # Preprocessing
    preprocessor = get_preprocessor(X_train)

    # Build model
    model = build_pipeline(preprocessor)

    # Fit on training data
    model.fit(X_train, y_train)

    # Tune threshold
    threshold = tune_threshold(model, X_train, y_train, TARGET_PRECISION)

    # Save artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "logistic_model.pkl"))
    joblib.dump(threshold, os.path.join(MODEL_DIR, "decision_threshold.pkl"))

    print("Model and threshold saved successfully.")
    print(f"Threshold for precision={TARGET_PRECISION}: {threshold}")


if __name__ == "__main__":
    main()