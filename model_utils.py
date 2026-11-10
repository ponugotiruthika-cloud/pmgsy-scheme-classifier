"""
ML Model Training and Evaluation for PMGSY Scheme Classification
Trains a RandomForest (local) and optionally deploys to IBM Watson ML.
"""

import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from data_utils import load_data, preprocess, SCHEME_LABELS, TARGET, ALL_FEATURES

MODEL_PATH = "pmgsy_model.pkl"
ARTIFACTS_PATH = "pmgsy_artifacts.pkl"


# ─────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────

def train_model(csv_path: str = None):
    """
    Load data → preprocess → train RandomForest → evaluate → save artifacts.
    Returns a results dict with metrics and the trained pipeline objects.
    """
    df = load_data(csv_path)
    X_train, X_test, y_train, y_test, label_encoders, le_target, scaler, feature_names = preprocess(df)

    # ── Model ensemble: pick best via cross-val on training set ──
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # ── Evaluation ──
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test, y_pred,
        target_names=le_target.classes_,
        output_dict=True,
    )

    # ── Feature importances ──
    importances = clf.feature_importances_
    feat_imp = sorted(
        zip(feature_names, importances), key=lambda x: x[1], reverse=True
    )

    # ── Persist ──
    artifacts = {
        "clf": clf,
        "scaler": scaler,
        "label_encoders": label_encoders,
        "le_target": le_target,
        "feature_names": feature_names,
        "scheme_labels": list(le_target.classes_),
    }
    joblib.dump(artifacts, ARTIFACTS_PATH)

    results = {
        "accuracy": acc,
        "f1_weighted": f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "feature_importances": feat_imp,
        "scheme_labels": list(le_target.classes_),
        "artifacts_path": ARTIFACTS_PATH,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
    }
    return results, artifacts


# ─────────────────────────────────────────────
# Inference (local)
# ─────────────────────────────────────────────

def load_artifacts():
    if not os.path.exists(ARTIFACTS_PATH):
        raise FileNotFoundError("Model not trained yet. Please train the model first.")
    return joblib.load(ARTIFACTS_PATH)


def predict_local(input_dict: dict, artifacts: dict) -> dict:
    """
    Predict scheme for a single record.
    input_dict keys must match ALL_FEATURES.
    Returns predicted scheme label and class probabilities.
    """
    from data_utils import NUMERIC_FEATURES, CATEGORICAL_FEATURES, ALL_FEATURES
    import pandas as pd

    row = {}
    for col in CATEGORICAL_FEATURES:
        le = artifacts["label_encoders"][col]
        val = str(input_dict.get(col, le.classes_[0]))
        if val not in le.classes_:
            val = le.classes_[0]
        row[col] = le.transform([val])[0]

    for col in NUMERIC_FEATURES:
        row[col] = float(input_dict.get(col, 0.0))

    X = np.array([[row[f] for f in ALL_FEATURES]])
    X[:, : len(NUMERIC_FEATURES)] = artifacts["scaler"].transform(
        X[:, : len(NUMERIC_FEATURES)]
    )

    pred_idx = artifacts["clf"].predict(X)[0]
    proba = artifacts["clf"].predict_proba(X)[0]

    predicted_label = artifacts["le_target"].inverse_transform([pred_idx])[0]
    proba_dict = {
        label: round(float(p), 4)
        for label, p in zip(artifacts["scheme_labels"], proba)
    }

    return {"predicted_scheme": predicted_label, "probabilities": proba_dict}
