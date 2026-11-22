"""Load trained artifacts, or bootstrap a lightweight model if the pickle is missing."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import sklearn.decomposition
import sklearn.linear_model
import sklearn.metrics
import sklearn.tree

from data_preprocessing import preprocess_dataframe


def bootstrap_artifacts(csv_path: str = "creditcard.csv", max_legitimate: int = 8_000) -> dict:
    raw = pd.read_csv(csv_path)
    fraud = raw[raw["Class"] == 1]
    legit = raw[raw["Class"] == 0].sample(n=min(max_legitimate, (raw["Class"] == 0).sum()), random_state=42)
    sample = pd.concat([fraud, legit]).sample(frac=1, random_state=42).reset_index(drop=True)
    data = preprocess_dataframe(sample)

    models = {
        "Logistic Regression": sklearn.linear_model.LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Decision Tree": sklearn.tree.DecisionTreeClassifier(
            class_weight="balanced", random_state=42, max_depth=8
        ),
    }

    results = {}
    cv_scores = {}
    for name, model in models.items():
        model.fit(data["X_train"], data["y_train"])
        preds = model.predict(data["X_test"])
        probas = model.predict_proba(data["X_test"])[:, 1]
        results[name] = {
            "accuracy": sklearn.metrics.accuracy_score(data["y_test"], preds),
            "precision": sklearn.metrics.precision_score(data["y_test"], preds, zero_division=0),
            "recall": sklearn.metrics.recall_score(data["y_test"], preds, zero_division=0),
            "f1": sklearn.metrics.f1_score(data["y_test"], preds, zero_division=0),
            "roc_auc": sklearn.metrics.roc_auc_score(data["y_test"], probas),
            "preds": preds,
            "probas": probas,
        }
        cv_scores[name] = "bootstrap (not cross-validated)"

    pca = sklearn.decomposition.PCA(n_components=2, random_state=42)
    x_pca_train = pca.fit_transform(data["X_train"])

    return {
        "models": models,
        "results": results,
        "cv_scores": cv_scores,
        "pca": pca,
        "X_pca_train": x_pca_train,
        "X_test": data["X_test"],
        "y_test": data["y_test"],
        "X_train": data["X_train"],
        "y_train": data["y_train"],
        "amount_scaler": data["amount_scaler"],
        "time_scaler": data["time_scaler"],
        "bootstrap": True,
    }


def load_artifacts(pkl_path: str = "saved_models.pkl", csv_path: str = "creditcard.csv") -> tuple[dict, bool]:
    path = Path(pkl_path)
    if path.exists():
        artifacts = joblib.load(path)
        artifacts["bootstrap"] = False
        return artifacts, False
    return bootstrap_artifacts(csv_path), True
