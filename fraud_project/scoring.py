"""Fraud scoring, risk bands, heuristic flags, and model input assembly."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from validation import PCA_FEATURES, validate_threshold, validate_transaction

BANDS = (
    (0.85, "Critical"),
    (0.55, "Elevated"),
    (0.25, "Watch"),
    (0.0, "Clear"),
)


def risk_band(probability: float) -> str:
    if not np.isfinite(probability):
        raise ValueError("Probability must be a finite number.")
    if probability < 0 or probability > 1:
        raise ValueError("Probability must be between 0 and 1.")
    for cutoff, label in BANDS:
        if probability >= cutoff:
            return label
    return "Clear"


def hour_from_time(time_value: float) -> float:
    return (float(time_value) % 86_400) / 3600.0


def heuristic_flags(amount: float, time_value: float, v_features: Mapping[str, float]) -> list[dict[str, str]]:
    flags: list[dict[str, str]] = []
    hour = hour_from_time(time_value)

    if amount >= 2_000:
        flags.append(
            {
                "code": "HIGH_AMOUNT",
                "severity": "high",
                "detail": f"${amount:,.2f} is well above typical card-present spend on this book.",
            }
        )
    elif amount >= 500:
        flags.append(
            {
                "code": "ELEVATED_AMOUNT",
                "severity": "medium",
                "detail": f"${amount:,.2f} is in the upper spend band.",
            }
        )

    if hour < 5 or hour >= 23:
        flags.append(
            {
                "code": "OFF_HOURS",
                "severity": "medium",
                "detail": f"Occurred at {hour:04.1f}h — off-hours activity is over-represented in fraud.",
            }
        )

    # V14 / V12 / V17 are the strongest linear correlates on the ULB dataset.
    if float(v_features.get("V14", 0.0)) <= -3.0:
        flags.append(
            {
                "code": "V14_STRESS",
                "severity": "high",
                "detail": "V14 is deeply negative — a classic fraud loading on this feature set.",
            }
        )
    if float(v_features.get("V12", 0.0)) <= -3.0:
        flags.append(
            {
                "code": "V12_STRESS",
                "severity": "high",
                "detail": "V12 is deeply negative.",
            }
        )
    if float(v_features.get("V17", 0.0)) <= -2.5:
        flags.append(
            {
                "code": "V17_STRESS",
                "severity": "medium",
                "detail": "V17 is unusually low relative to legitimate traffic.",
            }
        )
    if amount < 1 and float(v_features.get("V4", 0.0)) >= 2.0:
        flags.append(
            {
                "code": "PROBE_PATTERN",
                "severity": "medium",
                "detail": "Tiny amount with a stretched V4 — often a card-testing probe.",
            }
        )
    return flags


def _scale_values(scaler, column: str, values) -> np.ndarray:
    series = pd.Series(values, dtype=float)
    names = getattr(scaler, "feature_names_in_", None)
    if names is None:
        return scaler.transform(series.to_numpy().reshape(-1, 1)).ravel()
    label = str(names[0]) if len(names) else column
    frame = pd.DataFrame({label: series.to_numpy()})
    return scaler.transform(frame).ravel()


def build_model_input(
    amount: Any,
    time_value: Any,
    v_features: Mapping[str, Any],
    amount_scaler,
    time_scaler,
    feature_columns: list[str] | pd.Index,
) -> pd.DataFrame:
    check = validate_transaction(amount, time_value, v_features)
    check.raise_if_invalid()

    scaled_amount = float(_scale_values(amount_scaler, "Amount", [amount])[0])
    scaled_time = float(_scale_values(time_scaler, "Time", [time_value])[0])
    row = {name: float(v_features[name]) for name in PCA_FEATURES}
    row["scaled_amount"] = scaled_amount
    row["scaled_time"] = scaled_time
    frame = pd.DataFrame([row])[list(feature_columns)]
    return frame


def predict_with_threshold(model, input_df: pd.DataFrame, threshold: float = 0.5) -> dict[str, Any]:
    validate_threshold(threshold).raise_if_invalid()
    proba = float(model.predict_proba(input_df)[0, 1])
    prediction = int(proba >= float(threshold))
    return {
        "prediction": prediction,
        "probability": proba,
        "band": risk_band(proba),
        "label": "FRAUD" if prediction else "CLEAR",
        "threshold": float(threshold),
    }


def ensemble_predict(models: Mapping[str, Any], input_df: pd.DataFrame, threshold: float = 0.5) -> dict[str, Any]:
    validate_threshold(threshold).raise_if_invalid()
    if not models:
        raise ValueError("At least one model is required.")

    per_model: dict[str, float] = {}
    for name, model in models.items():
        per_model[name] = float(model.predict_proba(input_df)[0, 1])

    avg = float(np.mean(list(per_model.values())))
    votes = sum(prob >= threshold for prob in per_model.values())
    return {
        "probability": avg,
        "prediction": int(avg >= threshold),
        "band": risk_band(avg),
        "label": "FRAUD" if avg >= threshold else "CLEAR",
        "votes": int(votes),
        "panel_size": len(per_model),
        "per_model": per_model,
        "threshold": float(threshold),
    }


def feature_contributions(importances: pd.Series, row: pd.Series, top_n: int = 8) -> pd.Series:
    aligned = importances.reindex(row.index).fillna(0.0)
    contrib = aligned * row.abs()
    return contrib.sort_values(ascending=False).head(top_n)


def threshold_metrics(y_true, probabilities, threshold: float) -> dict[str, float]:
    validate_threshold(threshold).raise_if_invalid()
    y = np.asarray(y_true).astype(int)
    p = np.asarray(probabilities, dtype=float)
    pred = (p >= float(threshold)).astype(int)

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y) if len(y) else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def score_batch(model, frame: pd.DataFrame, amount_scaler, time_scaler, feature_columns, threshold: float) -> pd.DataFrame:
    from validation import validate_batch_frame

    validate_batch_frame(frame).raise_if_invalid()
    validate_threshold(threshold).raise_if_invalid()

    out = frame.copy()
    out["scaled_amount"] = _scale_values(amount_scaler, "Amount", out["Amount"])
    out["scaled_time"] = _scale_values(time_scaler, "Time", out["Time"])
    model_input = out[list(feature_columns)]
    out["fraud_probability"] = model.predict_proba(model_input)[:, 1]
    out["prediction"] = (out["fraud_probability"] >= threshold).astype(int)
    out["risk_band"] = out["fraud_probability"].map(risk_band)
    return out
