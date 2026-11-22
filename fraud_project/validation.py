"""Input validation for transaction scoring and batch uploads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np
import pandas as pd

PCA_FEATURES = tuple(f"V{i}" for i in range(1, 29))
REQUIRED_BATCH_COLUMNS = ("Time", "Amount", *PCA_FEATURES)

MIN_AMOUNT = 0.0
MAX_AMOUNT = 250_000.0
MIN_TIME = 0.0
MAX_TIME = 200_000.0  # ULB dataset spans ~2 days (~172,792s)
MIN_PCA = -80.0
MAX_PCA = 80.0


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise ValueError("; ".join(self.errors))


def _as_float(value: Any, label: str) -> tuple[float | None, str | None]:
    if value is None or value == "":
        return None, f"{label} is required."
    if isinstance(value, (bool, np.bool_)):
        return None, f"{label} must be a number."
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, f"{label} must be a number."
    if not np.isfinite(number):
        return None, f"{label} must be a finite number."
    return number, None


def validate_amount(value: Any) -> ValidationResult:
    result = ValidationResult(ok=True)
    amount, err = _as_float(value, "Amount")
    if err:
        result.ok = False
        result.errors.append(err)
        return result
    if amount < MIN_AMOUNT:
        result.ok = False
        result.errors.append("Amount cannot be negative.")
    elif amount > MAX_AMOUNT:
        result.ok = False
        result.errors.append(f"Amount exceeds the allowed maximum of ${MAX_AMOUNT:,.0f}.")
    elif amount >= 10_000:
        result.warnings.append("Very large amount — review against the cardholder's typical spend.")
    return result


def validate_time(value: Any) -> ValidationResult:
    result = ValidationResult(ok=True)
    time_value, err = _as_float(value, "Time")
    if err:
        result.ok = False
        result.errors.append(err)
        return result
    if time_value < MIN_TIME:
        result.ok = False
        result.errors.append("Time cannot be negative.")
    elif time_value > MAX_TIME:
        result.ok = False
        result.errors.append(
            f"Time exceeds the allowed window of {MAX_TIME:,.0f} seconds from the first transaction."
        )
    return result


def validate_pca_value(name: str, value: Any) -> ValidationResult:
    result = ValidationResult(ok=True)
    if name not in PCA_FEATURES:
        result.ok = False
        result.errors.append(f"Unknown PCA feature '{name}'.")
        return result
    number, err = _as_float(value, name)
    if err:
        result.ok = False
        result.errors.append(err)
        return result
    if number < MIN_PCA or number > MAX_PCA:
        result.ok = False
        result.errors.append(f"{name} is outside the accepted range [{MIN_PCA}, {MAX_PCA}].")
    elif abs(number) >= 20:
        result.warnings.append(f"{name} is an extreme PCA loading ({number:.2f}).")
    return result


def validate_pca_features(v_features: Mapping[str, Any] | None) -> ValidationResult:
    result = ValidationResult(ok=True)
    if v_features is None:
        result.ok = False
        result.errors.append("PCA features are required.")
        return result

    missing = [name for name in PCA_FEATURES if name not in v_features]
    if missing:
        result.ok = False
        result.errors.append(f"Missing PCA features: {', '.join(missing)}.")

    extra = [name for name in v_features if name not in PCA_FEATURES]
    if extra:
        result.warnings.append(f"Ignoring unexpected fields: {', '.join(sorted(extra))}.")

    for name in PCA_FEATURES:
        if name not in v_features:
            continue
        part = validate_pca_value(name, v_features[name])
        result.errors.extend(part.errors)
        result.warnings.extend(part.warnings)
        if not part.ok:
            result.ok = False
    return result


def validate_transaction(amount: Any, time_value: Any, v_features: Mapping[str, Any] | None) -> ValidationResult:
    merged = ValidationResult(ok=True)
    for part in (validate_amount(amount), validate_time(time_value), validate_pca_features(v_features)):
        merged.errors.extend(part.errors)
        merged.warnings.extend(part.warnings)
        if not part.ok:
            merged.ok = False
    return merged


def validate_threshold(value: Any) -> ValidationResult:
    result = ValidationResult(ok=True)
    threshold, err = _as_float(value, "Decision threshold")
    if err:
        result.ok = False
        result.errors.append(err)
        return result
    if threshold < 0.0 or threshold > 1.0:
        result.ok = False
        result.errors.append("Decision threshold must be between 0 and 1.")
    elif threshold < 0.05:
        result.warnings.append("A very low threshold will flag almost every transaction.")
    elif threshold > 0.9:
        result.warnings.append("A very high threshold will miss most fraud.")
    return result


def validate_batch_frame(df: pd.DataFrame) -> ValidationResult:
    result = ValidationResult(ok=True)
    if df is None or df.empty:
        result.ok = False
        result.errors.append("Uploaded file has no rows.")
        return result

    missing = [col for col in REQUIRED_BATCH_COLUMNS if col not in df.columns]
    if missing:
        result.ok = False
        result.errors.append(f"Missing required columns: {', '.join(missing)}.")
        return result

    if len(df) > 5_000:
        result.ok = False
        result.errors.append("Batch uploads are limited to 5,000 rows.")
        return result

    numeric_cols = list(REQUIRED_BATCH_COLUMNS)
    coerced = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    if coerced.isna().any().any():
        bad = coerced.columns[coerced.isna().any()].tolist()
        result.ok = False
        result.errors.append(f"Non-numeric values in: {', '.join(bad)}.")
        return result

    if (coerced["Amount"] < MIN_AMOUNT).any() or (coerced["Amount"] > MAX_AMOUNT).any():
        result.ok = False
        result.errors.append("One or more Amount values are outside the allowed range.")
    if (coerced["Time"] < MIN_TIME).any() or (coerced["Time"] > MAX_TIME).any():
        result.ok = False
        result.errors.append("One or more Time values are outside the allowed range.")

    pca = coerced[list(PCA_FEATURES)]
    if ((pca < MIN_PCA) | (pca > MAX_PCA)).any().any():
        result.ok = False
        result.errors.append("One or more PCA features are outside the allowed range.")

    return result
