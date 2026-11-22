import pandas as pd
import pytest

from validation import (
    PCA_FEATURES,
    validate_amount,
    validate_batch_frame,
    validate_pca_features,
    validate_threshold,
    validate_time,
    validate_transaction,
)


def test_amount_rejects_negative():
    result = validate_amount(-1)
    assert not result.ok
    assert "negative" in result.errors[0].lower()


def test_amount_rejects_none():
    assert not validate_amount(None).ok


def test_amount_rejects_nan():
    assert not validate_amount(float("nan")).ok


def test_amount_rejects_bool():
    assert not validate_amount(True).ok


def test_amount_warns_on_very_large():
    result = validate_amount(12_000)
    assert result.ok
    assert result.warnings


def test_amount_rejects_above_cap():
    assert not validate_amount(1_000_000).ok


def test_time_rejects_negative():
    assert not validate_time(-5).ok


def test_time_rejects_beyond_window():
    assert not validate_time(500_000).ok


def test_time_accepts_dataset_span():
    assert validate_time(172_792).ok


def test_pca_unknown_feature():
    result = validate_pca_features({"V99": 0.0, **{n: 0.0 for n in PCA_FEATURES}})
    assert result.ok
    assert result.warnings


def test_pca_missing_features():
    result = validate_pca_features({"V1": 0.0})
    assert not result.ok
    assert "Missing" in result.errors[0]


def test_pca_out_of_range():
    feats = {n: 0.0 for n in PCA_FEATURES}
    feats["V14"] = -200
    assert not validate_pca_features(feats).ok


def test_pca_extreme_warning():
    feats = {n: 0.0 for n in PCA_FEATURES}
    feats["V4"] = 25
    result = validate_pca_features(feats)
    assert result.ok
    assert result.warnings


def test_transaction_aggregates_errors(pca_zeros):
    result = validate_transaction(-10, -1, pca_zeros)
    assert not result.ok
    assert len(result.errors) >= 2


def test_valid_transaction_passes(pca_zeros):
    result = validate_transaction(42.5, 1000, pca_zeros)
    assert result.ok
    assert not result.errors


def test_threshold_bounds():
    assert not validate_threshold(-0.1).ok
    assert not validate_threshold(1.2).ok
    assert validate_threshold(0.35).ok


def test_threshold_warns_at_extremes():
    assert validate_threshold(0.02).warnings
    assert validate_threshold(0.95).warnings


def test_batch_empty():
    assert not validate_batch_frame(pd.DataFrame()).ok


def test_batch_missing_columns():
    assert not validate_batch_frame(pd.DataFrame({"Amount": [1]})).ok


def test_batch_non_numeric(pca_zeros):
    row = {"Time": 1, "Amount": "abc", **pca_zeros}
    assert not validate_batch_frame(pd.DataFrame([row])).ok


def test_batch_happy_path(pca_zeros):
    row = {"Time": 10, "Amount": 20, **pca_zeros}
    assert validate_batch_frame(pd.DataFrame([row])).ok


def test_raise_if_invalid(pca_zeros):
    with pytest.raises(ValueError):
        validate_transaction(-1, 0, pca_zeros).raise_if_invalid()
