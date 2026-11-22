import numpy as np
import pandas as pd
import pytest

from scoring import (
    build_model_input,
    ensemble_predict,
    feature_contributions,
    heuristic_flags,
    hour_from_time,
    predict_with_threshold,
    risk_band,
    score_batch,
    threshold_metrics,
)
from validation import PCA_FEATURES


def test_risk_band_edges():
    assert risk_band(0.0) == "Clear"
    assert risk_band(0.24) == "Clear"
    assert risk_band(0.25) == "Watch"
    assert risk_band(0.55) == "Elevated"
    assert risk_band(0.85) == "Critical"
    assert risk_band(1.0) == "Critical"


def test_risk_band_rejects_invalid():
    with pytest.raises(ValueError):
        risk_band(-0.1)
    with pytest.raises(ValueError):
        risk_band(1.5)
    with pytest.raises(ValueError):
        risk_band(float("nan"))


def test_hour_wraps_two_days():
    assert hour_from_time(0) == 0
    assert hour_from_time(86_400 + 3600) == pytest.approx(1.0)


def test_heuristic_high_amount_and_off_hours(pca_zeros):
    flags = heuristic_flags(3_000, 2 * 3600, pca_zeros)  # 02:00
    codes = {f["code"] for f in flags}
    assert "HIGH_AMOUNT" in codes
    assert "OFF_HOURS" in codes


def test_heuristic_v14_stress():
    feats = {n: 0.0 for n in PCA_FEATURES}
    feats["V14"] = -4.0
    codes = {f["code"] for f in heuristic_flags(20, 40_000, feats)}
    assert "V14_STRESS" in codes


def test_heuristic_probe_pattern():
    feats = {n: 0.0 for n in PCA_FEATURES}
    feats["V4"] = 3.0
    codes = {f["code"] for f in heuristic_flags(0.5, 40_000, feats)}
    assert "PROBE_PATTERN" in codes


def test_heuristic_quiet_legit(pca_zeros):
    assert heuristic_flags(40, 40_000, pca_zeros) == []


def test_build_model_input_validates(pca_zeros, fitted_scaler_pair, tiny_book):
    amount_scaler, time_scaler = fitted_scaler_pair
    from data_preprocessing import preprocess_dataframe

    cols = preprocess_dataframe(tiny_book)["X_train"].columns
    with pytest.raises(ValueError):
        build_model_input(-5, 10, pca_zeros, amount_scaler, time_scaler, cols)


def test_build_model_input_column_order(pca_zeros, fitted_scaler_pair, tiny_book):
    amount_scaler, time_scaler = fitted_scaler_pair
    from data_preprocessing import preprocess_dataframe

    cols = list(preprocess_dataframe(tiny_book)["X_train"].columns)
    frame = build_model_input(50, 1000, pca_zeros, amount_scaler, time_scaler, cols)
    assert list(frame.columns) == cols
    assert len(frame) == 1


def test_predict_and_ensemble(toy_model, pca_zeros):
    model, data = toy_model
    row = build_model_input(
        80,
        1000,
        pca_zeros,
        data["amount_scaler"],
        data["time_scaler"],
        data["X_train"].columns,
    )
    single = predict_with_threshold(model, row, 0.5)
    assert set(single) >= {"prediction", "probability", "band", "label"}
    assert 0 <= single["probability"] <= 1

    panel = ensemble_predict({"A": model, "B": model}, row, 0.4)
    assert panel["panel_size"] == 2
    assert panel["votes"] in {0, 1, 2}
    assert panel["probability"] == pytest.approx(single["probability"])


def test_ensemble_requires_models(toy_model, pca_zeros):
    model, data = toy_model
    row = build_model_input(80, 1000, pca_zeros, data["amount_scaler"], data["time_scaler"], data["X_train"].columns)
    with pytest.raises(ValueError):
        ensemble_predict({}, row, 0.5)


def test_feature_contributions_orders_by_magnitude():
    importances = pd.Series({"V1": 0.5, "V2": 0.1, "V3": 0.2})
    row = pd.Series({"V1": 1.0, "V2": 10.0, "V3": 0.0})
    contrib = feature_contributions(importances, row, top_n=2)
    assert list(contrib.index) == ["V2", "V1"]


def test_threshold_metrics_perfect_split():
    y = [0, 0, 1, 1]
    p = [0.1, 0.2, 0.9, 0.8]
    metrics = threshold_metrics(y, p, 0.5)
    assert metrics["tp"] == 2
    assert metrics["fp"] == 0
    assert metrics["fn"] == 0
    assert metrics["precision"] == 1
    assert metrics["recall"] == 1
    assert metrics["f1"] == 1


def test_threshold_metrics_all_negative():
    metrics = threshold_metrics([0, 0, 0], [0.1, 0.2, 0.3], 0.9)
    assert metrics["tp"] == 0
    assert metrics["precision"] == 0
    assert metrics["recall"] == 0


def test_score_batch_rejects_bad_file(pca_zeros, toy_model):
    model, data = toy_model
    bad = pd.DataFrame([{"Amount": 1}])
    with pytest.raises(ValueError):
        score_batch(model, bad, data["amount_scaler"], data["time_scaler"], data["X_train"].columns, 0.5)


def test_score_batch_adds_columns(tiny_book, toy_model):
    model, data = toy_model
    batch = tiny_book.drop(columns=["Class"]).head(8)
    scored = score_batch(
        model,
        batch,
        data["amount_scaler"],
        data["time_scaler"],
        data["X_train"].columns,
        0.5,
    )
    assert "fraud_probability" in scored.columns
    assert "risk_band" in scored.columns
    assert scored["prediction"].isin([0, 1]).all()
    assert np.isfinite(scored["fraud_probability"]).all()
