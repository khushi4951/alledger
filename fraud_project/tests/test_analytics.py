import pandas as pd
import pytest

from analytics import amount_buckets, champion_model, class_balance, compute_kpis, fraud_by_hour, metrics_table


def test_kpis_on_tiny_book(tiny_book):
    kpis = compute_kpis(tiny_book)
    assert kpis["transactions"] == len(tiny_book)
    assert kpis["fraud_count"] == int((tiny_book.Class == 1).sum())
    assert 0 < kpis["fraud_rate"] < 1
    assert kpis["fraud_loss"] == pytest.approx(tiny_book.loc[tiny_book.Class == 1, "Amount"].sum())


def test_kpis_empty_rejected():
    with pytest.raises(ValueError):
        compute_kpis(pd.DataFrame())


def test_kpis_missing_columns():
    with pytest.raises(ValueError):
        compute_kpis(pd.DataFrame({"x": [1]}))


def test_fraud_by_hour_covers_night():
    df = pd.DataFrame(
        {
            "Time": [3600, 3600, 50_000],
            "Amount": [10, 20, 30],
            "Class": [1, 0, 0],
        }
    )
    hourly = fraud_by_hour(df)
    one_am = hourly[hourly.hour == 1].iloc[0]
    assert one_am.frauds == 1
    assert one_am.fraud_rate == pytest.approx(0.5)


def test_amount_buckets_rates(tiny_book):
    buckets = amount_buckets(tiny_book)
    assert not buckets.empty
    assert (buckets.fraud_rate >= 0).all()
    assert (buckets.fraud_rate <= 1).all()


def test_metrics_table_drops_arrays():
    results = {
        "A": {"f1": 0.2, "precision": 0.1, "preds": [0, 1], "probas": [0.1, 0.9]},
        "B": {"f1": 0.9, "precision": 0.8, "preds": [1], "probas": [0.9]},
    }
    table = metrics_table(results)
    assert "preds" not in table.columns
    assert list(table.index)[0] == "B"


def test_champion_model():
    results = {"slow": {"f1": 0.1}, "sharp": {"f1": 0.8}}
    assert champion_model(results) == "sharp"


def test_champion_empty():
    with pytest.raises(ValueError):
        champion_model({})


def test_class_balance(tiny_book):
    bal = class_balance(tiny_book)
    assert set(bal.label) == {"Legitimate", "Fraud"}
    assert bal["share"].sum() == pytest.approx(1.0)
