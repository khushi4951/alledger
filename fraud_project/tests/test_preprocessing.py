import pandas as pd
import pytest

from data_preprocessing import load_and_preprocess_data, preprocess_dataframe
from validation import PCA_FEATURES


def test_preprocess_scales_and_splits(tiny_book):
    data = preprocess_dataframe(tiny_book, test_size=0.3, random_state=0)
    assert "scaled_amount" in data["X_train"].columns
    assert "scaled_time" in data["X_train"].columns
    assert "Amount" not in data["X_train"].columns
    assert set(data["y_train"].unique()) <= {0, 1}
    assert len(data["X_train"]) + len(data["X_test"]) == len(tiny_book)
    assert abs(data["X_train"]["scaled_amount"].mean()) < 0.5


def test_preprocess_rejects_single_class(tiny_book):
    only_legit = tiny_book[tiny_book.Class == 0]
    with pytest.raises(ValueError, match="Both legitimate"):
        preprocess_dataframe(only_legit)


def test_preprocess_missing_columns():
    with pytest.raises(ValueError, match="Missing"):
        preprocess_dataframe(pd.DataFrame({"Time": [1], "Class": [0]}))


def test_load_from_csv(tmp_path, tiny_book):
    path = tmp_path / "book.csv"
    tiny_book.to_csv(path, index=False)
    data = load_and_preprocess_data(str(path))
    for name in PCA_FEATURES:
        assert name in data["X_train"].columns
