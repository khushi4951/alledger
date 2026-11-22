import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from validation import PCA_FEATURES


def _tiny_book(n_legit: int = 40, n_fraud: int = 10, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for label, count in ((0, n_legit), (1, n_fraud)):
        for _ in range(count):
            row = {
                "Time": float(rng.uniform(0, 100_000)),
                "Amount": float(rng.uniform(1, 800) if label == 0 else rng.uniform(5, 2500)),
                "Class": label,
            }
            for i, name in enumerate(PCA_FEATURES):
                center = -2.5 if label == 1 and name in {"V14", "V12", "V17"} else 0.0
                row[name] = float(rng.normal(center, 1.0))
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture
def tiny_book() -> pd.DataFrame:
    return _tiny_book()


@pytest.fixture
def pca_zeros() -> dict[str, float]:
    return {name: 0.0 for name in PCA_FEATURES}


@pytest.fixture
def fitted_scaler_pair(tiny_book):
    amount = StandardScaler().fit(tiny_book[["Amount"]])
    time = StandardScaler().fit(tiny_book[["Time"]])
    return amount, time


@pytest.fixture
def toy_model(tiny_book, pca_zeros):
    from data_preprocessing import preprocess_dataframe

    data = preprocess_dataframe(tiny_book, test_size=0.3)
    model = LogisticRegression(max_iter=500, class_weight="balanced", random_state=42)
    model.fit(data["X_train"], data["y_train"])
    return model, data
