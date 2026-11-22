import pandas as pd
import sklearn.preprocessing
import sklearn.model_selection


def preprocess_dataframe(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    if df is None or df.empty:
        raise ValueError("Cannot preprocess an empty frame.")
    required = {"Time", "Amount", "Class"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}.")
    if df["Class"].nunique() < 2:
        raise ValueError("Both legitimate and fraud classes are required to split the data.")

    working = df.copy()
    amount_scaler = sklearn.preprocessing.StandardScaler()
    time_scaler = sklearn.preprocessing.StandardScaler()

    working["scaled_amount"] = amount_scaler.fit_transform(working[["Amount"]])
    working["scaled_time"] = time_scaler.fit_transform(working[["Time"]])
    working = working.drop(["Amount", "Time"], axis=1)

    X = working.drop("Class", axis=1)
    y = working["Class"].astype(int)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    return {
        "df": working,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "amount_scaler": amount_scaler,
        "time_scaler": time_scaler,
    }


def load_and_preprocess_data(csv_path="creditcard.csv"):
    return preprocess_dataframe(pd.read_csv(csv_path))
