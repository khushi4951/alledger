"""KPI and chart-ready summaries for the dashboard."""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_kpis(df: pd.DataFrame) -> dict[str, float]:
    if df is None or df.empty:
        raise ValueError("Cannot compute KPIs on an empty frame.")
    if "Class" not in df.columns or "Amount" not in df.columns:
        raise ValueError("Frame must include Class and Amount columns.")

    n = int(len(df))
    frauds = df[df["Class"] == 1]
    normals = df[df["Class"] == 0]
    fraud_count = int(len(frauds))
    fraud_rate = fraud_count / n
    fraud_loss = float(frauds["Amount"].sum()) if fraud_count else 0.0
    avg_fraud = float(frauds["Amount"].mean()) if fraud_count else 0.0
    avg_normal = float(normals["Amount"].mean()) if len(normals) else 0.0
    median_fraud = float(frauds["Amount"].median()) if fraud_count else 0.0
    return {
        "transactions": n,
        "fraud_count": fraud_count,
        "legitimate_count": n - fraud_count,
        "fraud_rate": fraud_rate,
        "fraud_loss": fraud_loss,
        "avg_fraud_amount": avg_fraud,
        "avg_legitimate_amount": avg_normal,
        "median_fraud_amount": median_fraud,
        "amount_ratio": (avg_fraud / avg_normal) if avg_normal else 0.0,
    }


def fraud_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["hour"] = ((working["Time"] % 86_400) // 3600).astype(int)
    grouped = (
        working.groupby("hour")
        .agg(
            transactions=("Class", "size"),
            frauds=("Class", "sum"),
            volume=("Amount", "sum"),
        )
        .reset_index()
    )
    grouped["fraud_rate"] = grouped["frauds"] / grouped["transactions"]
    return grouped.sort_values("hour")


def amount_buckets(df: pd.DataFrame) -> pd.DataFrame:
    bins = [0, 10, 50, 100, 250, 500, 1000, 5000, np.inf]
    labels = ["≤$10", "$10–50", "$50–100", "$100–250", "$250–500", "$500–1k", "$1k–5k", "$5k+"]
    working = df.copy()
    working["bucket"] = pd.cut(working["Amount"], bins=bins, labels=labels, right=True, include_lowest=True)
    grouped = (
        working.groupby("bucket", observed=True)
        .agg(transactions=("Class", "size"), frauds=("Class", "sum"), volume=("Amount", "sum"))
        .reset_index()
    )
    grouped["fraud_rate"] = grouped["frauds"] / grouped["transactions"].replace(0, np.nan)
    grouped["fraud_rate"] = grouped["fraud_rate"].fillna(0.0)
    return grouped


def metrics_table(results: dict) -> pd.DataFrame:
    rows = {}
    for name, payload in results.items():
        rows[name] = {k: v for k, v in payload.items() if k not in {"preds", "probas"}}
    table = pd.DataFrame(rows).T
    if "f1" in table.columns:
        table = table.sort_values("f1", ascending=False)
    return table


def champion_model(results: dict, metric: str = "f1") -> str:
    if not results:
        raise ValueError("No model results available.")
    return max(results.items(), key=lambda item: item[1][metric])[0]


def class_balance(df: pd.DataFrame) -> pd.DataFrame:
    counts = df["Class"].value_counts().rename({0: "Legitimate", 1: "Fraud"})
    out = counts.rename_axis("label").reset_index(name="count")
    out["share"] = out["count"] / out["count"].sum()
    return out
