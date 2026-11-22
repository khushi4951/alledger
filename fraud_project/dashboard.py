"""Streamlit page renderers for the fraud operations dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import sklearn.metrics
import streamlit as st

from analytics import amount_buckets, champion_model, class_balance, compute_kpis, fraud_by_hour, metrics_table
from scoring import (
    build_model_input,
    ensemble_predict,
    feature_contributions,
    heuristic_flags,
    predict_with_threshold,
    risk_band,
    score_batch,
    threshold_metrics,
)
from theme import COLORS, PLOTLY_LAYOUT
from validation import PCA_FEATURES, validate_batch_frame, validate_threshold, validate_transaction


def _style_fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(gridcolor="rgba(27,67,50,0.08)", zerolinecolor="rgba(27,67,50,0.15)")
    fig.update_yaxes(gridcolor="rgba(27,67,50,0.08)", zerolinecolor="rgba(27,67,50,0.15)")
    return fig


def inject_css(css: str) -> None:
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def sidebar_brand() -> None:
    st.markdown(
        """
        <div class="brand-mark">Verdant Ledger</div>
        <div class="brand-title">Fraud Desk</div>
        <div class="brand-sub">Private-bank monitoring for the ULB credit-card book. Models stay on-device; nothing leaves this machine.</div>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, copy: str) -> None:
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{copy}</p></div>', unsafe_allow_html=True)


def kpi_strip(kpis: dict) -> None:
    st.markdown(
        f"""
        <div class="kpi-grid">
          <div class="kpi"><div class="kpi-label">Book size</div><div class="kpi-value">{kpis['transactions']:,}</div><div class="kpi-hint">settled transactions</div></div>
          <div class="kpi"><div class="kpi-label">Confirmed fraud</div><div class="kpi-value">{kpis['fraud_count']:,}</div><div class="kpi-hint">{kpis['fraud_rate']:.3%} of volume</div></div>
          <div class="kpi"><div class="kpi-label">Fraud loss</div><div class="kpi-value">${kpis['fraud_loss']:,.0f}</div><div class="kpi-hint">sum of fraudulent amounts</div></div>
          <div class="kpi"><div class="kpi-label">Avg fraud vs legit</div><div class="kpi-value">{kpis['amount_ratio']:.1f}×</div><div class="kpi-hint">${kpis['avg_fraud_amount']:.0f} vs ${kpis['avg_legitimate_amount']:.0f}</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def band_badge(band: str) -> str:
    slug = band.lower()
    return f'<span class="badge badge-{slug}">{band}</span>'


def render_command_center(df_raw: pd.DataFrame, artifacts: dict, demo: bool) -> None:
    hero(
        "Command center",
        "A single desk for the credit-card book: loss exposure, when fraud clusters, and which model currently leads the panel.",
    )
    if demo:
        st.warning("Running in bootstrap mode — `saved_models.pkl` was not found, so a lightweight model was trained on a sample.")

    kpis = compute_kpis(df_raw)
    kpi_strip(kpis)

    champ = champion_model(artifacts["results"])
    champ_f1 = artifacts["results"][champ]["f1"]
    champ_auc = artifacts["results"][champ]["roc_auc"]

    left, right = st.columns((1.35, 1), gap="large")
    with left:
        hourly = fraud_by_hour(df_raw)
        fig = go.Figure()
        fig.add_bar(x=hourly["hour"], y=hourly["frauds"], name="Fraud count", marker_color=COLORS["burgundy"])
        fig.add_trace(
            go.Scatter(
                x=hourly["hour"],
                y=hourly["fraud_rate"] * hourly["frauds"].max() / max(hourly["fraud_rate"].max(), 1e-9),
                name="Rate (scaled)",
                mode="lines+markers",
                line=dict(color=COLORS["brass"], width=2),
            )
        )
        fig.update_layout(title="Fraud by hour of day", xaxis_title="Hour", yaxis_title="Fraudulent transactions")
        st.plotly_chart(_style_fig(fig), width="stretch")

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Champion model")
        st.markdown(f"**{champ}** leads the panel on F1.")
        c1, c2, c3 = st.columns(3)
        c1.metric("F1", f"{champ_f1:.3f}")
        c2.metric("ROC-AUC", f"{champ_auc:.3f}")
        c3.metric("Recall", f"{artifacts['results'][champ]['recall']:.3f}")
        st.caption("Accuracy is a vanity metric here — fraud is 0.17% of the book. We rank models on F1 and recall.")
        st.markdown("</div>", unsafe_allow_html=True)

        buckets = amount_buckets(df_raw)
        fig2 = go.Figure(
            go.Bar(
                x=buckets["bucket"].astype(str),
                y=buckets["fraud_rate"],
                marker_color=COLORS["forest"],
                name="Fraud rate",
            )
        )
        fig2.update_layout(title="Fraud rate by amount band", yaxis_tickformat=".2%")
        st.plotly_chart(_style_fig(fig2), width="stretch")


def render_intelligence(df_raw: pd.DataFrame) -> None:
    hero("Book intelligence", "Class imbalance, spend shape, and which anonymized components actually move with fraud.")
    balance = class_balance(df_raw)
    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure(
            go.Pie(
                labels=balance["label"],
                values=balance["count"],
                hole=0.62,
                marker=dict(colors=[COLORS["forest"], COLORS["burgundy"]]),
            )
        )
        fig.update_layout(title="Class mix")
        st.plotly_chart(_style_fig(fig), width="stretch")
        st.info("Fraud is a rare event. A model that always predicts legitimate still looks 99.8% accurate.")

    with c2:
        fig = go.Figure()
        fig.add_histogram(x=df_raw.loc[df_raw.Class == 0, "Amount"], name="Legitimate", opacity=0.7, marker_color=COLORS["moss"])
        fig.add_histogram(x=df_raw.loc[df_raw.Class == 1, "Amount"], name="Fraud", opacity=0.85, marker_color=COLORS["burgundy"])
        fig.update_layout(barmode="overlay", title="Amount distribution", xaxis_title="Amount ($)", yaxis_type="log")
        st.plotly_chart(_style_fig(fig), width="stretch")

    corr = df_raw.corr(numeric_only=True)["Class"].drop("Class").sort_values()
    fig = go.Figure(
        go.Bar(
            x=corr.values,
            y=corr.index,
            orientation="h",
            marker_color=[COLORS["burgundy"] if v > 0 else COLORS["forest"] for v in corr.values],
        )
    )
    fig.update_layout(title="Correlation with Class", height=720, margin=dict(l=80))
    st.plotly_chart(_style_fig(fig), width="stretch")


def render_pca(artifacts: dict) -> None:
    hero("Risk map", "A 2D PCA projection of the training fold. Fraud rarely sits in the dense legitimate cloud.")
    x_pca = artifacts["X_pca_train"]
    y_train = artifacts["y_train"]
    pca = artifacts["pca"]

    sample_idx = slice(None)
    if len(x_pca) > 12_000:
        # Keep every fraud point; downsample legitimate for a readable scatter.
        y_arr = y_train.to_numpy() if hasattr(y_train, "to_numpy") else y_train
        fraud_i = [i for i, v in enumerate(y_arr) if v == 1]
        legit_i = [i for i, v in enumerate(y_arr) if v == 0][:: max(1, len(y_arr) // 8000)]
        sample_idx = sorted(set(fraud_i + legit_i))
        x_plot = x_pca[sample_idx]
        y_plot = y_arr[sample_idx]
    else:
        x_plot = x_pca
        y_plot = y_train.to_numpy() if hasattr(y_train, "to_numpy") else y_train

    fig = go.Figure()
    fig.add_scatter(
        x=x_plot[y_plot == 0, 0],
        y=x_plot[y_plot == 0, 1],
        mode="markers",
        name="Legitimate",
        marker=dict(size=5, color=COLORS["moss"], opacity=0.28),
    )
    fig.add_scatter(
        x=x_plot[y_plot == 1, 0],
        y=x_plot[y_plot == 1, 1],
        mode="markers",
        name="Fraud",
        marker=dict(size=8, color=COLORS["burgundy"], opacity=0.85),
    )
    var = pca.explained_variance_ratio_
    fig.update_layout(
        title="Training set · 2 principal components",
        xaxis_title=f"PC1 ({var[0]:.1%} var)",
        yaxis_title=f"PC2 ({var[1]:.1%} var)",
        height=560,
    )
    st.plotly_chart(_style_fig(fig), width="stretch")
    st.caption(f"Explained variance: PC1 {var[0]:.3f} · PC2 {var[1]:.3f}. Two axes cannot hold the full book — this is a map, not the model.")


def render_model_lab(artifacts: dict) -> None:
    hero("Model lab", "Compare the panel, inspect errors, and tune the decision threshold against precision and recall.")
    table = metrics_table(artifacts["results"])
    st.subheader("Held-out test performance")
    st.dataframe(table.style.format("{:.4f}"), width="stretch")

    st.subheader("Cross-validation (F1)")
    cv_df = pd.DataFrame(list(artifacts["cv_scores"].items()), columns=["Model", "CV F1"])
    st.table(cv_df)

    names = list(artifacts["models"].keys())
    selected = st.selectbox("Focus model", names, index=0)
    y_test = artifacts["y_test"]
    res = artifacts["results"][selected]

    tcol, rcol = st.columns(2)
    with tcol:
        cm = sklearn.metrics.confusion_matrix(y_test, res["preds"])
        fig = go.Figure(
            go.Heatmap(
                z=cm,
                x=["Predicted legit", "Predicted fraud"],
                y=["True legit", "True fraud"],
                colorscale=[[0, "#FBF7F0"], [1, "#1B4332"]],
                text=cm,
                texttemplate="%{text}",
                showscale=False,
            )
        )
        fig.update_layout(title=f"Confusion matrix · {selected}", height=420)
        st.plotly_chart(_style_fig(fig), width="stretch")

    with rcol:
        fig = go.Figure()
        for name, payload in artifacts["results"].items():
            fpr, tpr, _ = sklearn.metrics.roc_curve(y_test, payload["probas"])
            fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{name} ({payload['roc_auc']:.3f})")
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash", color="#9C7A32"))
        fig.update_layout(title="ROC", xaxis_title="False positive rate", yaxis_title="True positive rate", height=420)
        st.plotly_chart(_style_fig(fig), width="stretch")

    fig = go.Figure()
    for name, payload in artifacts["results"].items():
        prec, rec, _ = sklearn.metrics.precision_recall_curve(y_test, payload["probas"])
        fig.add_scatter(x=rec, y=prec, mode="lines", name=name)
    fig.update_layout(title="Precision–recall", xaxis_title="Recall", yaxis_title="Precision", height=420)
    st.plotly_chart(_style_fig(fig), width="stretch")

    st.subheader("Threshold tuner")
    st.caption("Move the cutoff. Lower catches more fraud (and more false alarms); higher protects the operations queue.")
    threshold = st.slider("Decision threshold", 0.05, 0.95, 0.35, 0.01)
    tuned = threshold_metrics(y_test, res["probas"], threshold)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Precision", f"{tuned['precision']:.3f}")
    m2.metric("Recall", f"{tuned['recall']:.3f}")
    m3.metric("F1", f"{tuned['f1']:.3f}")
    m4.metric("False alarms", f"{tuned['fp']:,}")
    m5.metric("Missed fraud", f"{tuned['fn']:,}")


def render_drivers(artifacts: dict) -> None:
    hero("Risk drivers", "What the tree ensemble actually uses, plus a local explanation for a scored row.")
    models = artifacts["models"]
    if "Random Forest" in models:
        model = models["Random Forest"]
        title = "Random Forest"
    else:
        # Bootstrap fallback — use logistic coefficients as a stand-in.
        model = models.get("Logistic Regression") or next(iter(models.values()))
        title = type(model).__name__

    columns = artifacts["X_train"].columns
    if hasattr(model, "feature_importances_"):
        importances = pd.Series(model.feature_importances_, index=columns).sort_values(ascending=True)
    elif hasattr(model, "coef_"):
        importances = pd.Series(abs(model.coef_[0]), index=columns).sort_values(ascending=True)
    else:
        st.error("The loaded model does not expose importances.")
        return

    top = importances.tail(15)
    fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h", marker_color=COLORS["moss"]))
    fig.update_layout(title=f"{title} · top 15 drivers", height=520)
    st.plotly_chart(_style_fig(fig), width="stretch")

    st.caption("Importances are global. For a single case, use the scoring desk — it ranks |feature × importance| on that row.")
    return importances


def _init_tx_state(df_raw: pd.DataFrame) -> None:
    if "tx_amount" not in st.session_state:
        st.session_state.tx_amount = 88.0
        st.session_state.tx_time = 40_000.0
        for name in PCA_FEATURES:
            st.session_state[f"tx_{name}"] = 0.0


def _load_sample(row: pd.Series) -> None:
    st.session_state.tx_amount = float(row["Amount"])
    st.session_state.tx_time = float(row["Time"])
    for name in PCA_FEATURES:
        st.session_state[f"tx_{name}"] = float(row[name])


def render_scoring_desk(df_raw: pd.DataFrame, artifacts: dict) -> None:
    hero(
        "Scoring desk",
        "Validate a transaction, run the panel, and see heuristic flags beside the model probability. V1–V28 are anonymized PCA components — load a real row if you want a meaningful score.",
    )
    _init_tx_state(df_raw)

    b1, b2, b3 = st.columns(3)
    if b1.button("Load a known fraud"):
        _load_sample(df_raw[df_raw.Class == 1].sample(1, random_state=None).iloc[0])
        st.rerun()
    if b2.button("Load a legitimate row"):
        _load_sample(df_raw[df_raw.Class == 0].sample(1, random_state=None).iloc[0])
        st.rerun()
    if b3.button("Reset to zeros"):
        st.session_state.tx_amount = 88.0
        st.session_state.tx_time = 40_000.0
        for name in PCA_FEATURES:
            st.session_state[f"tx_{name}"] = 0.0
        st.rerun()

    model_names = ["Ensemble (mean probability)", *artifacts["models"].keys()]
    chosen = st.selectbox("Model", model_names)
    threshold = st.slider("Flag threshold", 0.05, 0.95, 0.35, 0.01, key="score_threshold")

    c1, c2 = st.columns(2)
    amount = c1.number_input("Amount ($)", min_value=0.0, max_value=250000.0, step=1.0, key="tx_amount")
    time_value = c2.number_input("Time since first txn (seconds)", min_value=0.0, max_value=200000.0, step=1.0, key="tx_time")

    st.markdown("##### Anonymized components (V1–V28)")
    v_features = {}
    cols = st.columns(4)
    for i, name in enumerate(PCA_FEATURES):
        v_features[name] = cols[i % 4].number_input(name, step=0.1, key=f"tx_{name}")

    if st.button("Score transaction", type="primary"):
        check = validate_transaction(amount, time_value, v_features)
        if not check.ok:
            for err in check.errors:
                st.error(err)
            return
        for warn in check.warnings:
            st.warning(warn)

        input_df = build_model_input(
            amount,
            time_value,
            v_features,
            artifacts["amount_scaler"],
            artifacts["time_scaler"],
            artifacts["X_train"].columns,
        )

        if chosen.startswith("Ensemble"):
            outcome = ensemble_predict(artifacts["models"], input_df, threshold)
        else:
            outcome = predict_with_threshold(artifacts["models"][chosen], input_df, threshold)
            outcome["per_model"] = {chosen: outcome["probability"]}

        flags = heuristic_flags(amount, time_value, v_features)
        st.markdown("---")
        st.markdown(
            f"### {outcome['label']} &nbsp; {band_badge(outcome['band'])}",
            unsafe_allow_html=True,
        )
        p1, p2, p3 = st.columns(3)
        p1.metric("Fraud probability", f"{outcome['probability']:.1%}")
        p2.metric("Threshold", f"{outcome['threshold']:.0%}")
        if "votes" in outcome:
            p3.metric("Panel votes", f"{outcome['votes']} / {outcome['panel_size']}")
        else:
            p3.metric("Decision", outcome["label"])

        if outcome["prediction"]:
            st.error("Hold and route to investigation. Do not auto-clear.")
        else:
            st.success("Below threshold. Heuristic flags can still justify a manual look.")

        if flags:
            st.subheader("Heuristic flags")
            st.dataframe(pd.DataFrame(flags), width="stretch", hide_index=True)
        else:
            st.caption("No heuristic flags fired.")

        if "per_model" in outcome:
            fig = go.Figure(
                go.Bar(
                    x=list(outcome["per_model"].keys()),
                    y=list(outcome["per_model"].values()),
                    marker_color=COLORS["forest"],
                )
            )
            fig.update_layout(title="Panel probabilities", yaxis_tickformat=".0%", yaxis_range=[0, 1])
            st.plotly_chart(_style_fig(fig), width="stretch")

        model_for_imp = artifacts["models"].get("Random Forest") or next(iter(artifacts["models"].values()))
        if hasattr(model_for_imp, "feature_importances_"):
            imps = pd.Series(model_for_imp.feature_importances_, index=artifacts["X_train"].columns)
        elif hasattr(model_for_imp, "coef_"):
            imps = pd.Series(abs(model_for_imp.coef_[0]), index=artifacts["X_train"].columns)
        else:
            imps = None
        if imps is not None:
            contrib = feature_contributions(imps, input_df.iloc[0], top_n=8)
            fig = go.Figure(go.Bar(x=contrib.values, y=contrib.index, orientation="h", marker_color=COLORS["brass"]))
            fig.update_layout(title="Local drivers (|value| × importance)", height=360)
            st.plotly_chart(_style_fig(fig), width="stretch")


def render_queue(artifacts: dict) -> None:
    hero(
        "Investigation queue",
        "Highest-probability cases on the held-out test fold. Review, clear, or confirm — decisions stay in this session.",
    )
    if "reviews" not in st.session_state:
        st.session_state.reviews = {}

    champ = champion_model(artifacts["results"])
    model = artifacts["models"][champ]
    x_test = artifacts["X_test"]
    y_test = artifacts["y_test"]
    probas = artifacts["results"][champ]["probas"]
    amounts = artifacts["amount_scaler"].inverse_transform(x_test[["scaled_amount"]]).ravel()
    times = artifacts["time_scaler"].inverse_transform(x_test[["scaled_time"]]).ravel()

    queue = pd.DataFrame(
        {
            "case_id": x_test.index.astype(str),
            "probability": probas,
            "actual": y_test.to_numpy() if hasattr(y_test, "to_numpy") else y_test,
            "amount": amounts,
            "time": times,
        }
    )
    queue["band"] = queue["probability"].map(lambda p: risk_band(float(p)))
    queue = queue.sort_values("probability", ascending=False).head(75)
    queue["review"] = queue["case_id"].map(lambda i: st.session_state.reviews.get(i, "Pending"))

    min_p = st.slider("Minimum probability", 0.0, 1.0, 0.5, 0.05)
    view = queue[queue["probability"] >= min_p]
    st.caption(f"Champion: **{champ}**. Showing {len(view)} cases.")
    st.dataframe(
        view[["case_id", "probability", "band", "amount", "actual", "review"]].style.format(
            {"probability": "{:.1%}", "amount": "${:,.2f}"}
        ),
        width="stretch",
        hide_index=True,
    )

    if view.empty:
        st.info("No cases above this cutoff.")
        return

    selected = st.selectbox("Open case", view["case_id"].tolist())
    row = view.set_index("case_id").loc[selected]
    a, b, c = st.columns(3)
    a.metric("Probability", f"{row['probability']:.1%}")
    b.metric("Amount", f"${row['amount']:,.2f}")
    c.metric("True label", "Fraud" if int(row["actual"]) == 1 else "Legitimate")

    r1, r2, r3 = st.columns(3)
    if r1.button("Confirm fraud"):
        st.session_state.reviews[selected] = "Confirmed fraud"
        st.rerun()
    if r2.button("Clear as legitimate"):
        st.session_state.reviews[selected] = "Cleared"
        st.rerun()
    if r3.button("Mark needs more info"):
        st.session_state.reviews[selected] = "Needs info"
        st.rerun()

    counts = pd.Series(st.session_state.reviews).value_counts() if st.session_state.reviews else pd.Series(dtype=int)
    st.subheader("Session dispositions")
    if counts.empty:
        st.caption("No reviews yet this session.")
    else:
        st.bar_chart(counts)

    st.download_button(
        "Export queue CSV",
        view.to_csv(index=False),
        file_name="investigation_queue.csv",
        mime="text/csv",
    )


def render_batch(artifacts: dict) -> None:
    hero(
        "Batch monitor",
        "Score a CSV of transactions. Required columns: Time, Amount, V1…V28. Maximum 5,000 rows. Invalid files are rejected before anything is scored.",
    )
    champ = champion_model(artifacts["results"])
    model_name = st.selectbox("Scoring model", list(artifacts["models"].keys()), index=list(artifacts["models"].keys()).index(champ) if champ in artifacts["models"] else 0)
    threshold = st.slider("Batch threshold", 0.05, 0.95, 0.35, 0.01, key="batch_threshold")
    thresh_check = validate_threshold(threshold)
    for warn in thresh_check.warnings:
        st.warning(warn)

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded is None:
        st.caption("Tip: export a slice of `creditcard.csv` without the Class column, or keep Class to compare against labels.")
        return

    try:
        frame = pd.read_csv(uploaded)
    except Exception:
        st.error("Could not parse that file as CSV.")
        return

    check = validate_batch_frame(frame)
    if not check.ok:
        for err in check.errors:
            st.error(err)
        return

    scored = score_batch(
        artifacts["models"][model_name],
        frame,
        artifacts["amount_scaler"],
        artifacts["time_scaler"],
        artifacts["X_train"].columns,
        threshold,
    )
    flagged = int(scored["prediction"].sum())
    k1, k2, k3 = st.columns(3)
    k1.metric("Rows scored", f"{len(scored):,}")
    k2.metric("Flagged", f"{flagged:,}")
    k3.metric("Flag rate", f"{flagged / len(scored):.1%}")

    if "Class" in scored.columns:
        y = scored["Class"].astype(int)
        p = scored["prediction"]
        st.write(
            f"If Class is the true label: precision {sklearn.metrics.precision_score(y, p, zero_division=0):.3f} · "
            f"recall {sklearn.metrics.recall_score(y, p, zero_division=0):.3f} · "
            f"F1 {sklearn.metrics.f1_score(y, p, zero_division=0):.3f}"
        )

    st.dataframe(
        scored[["Time", "Amount", "fraud_probability", "prediction", "risk_band"]].head(200).style.format(
            {"fraud_probability": "{:.1%}", "Amount": "${:,.2f}"}
        ),
        width="stretch",
    )
    st.download_button("Download scored file", scored.to_csv(index=False), file_name="scored_transactions.csv", mime="text/csv")
