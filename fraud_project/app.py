import pandas as pd
import streamlit as st

from artifacts_io import load_artifacts
from dashboard import (
    inject_css,
    render_batch,
    render_command_center,
    render_drivers,
    render_intelligence,
    render_model_lab,
    render_pca,
    render_queue,
    render_scoring_desk,
    sidebar_brand,
)
from theme import CSS

st.set_page_config(
    page_title="Verdant Ledger · Fraud Desk",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css(CSS)


@st.cache_resource
def _artifacts():
    return load_artifacts()


@st.cache_data
def _raw():
    return pd.read_csv("creditcard.csv")


try:
    artifacts, demo = _artifacts()
    df_raw = _raw()
except FileNotFoundError:
    st.error("creditcard.csv is missing. Place the Kaggle ULB file in this folder and reload.")
    st.stop()

with st.sidebar:
    sidebar_brand()
    st.markdown("---")
    menu = st.radio(
        "Desk",
        [
            "Command center",
            "Book intelligence",
            "Risk map",
            "Model lab",
            "Risk drivers",
            "Scoring desk",
            "Investigation queue",
            "Batch monitor",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption("24CSE0316 · AI and Machine Learning")
    if demo:
        st.caption("Bootstrap models — run `python train.py` for the full panel.")

if menu == "Command center":
    render_command_center(df_raw, artifacts, demo)
elif menu == "Book intelligence":
    render_intelligence(df_raw)
elif menu == "Risk map":
    render_pca(artifacts)
elif menu == "Model lab":
    render_model_lab(artifacts)
elif menu == "Risk drivers":
    render_drivers(artifacts)
elif menu == "Scoring desk":
    render_scoring_desk(df_raw, artifacts)
elif menu == "Investigation queue":
    render_queue(artifacts)
else:
    render_batch(artifacts)
