# Credit Card Fraud Detection Dashboard

Course project for 24CSE0316 (AI and Machine Learning).

## Setup

1. Put `creditcard.csv` (Kaggle, ULB dataset) in this folder.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Run

1. Train the models (only needs to be done once, or whenever you change something in `train.py`):
   ```
   python train.py
   ```
   This creates `saved_models.pkl` in this folder. Takes a few minutes - KNN and Random Forest are the slow ones.

2. Launch the dashboard:
   ```
   streamlit run app.py
   ```
   This opens in your browser automatically (usually `localhost:8501`).

## Files

- `data_preprocessing.py` - loads the CSV, scales Amount/Time, splits into train/test
- `train.py` - trains all 5 models (Logistic Regression, Decision Tree, Random Forest, Naive Bayes, KNN), evaluates them, saves everything to `saved_models.pkl`
- `app.py` - Streamlit dashboard that reads `saved_models.pkl` and shows EDA, PCA projection, model comparison, feature importance, and a page to test a custom transaction
- `saved_models.pkl` / `creditcard.csv` - not included here (too large for git) - generate the pkl yourself by running `train.py`, get the CSV from Kaggle
