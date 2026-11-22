# Alledger
An interactive Streamlit dashboard for exploratory data analysis, model benchmarking, and real-time transaction fraud prediction using the ULB Credit Card Fraud dataset.

## Setup

1. Download `creditcard.csv` from the [Kaggle ULB Credit Card Fraud Dataset](https://kaggle.com) and place it in the project root folder.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Train the models:**
   ```bash
   python train.py
   ```
   *Note: You only need to run this once (or whenever `train.py` is modified). It generates `saved_models.pkl` and may take a few minutes due to KNN and Random Forest training.*

2. **Launch the dashboard:**
   ```bash
   streamlit run app.py
   ```
   The app will open automatically in your browser at `http://localhost:8501`.

## Project Structure

| File | Description |
| :--- | :--- |
| `data_preprocessing.py` | Loads `creditcard.csv`, scales `Amount` and `Time`, and splits the data into training and testing sets. |
| `train.py` | Trains and evaluates 5 classifiers (Logistic Regression, Decision Tree, Random Forest, Naive Bayes, and KNN) and saves the results to `saved_models.pkl`. |
| `app.py` | Streamlit web application featuring EDA, PCA projections, model comparisons, feature importance, and a custom transaction tester. |
| `saved_models.pkl` | Generated locally by running `train.py` (excluded from Git due to file size). |
| `creditcard.csv` | Source dataset downloaded from Kaggle (excluded from Git due to file size). |
