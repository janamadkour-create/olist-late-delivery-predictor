# Olist Late Delivery Predictor

A Streamlit app that estimates late-delivery risk for Brazilian e-commerce orders using a saved Random Forest classifier. The app supports single-order and CSV batch predictions.

> Risk scores are model estimates for prioritizing review. They are not calibrated probabilities or guarantees of delivery outcomes.

## Run locally

Use Python 3.12 (the saved model was created with scikit-learn 1.6.1; that version supports Python 3.9–3.13).

```powershell
cd olist-late-delivery-predictor
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local address Streamlit prints, usually http://localhost:8501.

If the Python launcher command `py -3.12` is unavailable, create the environment with the full path to your Python 3.12 executable instead.

## Using the app

- Enter order and delivery features in the single-order form, choose a prediction strategy, and submit.
- For batch prediction, download the CSV template, fill one row per order, upload it, choose a strategy, and run predictions.
- The batch uploader expects the engineered features listed in `artifacts/feature_schema.json`. It does not accept raw Olist tables directly.

## Model snapshot

The included metadata reports ROC-AUC 0.7625, average precision 0.1738, and Brier score 0.1095. These values are copied from the saved model metadata; the training notebook and evaluation methodology should be added to this repository when available.

## Data

The project uses the Brazilian E-Commerce Public Dataset by Olist, available on [Kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce). Raw data files are excluded from this repository. Download them from the source if needed for analysis.

## Project contents

- `olist-late-delivery-predictor/app.py` — Streamlit app
- `olist-late-delivery-predictor/artifacts/` — saved model, feature schema, and metadata required by the app
- `olist-late-delivery-predictor/requirements.txt` — Python dependencies
- `olist_prediction_template.csv` — sample engineered-feature row for batch input
