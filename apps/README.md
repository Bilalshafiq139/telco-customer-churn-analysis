# Customer Churn Prediction App

Streamlit app for predicting telecom customer churn from customer profile, contract, billing, and service inputs.

## Run From Project Root

```bash
pip install -r requirements.txt
python models/train_and_save_models.py
streamlit run apps/prediction.py
```

The app opens at `http://localhost:8501`.

## Required Model Artifacts

The app expects these generated files in `models/`:

```text
random_forest_model.pkl
logistic_regression_model.pkl
scaler.pkl
feature_names.pkl
```

If they are missing, regenerate them with:

```bash
python models/train_and_save_models.py
```

## Features

- Random Forest and Logistic Regression model selection
- Customer profile form with demographics, billing, contract, and services
- Churn probability and retention probability output
- CSV download for individual predictions
- Automatic prediction logging to `data/predictions_log.csv`

## Risk Bands

| Churn probability | Risk level |
| ---: | --- |
| Less than 40% | Low |
| 40% to 60% | Medium |
| More than 60% | High |

## Notes

Model artifacts and prediction logs are generated locally and ignored by Git. This keeps the repository lightweight while making the app fully reproducible.
