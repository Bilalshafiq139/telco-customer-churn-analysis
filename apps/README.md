# Customer Churn Prediction App

## Overview

This Streamlit application provides a user-friendly interface to predict customer churn probability using machine learning models trained on historical customer data.

## Features

Key Features:
- Dual Model Support: Choose between Random Forest (77.26% accuracy) and Logistic Regression (72.57% accuracy)
- Interactive Input Form: User-friendly interface to enter customer information
- Real-time Predictions: Instant churn probability predictions
- Risk Assessment: Visual indicators showing low/medium/high churn risk
- Prediction Logging: Automatic saving of all predictions to a CSV file
- CSV Export: Download individual predictions as CSV files

## Installation

### 1. Install Required Packages

```bash
pip install -r requirements.txt
```

Or manually install:
```bash
pip install pandas matplotlib seaborn numpy jupyter scikit-learn streamlit
```

### 2. Ensure Model Files Exist

The following files should be in the `models/` directory:
- `random_forest_model.pkl` - Trained Random Forest model
- `logistic_regression_model.pkl` - Trained Logistic Regression model
- `scaler.pkl` - StandardScaler for numeric features
- `feature_names.pkl` - List of feature names

If these files don't exist, run the training script:
```bash
cd models/
python train_and_save_models.py
cd ..
```

## Running the App

### Option 1: From the Project Root
```bash
streamlit run apps/prediction.py
```

### Option 2: From the Apps Directory
```bash
cd apps/
streamlit run prediction.py
```

The app will open in your default web browser at `http://localhost:8501`

## How to Use

### Step 1: Configure Model
- Use the sidebar to select your preferred model (Random Forest or Logistic Regression)

### Step 2: Enter Customer Information

Fill in the following sections:

**Demographics:**
- Gender (Male/Female)
- Senior Citizen status
- Partner status
- Dependents

**Service Duration:**
- Tenure (in months)

**Billing Information:**
- Monthly Charges ($)
- Total Charges ($)

**Contract & Services:**
- Contract Type (Month-to-Month, One Year, Two Years)
- Internet Service (DSL, Fiber Optic, No Internet Service)
- Payment Method

**Additional Services:**
- Phone Service
- Online Security
- Online Backup
- Device Protection
- Tech Support
- Streaming TV
- Streaming Movies
- Paperless Billing
- Multiple Lines

### Step 3: Make Prediction
Click the "Predict Churn" button to generate predictions

### Step 4: Review Results
The app will display:
- **Churn Probability**: Percentage likelihood the customer will leave
- **Retention Probability**: Percentage likelihood the customer will stay
- **Risk Level**: Visual indicator (Low/Medium/High)
- **Prediction**: Churn or No Churn

### Step 5: Save Results (Optional)
- **Download CSV**: Export individual prediction
- **Automatic Logging**: All predictions are automatically saved to `data/predictions_log.csv`

## Data Structure

### Input Features (24 total)

**Numeric Features (3):**
- Tenure: 0-72 months
- Monthly Charges: $18.00-$120.00
- Total Charges: $0-$8600.00

**Binary Features (13):**
- Gender, SeniorCitizen, Partner, Dependents
- PhoneService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport
- StreamingTV, StreamingMovies, PaperlessBilling

**Categorical Features (8 - one-hot encoded):**
- Contract: Month-to-Month (default), One Year, Two Years
- Internet Service: DSL (default), Fiber Optic, No Internet Service
- Payment Method: 4 types (Bank Transfer, Credit Card, Electronic Check, Mailed Check)
- Multiple Lines: No (default), No Phone Service, Yes

## Output Format

### Predictions Log CSV Columns
- Timestamp
- Model (Random Forest / Logistic Regression)
- All customer input features
- Churn_Probability (%)
- Retention_Probability (%)
- Prediction (Churn / No Churn)

### Risk Classification
- LOW RISK: < 40% churn probability
- MEDIUM RISK: 40-60% churn probability
- HIGH RISK: > 60% churn probability

## Model Performance

### Random Forest
- Accuracy: 77.26%
- Better at capturing complex patterns
- Recommended for production use
- Higher recall on churn detection

### Logistic Regression
- Accuracy: 72.57%
- Faster inference time
- More interpretable results
- Good baseline model

## Troubleshooting

### Issue: "Error loading models"
**Solution:** Ensure all pickle files exist in the `models/` directory. Run:
```bash
python models/train_and_save_models.py
```

### Issue: "ModuleNotFoundError: No module named 'streamlit'"
**Solution:** Install Streamlit:
```bash
pip install streamlit
```

### Issue: "FileNotFoundError: No such file or directory"
**Solution:** Make sure you're running the app from the project root directory:
```bash
cd "c:\Users\HP\Desktop\dataanlytics\churn analysis"
streamlit run apps/prediction.py
```

## Application Architecture

```
churn analysis/
├── apps/
│   ├── __init__.py
│   ├── prediction.py          # Main Streamlit app
│   └── utils.py               # Utility functions & preprocessing
├── models/
│   ├── random_forest_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   └── train_and_save_models.py
├── data/
│   ├── churn_raw.csv
│   ├── cleaned_dataset_for_EDA.csv
│   ├── data_for_model_building.csv
│   ├── train_test_splits.pkl
│   └── predictions_log.csv    # Auto-generated by app
├── notebooks/
│   ├── 1_data_understanding.ipynb
│   ├── 2_data_cleaning.ipynb
│   ├── 3_EDA_analysis.ipynb
│   ├── 4_feature_engineering.ipynb
│   ├── 5_model_building.ipynb
│   ├── 6_model_evaluation.ipynb
│   └── 7_business_recommendations.ipynb
├── README.md
└── requirements.txt
```

## Business Use Cases

### 1. **Identify At-Risk Customers**
- Use HIGH RISK predictions for immediate retention outreach
- Allocate retention resources efficiently

### 2. **Proactive Retention Strategies**
- Contact MEDIUM RISK customers with special offers
- Bundle additional services to increase switching costs

### 3. **Customer Segmentation**
- Identify patterns in churn predictors
- Create targeted retention campaigns

### 4. **Cost-Benefit Analysis**
- Calculate retention cost vs. acquisition cost
- ROI on retention campaigns vs. new customer acquisition

## Feature Importance Insights

Based on the trained models, top churn drivers are:

1. **Contract Type** - Month-to-month contracts have highest churn
2. **Tenure** - New customers (< 12 months) are high risk
3. **Monthly/Total Charges** - Price sensitivity drives churn
4. **Internet Service Type** - Fiber optic has different patterns
5. **Payment Method** - Electronic check users churn more

## Future Enhancements

Planned Features:
- Batch prediction upload (CSV input)
- Customer segmentation analysis
- Recommendation engine for retention strategies
- API endpoint for integration
- Historical trend analysis
- A/B testing framework for retention campaigns

## Support & Questions

For issues, questions, or feature requests:
1. Check the troubleshooting section above
2. Review model training notebook (5_model_building.ipynb)
3. Check model evaluation notebook (6_model_evaluation.ipynb)

---

**Last Updated:** 2026-03-18
**Model Training Date:** Latest run from train_and_save_models.py
