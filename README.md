# Telco Customer Churn Analysis & Prediction
![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=flat&logo=Matplotlib&logoColor=black)
![Seaborn](https://img.shields.io/badge/Seaborn-%234479A1.svg?style=flat&logo=Seaborn&logoColor=white)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)
![Random Forest](https://img.shields.io/badge/Algorithm-Random%20Forest-green)
![Logistic Regression](https://img.shields.io/badge/Algorithm-Logistic%20Regression-blue)


### Project Overview
This project predicts customer churn for a telecom company using machine learning. It combines exploratory data analysis (EDA), feature engineering, model building, and actionable business recommendations.

### Dataset
- Source: [Kaggle - Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn)
- Rows: 7,032
- Columns: 21
- Target Variable: `Churn` (0 = Stayed, 1 = Left)

### Key Insights
- 26.5% of customers churned, making the dataset imbalanced.
- High-risk churners: Month-to-month contracts, Fiber optic users, low tenure, manual payments, lack of tech support.
- Random Forest model achieved:
  - Accuracy: 77%
  - Churn recall: 73% (crucial for retention)

### Modeling Approach
1. Data Cleaning & Preprocessing
2. Feature Engineering (Binary Mapping, One-Hot Encoding)
3. Model Building:
   - Logistic Regression
   - Random Forest (best recall)
4. Evaluation:
   - Confusion Matrix
   - Precision, Recall, F1-Score
5. Business Recommendations

### Actionable Recommendations
- Focus retention campaigns on high-risk segments.
- Upsell Tech Support & Online Security.
- Encourage automated payment methods.
- Improve Fiber optic customer satisfaction.

### Visualizations
- Churn distribution
- Feature importance (Random Forest)
- Tenure vs Churn boxplots

---

##  Customer Churn Prediction App

We've built an interactive **Streamlit application** that allows anyone to predict customer churn probability using our trained machine learning models!

### Quick Start

#### Option 1: Windows Users
Double-click `run_app.bat` or run:
```bash
run_app.bat
```

#### Option 2: Mac/Linux Users
Run:
```bash
bash run_app.sh
```

#### Option 3: Universal (Any OS)
```bash
streamlit run apps/prediction.py
```

The app will open at `http://localhost:8501`

### App Features
 **Interactive Churn Prediction**
- Enter customer data through an easy-to-use form
- Choose between two models (Random Forest or Logistic Regression)
- Get instant churn probability predictions
- Visual risk assessment (Low/Medium/High)
- Save predictions to CSV files
- Automatic prediction logging

### How to Use
1. Select your preferred model (Random Forest recommended)
2. Fill in customer information:
   - Demographics (gender, age, dependents, etc.)
   - Service details (tenure, charges, contract type)
   - Additional services (phone, internet, security, etc.)
3. Click " Predict Churn"
4. Review prediction results
5. Download or save predictions

### Model Comparison
| Metric | Random Forest | Logistic Regression |
|--------|--------------|-------------------|
| Accuracy | 77.26% | 72.57% |
| Churn Recall | 73% | 56% |
| Speed | Moderate | Fast |
| Recommendation | Primary | Secondary |

### App Structure
```
apps/
├── prediction.py          # Main Streamlit application
├── utils.py              # Preprocessing & model utilities
├── __init__.py           # Package initialization
└── README.md             # Detailed app documentation
```

### Model Files Location
```
models/
├── random_forest_model.pkl
├── logistic_regression_model.pkl
├── scaler.pkl
└── feature_names.pkl
```

### Output
The app automatically saves all predictions to:
```
data/predictions_log.csv
```

This file tracks:
- Timestamp of prediction
- Model used
- All customer input features
- Churn probability (%)
- Retention probability (%)
- Final prediction

### Requirements
All requirements are listed in `requirements.txt`:
```
pandas
matplotlib
seaborn
numpy
jupyter
scikit-learn
streamlit
```

Install with:
```bash
pip install -r requirements.txt
```

### Troubleshooting
**Q: The app won't start**
A: Make sure you're in the project root directory and all requirements are installed:
```bash
pip install -r requirements.txt
streamlit run apps/prediction.py
```

**Q: Where do I find my prediction results?**
A: Check `data/predictions_log.csv` for all saved predictions, or download individual predictions from the app.

**Q: Which model should I use?**
A: Random Forest is recommended for best accuracy (77.26%) and better churn detection (73% recall).

---
- Contract type vs Churn rate
- Payment method vs Churn rate

### Tech Stack
- Python 3.12
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn


