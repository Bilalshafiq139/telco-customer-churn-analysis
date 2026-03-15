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
- Contract type vs Churn rate
- Payment method vs Churn rate

### Tech Stack
- Python 3.12
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn


