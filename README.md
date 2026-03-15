# Telco Customer Churn Analysis & Prediction

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


