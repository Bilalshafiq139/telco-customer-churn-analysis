"""
Script to train and save the churn prediction models
This ensures models are available for the Streamlit app
"""

import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def train_and_save_models():
    """Load data, train models, and save them for Streamlit app"""
    
    # Load train/test splits
    with open("../data/train_test_splits.pkl", "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)
    
    print("Data loaded successfully!")
    print(f"Features shape: {X_train.shape}")
    
    # Step 1: Create and fit scaler on numeric columns
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    scaler = StandardScaler()
    
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    print(f"\nScaler fitted on {len(numeric_cols)} numeric features")
    
    # Step 2: Train Logistic Regression Model
    print("\nTraining Logistic Regression model...")
    lr_model = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    
    # Get accuracy
    lr_accuracy = lr_model.score(X_test_scaled, y_test)
    print(f"Logistic Regression Test Accuracy: {lr_accuracy:.4f}")
    
    # Step 3: Train Random Forest Model
    print("\nTraining Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight='balanced'
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # Get accuracy
    rf_accuracy = rf_model.score(X_test_scaled, y_test)
    print(f"Random Forest Test Accuracy: {rf_accuracy:.4f}")
    
    # Step 4: Save models and scaler
    print("\nSaving models and scaler...")
    
    with open("logistic_regression_model.pkl", "wb") as f:
        pickle.dump(lr_model, f)
    print("✓ Logistic Regression model saved")
    
    with open("random_forest_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    print("✓ Random Forest model saved")
    
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("✓ Scaler saved")
    
    # Step 5: Save feature names for reference
    with open("feature_names.pkl", "wb") as f:
        pickle.dump(X_train.columns.tolist(), f)
    print("✓ Feature names saved")
    
    print("\n" + "="*50)
    print("✓ All models and artifacts saved successfully!")
    print("="*50)


if __name__ == "__main__":
    train_and_save_models()
