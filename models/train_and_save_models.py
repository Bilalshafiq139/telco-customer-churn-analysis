"""
Train and save the churn prediction models used by the Streamlit app.
"""

import pickle
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
NUMERIC_COLUMNS = ["tenure", "MonthlyCharges", "TotalCharges"]


def print_model_report(model_name, model, X_test, y_test):
    """Print accuracy, classification report, and confusion matrix for a model."""
    y_pred = model.predict(X_test)
    accuracy = model.score(X_test, y_test)

    print(f"\n{model_name} test accuracy: {accuracy:.4f}")
    print(f"\n{model_name} classification report:")
    print(classification_report(y_test, y_pred))
    print(f"{model_name} confusion matrix:")
    print(confusion_matrix(y_test, y_pred))


def train_and_save_models():
    """Load train/test splits, train models, and save app artifacts."""
    split_path = DATA_DIR / "train_test_splits.pkl"
    with open(split_path, "rb") as f:
        X_train, X_test, y_train, y_test = pickle.load(f)

    print("Data loaded successfully")
    print(f"Features shape: {X_train.shape}")

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[NUMERIC_COLUMNS] = scaler.fit_transform(X_train[NUMERIC_COLUMNS])
    X_test_scaled[NUMERIC_COLUMNS] = scaler.transform(X_test[NUMERIC_COLUMNS])
    print(f"Scaler fitted on {len(NUMERIC_COLUMNS)} numeric features")

    print("Training Logistic Regression model...")
    lr_model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    print_model_report("Logistic Regression", lr_model, X_test_scaled, y_test)

    print("Training Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        class_weight="balanced",
    )
    rf_model.fit(X_train_scaled, y_train)
    print_model_report("Random Forest", rf_model, X_test_scaled, y_test)

    artifacts = {
        "logistic_regression_model.pkl": lr_model,
        "random_forest_model.pkl": rf_model,
        "scaler.pkl": scaler,
        "feature_names.pkl": X_train.columns.tolist(),
    }

    MODELS_DIR.mkdir(exist_ok=True)
    for filename, artifact in artifacts.items():
        with open(MODELS_DIR / filename, "wb") as f:
            pickle.dump(artifact, f)
        print(f"Saved {filename}")

    print("All model artifacts saved successfully")


if __name__ == "__main__":
    train_and_save_models()
