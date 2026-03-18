"""
Preprocessing and utility functions for churn prediction app
Handles data formatting and model preparation
"""

import pickle
import pandas as pd
import numpy as np


class ChurnPredictor:
    """Handle model loading and predictions"""
    
    def __init__(self, model_path, scaler_path, feature_names_path):
        """Initialize the predictor with model and preprocessing artifacts"""
        self.model = self._load_pickle(model_path)
        self.scaler = self._load_pickle(scaler_path)
        self.feature_names = self._load_pickle(feature_names_path)
        self.numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    
    @staticmethod
    def _load_pickle(filepath):
        """Load pickle file"""
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    def preprocess_input(self, input_data):
        """
        Preprocess user input to match training format
        Args:
            input_data: dictionary with user inputs
        Returns:
            preprocessed DataFrame ready for prediction
        """
        # Create DataFrame from input
        df = pd.DataFrame({feature: [input_data.get(feature, 0)] for feature in self.feature_names})
        
        # Ensure correct data types
        df = df.astype(int)
        
        # Scale numeric columns
        df_scaled = df.copy()
        df_scaled[self.numeric_cols] = self.scaler.transform(df[self.numeric_cols])
        
        return df_scaled
    
    def predict(self, input_data):
        """
        Make prediction for churn likelihood
        Args:
            input_data: dictionary with customer features
        Returns:
            prediction (0/1) and probability
        """
        df_processed = self.preprocess_input(input_data)
        
        # Get prediction and probability
        prediction = self.model.predict(df_processed)[0]
        probability = self.model.predict_proba(df_processed)[0]
        
        return {
            'prediction': prediction,
            'probability_no_churn': probability[0],
            'probability_churn': probability[1]
        }


def get_feature_input_specs():
    """
    Return specifications for each feature including display name and input type
    """
    specs = {
        # Demographic Features (Binary - 0/1)
        'gender': {'display': 'Gender', 'type': 'binary', 'options': {'Male': 1, 'Female': 0}},
        'SeniorCitizen': {'display': 'Senior Citizen', 'type': 'binary', 'options': {'No': 0, 'Yes': 1}},
        'Partner': {'display': 'Partner', 'type': 'binary', 'options': {'No': 0, 'Yes': 1}},
        'Dependents': {'display': 'Dependents', 'type': 'binary', 'options': {'No': 0, 'Yes': 1}},
        
        # Numeric Features
        'tenure': {'display': 'Tenure (months)', 'type': 'numeric', 'min': 0, 'max': 72, 'default': 12},
        'MonthlyCharges': {'display': 'Monthly Charges ($)', 'type': 'numeric', 'min': 18.0, 'max': 120.0, 'default': 65.0},
        'TotalCharges': {'display': 'Total Charges ($)', 'type': 'numeric', 'min': 0, 'max': 8600, 'default': 1000},
        
        # Service Features (Binary)
        'PhoneService': {'display': 'Phone Service', 'type': 'binary', 'options': {'No': 0, 'Yes': 1}},
        'OnlineSecurity': {'display': 'Online Security', 'type': 'binary', 'options': {'No': 0, 'Yes': 1}},
        'OnlineBackup': {'display': 'Online Backup', 'type': 'binary', 'options': {'No': 0, 'Yes': 1}},
        'DeviceProtection': {'display': 'Device Protection', 'type': 'binary', 'options': {'No': 0, 'Yes': 1}},
        'TechSupport': {'display': 'Tech Support', 'type': 'binary', 'options': {'No': 0, 'Yes': 1}},
        'StreamingTV': {'display': 'Streaming TV', 'type': 'binary', 'options': {'No': 0, 'Yes': 1}},
        'StreamingMovies': {'display': 'Streaming Movies', 'type': 'binary', 'options': {'No': 0, 'Yes': 1}},
        'PaperlessBilling': {'display': 'Paperless Billing', 'type': 'binary', 'options': {'No': 0, 'Yes': 1}},
        
        # Contract Types (mutually exclusive)
        'Contract_One year': {'display': 'Contract: One Year', 'type': 'contract'},
        'Contract_Two year': {'display': 'Contract: Two Years', 'type': 'contract'},
        
        # Internet Service (mutually exclusive)
        'InternetService_Fiber optic': {'display': 'Internet: Fiber Optic', 'type': 'internet'},
        'InternetService_No': {'display': 'Internet: No Internet', 'type': 'internet'},
        
        # Payment Method (mutually exclusive)
        'PaymentMethod_Credit card (automatic)': {'display': 'Payment: Credit Card (Auto)', 'type': 'payment'},
        'PaymentMethod_Electronic check': {'display': 'Payment: Electronic Check', 'type': 'payment'},
        'PaymentMethod_Mailed check': {'display': 'Payment: Mailed Check', 'type': 'payment'},
        
        # Multiple Lines (mutually exclusive)
        'MultipleLines_No phone service': {'display': 'Multiple Lines: No Service', 'type': 'multiline'},
        'MultipleLines_Yes': {'display': 'Multiple Lines: Yes', 'type': 'multiline'},
    }
    return specs


def get_categorical_options():
    """Get options for categorical features"""
    return {
        'Contract': ['Month-to-Month', 'One Year', 'Two Years'],
        'InternetService': ['DSL', 'Fiber Optic', 'No Internet Service'],
        'PaymentMethod': ['Bank Transfer (Automatic)', 'Credit Card (Automatic)', 'Electronic Check', 'Mailed Check'],
        'MultipleLines': ['No', 'No Phone Service', 'Yes']
    }


def get_default_values():
    """Get default input values"""
    return {
        'gender': 'Male',
        'SeniorCitizen': 'No',
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 24,
        'PhoneService': 'Yes',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'PaperlessBilling': 'No',
        'MonthlyCharges': 65.0,
        'TotalCharges': 1560.0,
        'Contract': 'One Year',
        'InternetService': 'Fiber Optic',
        'PaymentMethod': 'Bank Transfer (Automatic)',
        'MultipleLines': 'No'
    }
