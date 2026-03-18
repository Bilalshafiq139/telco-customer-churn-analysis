"""
Streamlit Customer Churn Prediction App
Allows users to input customer data and predict churn probability
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from apps.utils import (
    ChurnPredictor, 
    get_feature_input_specs, 
    get_categorical_options,
    get_default_values
)


# ====================[PAGE CONFIG]====================
st.set_page_config(
    page_title="Churn Prediction App",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ====================[SIDEBAR CONFIG]====================
with st.sidebar:
    st.title("Model Configuration")
    
    model_choice = st.radio(
        "Select Prediction Model:",
        ["Random Forest (Recommended)", "Logistic Regression"],
        help="Choose between Random Forest (higher accuracy) or Logistic Regression (faster)"
    )
    
    st.info(
        "This app predicts customer churn probability based on customer data. "
        "The model uses historical customer information to identify at-risk customers."
    )


# ====================[LOAD MODELS]====================
@st.cache_resource
def load_models():
    """Load both trained models"""
    base_path = os.path.dirname(os.path.abspath(__file__))
    models_path = os.path.join(base_path, '..', 'models')
    
    lr_predictor = ChurnPredictor(
        model_path=os.path.join(models_path, 'logistic_regression_model.pkl'),
        scaler_path=os.path.join(models_path, 'scaler.pkl'),
        feature_names_path=os.path.join(models_path, 'feature_names.pkl')
    )
    
    rf_predictor = ChurnPredictor(
        model_path=os.path.join(models_path, 'random_forest_model.pkl'),
        scaler_path=os.path.join(models_path, 'scaler.pkl'),
        feature_names_path=os.path.join(models_path, 'feature_names.pkl')
    )
    
    return lr_predictor, rf_predictor


try:
    lr_predictor, rf_predictor = load_models()
    predictor = rf_predictor if "Random Forest" in model_choice else lr_predictor
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


# ====================[MAIN PAGE]====================
st.title("Customer Churn Prediction System")

st.markdown("Enter customer information below to predict the likelihood of churn (customer leaving the service).")


# Create form for customer input
with st.form("customer_form", clear_on_submit=False):
    
    st.subheader("Customer Information")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    # ==================[DEMOGRAPHIC SECTION]==================
    with col1:
        st.markdown("### Demographics")
        gender = st.selectbox(
            "Gender",
            ["Male", "Female"],
            index=0
        )
        
        senior_citizen = st.selectbox(
            "Senior Citizen",
            ["No", "Yes"],
            index=0
        )
        
        partner = st.selectbox(
            "Has Partner",
            ["No", "Yes"],
            index=0
        )
        
        dependents = st.selectbox(
            "Has Dependents",
            ["No", "Yes"],
            index=0
        )
    
    with col2:
        st.markdown("### Service Duration")
        tenure = st.number_input(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            value=24,
            help="How long the customer has been with the company"
        )
    
    # ==================[BILLING SECTION]==================
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("### Billing Information")
        monthly_charges = st.number_input(
            "Monthly Charges ($)",
            min_value=18.0,
            max_value=120.0,
            value=65.0,
            step=0.50
        )
    
    with col4:
        total_charges = st.number_input(
            "Total Charges ($)",
            min_value=0.0,
            max_value=8600.0,
            value=1560.0,
            step=10.0
        )
    
    # ==================[CONTRACT & SERVICE SECTION]==================
    col5, col6, col7 = st.columns(3)
    
    with col5:
        st.markdown("### Contract")
        contract = st.selectbox(
            "Contract Type",
            ["Month-to-Month", "One Year", "Two Years"],
            index=1,
            help="Type of contract the customer has"
        )
    
    with col6:
        st.markdown("### Internet Service")
        internet_service = st.selectbox(
            "Internet Service Type",
            ["DSL", "Fiber Optic", "No Internet Service"],
            index=1,
            help="Type of internet service"
        )
    
    with col7:
        st.markdown("### Payment Method")
        payment_method = st.selectbox(
            "Payment Method",
            ["Bank Transfer (Automatic)", "Credit Card (Automatic)", "Electronic Check", "Mailed Check"],
            index=0,
            help="How the customer pays"
        )
    
    # ==================[SERVICES SECTION]==================
    st.markdown("### Additional Services")
    
    col8, col9, col10, col11 = st.columns(4)
    
    with col8:
        phone_service = st.selectbox("Phone Service", ["No", "Yes"], index=1)
        online_security = st.selectbox("Online Security", ["No", "Yes"], index=0)
    
    with col9:
        online_backup = st.selectbox("Online Backup", ["No", "Yes"], index=0)
        device_protection = st.selectbox("Device Protection", ["No", "Yes"], index=0)
    
    with col10:
        tech_support = st.selectbox("Tech Support", ["No", "Yes"], index=0)
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes"], index=0)
    
    with col11:
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes"], index=0)
        paperless_billing = st.selectbox("Paperless Billing", ["No", "Yes"], index=0)
    
    # ==================[OTHER SERVICES]====================
    col12, col13 = st.columns(2)
    
    with col12:
        multiple_lines = st.selectbox(
            "Multiple Lines",
            ["No", "No Phone Service", "Yes"],
            index=0
        )
    
    # ==================[SUBMIT BUTTON]====================
    col_submit, col_reset = st.columns(2)
    
    with col_submit:
        submit_button = st.form_submit_button(
            "Predict Churn",
            use_container_width=True,
            type="primary"
        )
    
    with col_reset:
        reset_button = st.form_submit_button(
            "Reset Form",
            use_container_width=True
        )


# ====================[PROCESS PREDICTION]====================
if submit_button:
    
    # Create input dictionary matching the model's expected format
    input_data = {
        'gender': 1 if gender == 'Male' else 0,
        'SeniorCitizen': 1 if senior_citizen == 'Yes' else 0,
        'Partner': 1 if partner == 'Yes' else 0,
        'Dependents': 1 if dependents == 'Yes' else 0,
        'tenure': tenure,
        'PhoneService': 1 if phone_service == 'Yes' else 0,
        'OnlineSecurity': 1 if online_security == 'Yes' else 0,
        'OnlineBackup': 1 if online_backup == 'Yes' else 0,
        'DeviceProtection': 1 if device_protection == 'Yes' else 0,
        'TechSupport': 1 if tech_support == 'Yes' else 0,
        'StreamingTV': 1 if streaming_tv == 'Yes' else 0,
        'StreamingMovies': 1 if streaming_movies == 'Yes' else 0,
        'PaperlessBilling': 1 if paperless_billing == 'Yes' else 0,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges,
        'Contract_One year': 1 if contract == 'One Year' else 0,
        'Contract_Two year': 1 if contract == 'Two Years' else 0,
        'InternetService_Fiber optic': 1 if internet_service == 'Fiber Optic' else 0,
        'InternetService_No': 1 if internet_service == 'No Internet Service' else 0,
        'PaymentMethod_Credit card (automatic)': 1 if payment_method == 'Credit Card (Automatic)' else 0,
        'PaymentMethod_Electronic check': 1 if payment_method == 'Electronic Check' else 0,
        'PaymentMethod_Mailed check': 1 if payment_method == 'Mailed Check' else 0,
        'MultipleLines_No phone service': 1 if multiple_lines == 'No Phone Service' else 0,
        'MultipleLines_Yes': 1 if multiple_lines == 'Yes' else 0,
    }
    
    try:
        # Get prediction
        result = predictor.predict(input_data)
        
        # Display results
        st.subheader("Prediction Result")
        
        # Create result columns
        pred_col1, pred_col2, pred_col3 = st.columns(3)
        
        with pred_col1:
            churn_probability = result['probability_churn'] * 100
            if churn_probability > 60:
                st.error(f"Churn Probability: {churn_probability:.1f}%")
            elif churn_probability > 40:
                st.warning(f"Churn Probability: {churn_probability:.1f}%")
            else:
                st.success(f"Churn Probability: {churn_probability:.1f}%")
        
        with pred_col2:
            no_churn_probability = result['probability_no_churn'] * 100
            st.info(f"Retention Probability: {no_churn_probability:.1f}%")
        
        with pred_col3:
            if result['prediction'] == 1:
                st.error(f"Prediction: Likely to Churn")
            else:
                st.success(f"Prediction: Likely to Stay")
        
        # Additional insights
        st.subheader("Detailed Analysis")
        
        # Create a summary table
        summary_data = {
            'Metric': ['Churn Probability', 'Retention Probability', 'Prediction'],
            'Value': [
                f"{churn_probability:.2f}%",
                f"{no_churn_probability:.2f}%",
                "Churn" if result['prediction'] == 1 else "No Churn"
            ]
        }
        
        st.dataframe(
            pd.DataFrame(summary_data),
            use_container_width=True,
            hide_index=True
        )
        
        # Save prediction to CSV
        st.subheader("Save Prediction")
        
        # Create prediction record
        prediction_record = {
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Model': 'Random Forest' if "Random Forest" in model_choice else 'Logistic Regression',
            'Gender': gender,
            'Senior_Citizen': senior_citizen,
            'Partner': partner,
            'Dependents': dependents,
            'Tenure_Months': tenure,
            'Monthly_Charges': monthly_charges,
            'Total_Charges': total_charges,
            'Contract': contract,
            'Internet_Service': internet_service,
            'Payment_Method': payment_method,
            'Phone_Service': phone_service,
            'Online_Security': online_security,
            'Online_Backup': online_backup,
            'Device_Protection': device_protection,
            'Tech_Support': tech_support,
            'Streaming_TV': streaming_tv,
            'Streaming_Movies': streaming_movies,
            'Paperless_Billing': paperless_billing,
            'Multiple_Lines': multiple_lines,
            'Churn_Probability': round(churn_probability, 2),
            'Retention_Probability': round(no_churn_probability, 2),
            'Prediction': 'Churn' if result['prediction'] == 1 else 'No Churn'
        }
        
        # Convert to DataFrame
        pred_df = pd.DataFrame([prediction_record])
        
        # Option to download as CSV
        csv = pred_df.to_csv(index=False)
        st.download_button(
            label="Download Prediction as CSV",
            data=csv,
            file_name=f"churn_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
        
        # Save to persistent log file
        log_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'predictions_log.csv')
        
        if os.path.exists(log_path):
            existing_df = pd.read_csv(log_path)
            updated_df = pd.concat([existing_df, pred_df], ignore_index=True)
        else:
            updated_df = pred_df
        
        updated_df.to_csv(log_path, index=False)
        st.success("Prediction saved to log file!")
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.warning("Please check your inputs and try again.")


# ====================[FOOTER]====================
st.markdown("""
About this App:
- Uses machine learning models trained on historical customer data
- Helps identify at-risk customers for targeted retention strategies
- Provides actionable insights for business decision-making

Model Performance:
- Random Forest Accuracy: 77.26%
- Logistic Regression Accuracy: 72.57%

For questions or feedback, please contact the analytics team.
""")
