#!/bin/bash
# Script to run the Churn Prediction Streamlit App
# This script is for Linux/Mac

echo ""
echo "============================================"
echo "  Customer Churn Prediction App"
echo "============================================"
echo ""

# Check if streamlit is installed
python -m pip show streamlit > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "Streamlit not found. Installing required packages..."
    python -m pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Error installing packages. Please run:"
        echo "python -m pip install -r requirements.txt"
        exit 1
    fi
fi

echo "Starting Churn Prediction App..."
echo ""
echo "The app will open in your browser at: http://localhost:8501"
echo "Press Ctrl+C to stop the app"
echo ""

streamlit run apps/prediction.py
