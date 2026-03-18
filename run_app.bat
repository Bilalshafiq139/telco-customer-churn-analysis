@echo off
REM Batch script to run the Churn Prediction Streamlit App
REM This script is for Windows

echo.
echo ============================================
echo  Customer Churn Prediction App
echo ============================================
echo.

REM Check if streamlit is installed
python -m pip show streamlit >nul 2>&1
if %errorlevel% neq 0 (
    echo Streamlit not found. Installing required packages...
    python -m pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo Error installing packages. Please run:
        echo python -m pip install -r requirements.txt
        exit /b 1
    )
)

echo Starting Churn Prediction App...
echo.
echo The app will open in your browser at: http://localhost:8501
echo Press Ctrl+C to stop the app
echo.

python -m streamlit run apps/prediction.py

pause
