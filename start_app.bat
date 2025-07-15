@echo off
echo Starting Heart Disease Prediction App...
echo ========================================

cd ui
python -m streamlit run app.py --server.port 8501

pause 