@echo off
echo ========================================================
echo  Launching Automated Essay Scoring (AES)
echo  URL: http://localhost:8501
echo ========================================================
cd /d "%~dp0CODE-A-NOVA\Automated-Essay-Scoring"
python -m streamlit run app.py --server.port 8501
pause

