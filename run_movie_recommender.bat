@echo off
echo ========================================================
echo  Launching Movie Recommendation System (CineMatch)
echo  URL: http://localhost:8502
echo ========================================================
cd /d "%~dp0CODE-A-NOVA\Movie-Recommendation-System"
python -m streamlit run app.py --server.port 8502
pause

