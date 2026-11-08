@echo off
echo ========================================================
echo  Launching Both CODE-A-NOVA Machine Learning Apps
echo  1. Automated Essay Scoring:      http://localhost:8501
echo  2. Movie Recommendation System:  http://localhost:8502
echo ========================================================

start "Automated Essay Scoring" cmd /k "cd /d "%~dp0CODE-A-NOVA\Automated-Essay-Scoring" && python -m streamlit run app.py --server.port 8501"
start "Movie Recommender" cmd /k "cd /d "%~dp0CODE-A-NOVA\Movie-Recommendation-System" && python -m streamlit run app.py --server.port 8502"

echo Both applications have been launched in separate windows!

