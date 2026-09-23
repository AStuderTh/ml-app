@echo off
cd /d "%~dp0"

rem Ferme d'éventuelles anciennes instances restées ouvertes sur le port 8501
rem (sinon plusieurs serveurs peuvent tourner en même temps et servir une
rem version périmée de l'app sans erreur visible).
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8501" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%p >nul 2>&1
)

set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"

"%PYTHON%" -m streamlit run app\streamlit_app.py
pause
