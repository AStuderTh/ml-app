@echo off
cd /d "%~dp0"

rem Close any old instances still listening on port 8501 to avoid serving a
rem stale version of the app from multiple Streamlit servers.
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8501" ^| findstr "LISTENING"') do (
    taskkill /F /PID %%p >nul 2>&1
)

set "PYTHON=python"
if exist ".venv\Scripts\python.exe" set "PYTHON=.venv\Scripts\python.exe"

"%PYTHON%" -m streamlit run app\streamlit_app.py
pause
