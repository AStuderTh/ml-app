@echo off
cd /d "%~dp0"
"%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe" -m streamlit run app\streamlit_app.py
pause
