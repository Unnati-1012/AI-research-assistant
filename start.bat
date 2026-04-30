@echo off
cd /d "%~dp0"
echo Starting Noviq AI (Backend + Frontend)...
echo.
echo Backend will run on: http://localhost:8000
echo Frontend will run on: http://localhost:8501
echo.

REM Start backend in new window
start "Noviq AI Backend" cmd /k "cd backend && ..\venv\Scripts\activate && uvicorn main:app --reload"

REM Wait a moment for backend to start
timeout /t 3 /nobreak

REM Start frontend in new window
start "Noviq AI Frontend" cmd /k "venv\Scripts\activate && streamlit run frontend\streamlit_ui.py"

echo.
echo Both services started! Opening frontend in browser...
timeout /t 3 /nobreak
start http://localhost:8501
