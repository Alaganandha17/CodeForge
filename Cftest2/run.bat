@echo off
REM ============================================================
REM CodeForge - One-Click Startup (Windows)
REM Usage: Double-click this file or run: run.bat
REM ============================================================

title CodeForge
cd /d "%~dp0"

echo.
echo  ================================================
echo     CodeForge - Python Code Optimizer
echo  ================================================
echo.

REM ---- Check Python ----
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Download from https://www.python.org/downloads/
    pause
    exit /b 1
)
echo [1/5] Python found.

REM ---- Install Dependencies ----
echo [2/5] Installing dependencies...
pip install -q -r codeforge-frontend\requirements.txt 2>nul
if errorlevel 1 (
    echo [WARNING] Some dependencies may have failed. Trying to continue...
)
echo       Dependencies ready.

REM ---- Setup .env if missing ----
if not exist backend_new\.env (
    if exist backend_new\.env.example (
        echo [INFO] Creating .env from template...
        copy backend_new\.env.example backend_new\.env >nul 2>nul
        echo       Edit backend_new\.env to add your GEMINI_API_KEY for AI mode.
    ) else (
        echo [INFO] Creating default .env...
        echo GEMINI_API_KEY=> backend_new\.env
        echo MODEL_NAME=gemini-2.5-flash>> backend_new\.env
        echo       Edit backend_new\.env to add your GEMINI_API_KEY for AI mode.
    )
)

REM ---- Kill any existing backend on port 8000 ----
echo [3/5] Checking for existing processes...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8000.*LISTENING" 2^>nul') do (
    taskkill /f /pid %%a >nul 2>&1
)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":8501.*LISTENING" 2^>nul') do (
    taskkill /f /pid %%a >nul 2>&1
)

REM ---- Start Backend ----
echo [4/5] Starting backend server...
start "CodeForge-Backend" cmd /c "cd /d "%~dp0backend_new" && python jeremy_final.py 2>&1 || (echo. && echo [ERROR] Backend crashed! Check the error above. && pause)"

REM Wait for backend to actually be ready (poll instead of blind wait)
echo       Waiting for backend...
set /a TRIES=0
:wait_backend
set /a TRIES+=1
if %TRIES% gtr 20 (
    echo [WARNING] Backend may not have started. Check the backend window for errors.
    goto start_frontend
)
timeout /t 1 /nobreak >nul
python -c "import requests; requests.get('http://127.0.0.1:8000/', timeout=2)" >nul 2>&1
if errorlevel 1 goto wait_backend
echo       Backend running on http://localhost:8000

REM ---- Start Frontend ----
:start_frontend
echo [5/5] Starting frontend...
echo.
echo  ================================================
echo    CodeForge is ready!
echo    Open:  http://localhost:8501
echo    Login: admin / admin123
echo  ================================================
echo.
echo  Press Ctrl+C or close this window to stop.
echo.

cd /d "%~dp0codeforge-frontend"
streamlit run app.py --server.port 8501 --server.headless true

REM ---- Cleanup when frontend stops ----
echo.
echo Stopping backend...
taskkill /f /fi "WINDOWTITLE eq CodeForge-Backend" >nul 2>&1
echo CodeForge stopped.
pause
