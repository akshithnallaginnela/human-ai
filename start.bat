@echo off
title Human Activity Detection - Launcher
color 0A

echo ============================================
echo   Human Activity Detection System
echo ============================================
echo.

:: -- Check venv exists --
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Virtual environment not found!
    echo Please run: python -m venv .venv --clear --upgrade-deps
    echo Then:       .venv\Scripts\pip.exe install -r backend\requirements.txt
    pause
    exit /b 1
)

echo [1/3] Starting Backend (Flask on port 5000)...
start "HAD - Backend" cmd /k ".venv\Scripts\python.exe backend\app.py"

echo [2/3] Waiting for backend to initialise...
timeout /t 4 /nobreak >nul

echo [3/3] Starting Frontend (HTTP server on port 8080)...
start "HAD - Frontend" cmd /k "cd /d frontend && python -m http.server 8080"

echo.
echo [4/4] Opening browser...
timeout /t 2 /nobreak >nul
start http://localhost:8080

echo.
echo ============================================
echo   Both servers are running!
echo   Backend  : http://localhost:5000
echo   Frontend : http://localhost:8080
echo ============================================
echo.
echo Close the Backend and Frontend windows to stop the servers.
pause
