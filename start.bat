@echo off
title TradingAgents
echo ==========================================
echo  TradingAgents - Starting Application
echo ==========================================
echo.

:: Kill any stale processes on ports 8000 and 3000
echo Cleaning up stale processes...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do taskkill /PID %%a /F >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :3000 ^| findstr LISTENING') do taskkill /PID %%a /F >nul 2>&1
timeout /t 2 /nobreak >nul

:: Start backend
echo Starting backend (port 8000)...
cd /d "%~dp0web\backend"
start "TradingAgents Backend" cmd /k "title TradingAgents Backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000"

:: MT5 worker commands: --mt5-restart, --mt5-stop
if "%1"=="--mt5-restart" (
    echo Restarting MT5 worker...
    cd /d "%~dp0web\backend"
    python mt5_worker.py stop
    timeout /t 3 /nobreak >nul
    start "MT5 Worker" cmd /k "title MT5 Worker && python mt5_worker.py start"
    goto :end
)
if "%1"=="--mt5-stop" (
    echo Stopping MT5 worker...
    cd /d "%~dp0web\backend"
    python mt5_worker.py stop
    goto :end
)
if "%1"=="--mt5" (
    echo Starting MT5 worker...
    cd /d "%~dp0web\backend"
    start "MT5 Worker" cmd /k "title MT5 Worker && python mt5_worker.py start"
)

:: Wait for backend to be ready
echo Waiting for backend...
timeout /t 5 /nobreak >nul

:: Start frontend
echo Starting frontend (port 3000)...
cd /d "%~dp0web\frontend"
if not exist "node_modules" (
    echo Installing frontend dependencies...
    npm install
)
start "TradingAgents Frontend" cmd /k "title TradingAgents Frontend && npm run dev"

:: Wait for frontend to be ready
timeout /t 5 /nobreak >nul

echo.
echo ==========================================
echo  Application started!
echo  Frontend: http://localhost:3000
echo  Backend:  http://localhost:8000
echo ==========================================
echo.
echo Close the Backend and Frontend windows to stop.
:end
pause
