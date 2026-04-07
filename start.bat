@echo off
title TradingAgents

set "ROOT=%~dp0"
set "ACTIVATE=%ROOT%.venv\Scripts\activate.bat"
set "PYTHON=%ROOT%.venv\Scripts\python.exe"

echo ==========================================
echo  TradingAgents - Starting Application
echo ==========================================
echo.

:: Kill any stale processes on ports 8000 and 3000
echo Cleaning up stale processes...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :8000 ^| findstr LISTENING') do taskkill /PID %%a /F >nul 2>&1
for /f "tokens=5" %%a in ('netstat -ano ^| findstr :3000 ^| findstr LISTENING') do taskkill /PID %%a /F >nul 2>&1
timeout /t 2 /nobreak >nul

:: Install frontend deps if needed
cd /d "%ROOT%web\frontend"
if not exist "node_modules" (
    echo Installing frontend dependencies...
    call npm install
)

:: Start backend (minimized)
echo Starting backend (port 8000)...
start /min "TradingAgents Backend" cmd /c "title TradingAgents Backend && cd /d %ROOT%web\backend && call %ACTIVATE% && python -m uvicorn main:app --host 0.0.0.0 --port 8000"

:: Wait for backend
timeout /t 5 /nobreak >nul

:: Start MT5 worker (minimized)
echo Starting MT5 worker...
start /min "MT5 Worker" cmd /c "title MT5 Worker && cd /d %ROOT%web\backend && call %ACTIVATE% && python mt5_worker.py start"

:: Start TMA worker (minimized)
echo Starting TMA worker...
start /min "TMA Worker" cmd /c "title TMA Worker && cd /d %ROOT%web\backend && call %ACTIVATE% && python tma_worker.py start"

:: Start frontend (minimized)
echo Starting frontend (port 3000)...
start /min "TradingAgents Frontend" cmd /c "title TradingAgents Frontend && cd /d %ROOT%web\frontend && npx next dev -p 3000"

timeout /t 3 /nobreak >nul

echo.
echo ==========================================
echo  Application started!
echo  Frontend: http://localhost:3000
echo  Backend:  http://localhost:8000
echo ==========================================
echo.
echo Close this window to leave services running,
echo or press any key then close all titled windows to stop.
pause
