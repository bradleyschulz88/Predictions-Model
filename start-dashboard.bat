@echo off
setlocal
cd /d "%~dp0"

echo.
echo Starting MLB Games Dashboard...
echo.

for /f "tokens=5" %%a in ('netstat -ano ^| findstr /R /C:":8765 .*LISTENING"') do (
  echo Stopping old dashboard on port 8765...
  taskkill /PID %%a /F >nul 2>&1
)

python mlb_sbr.py dashboard --source espn --insecure
if errorlevel 1 (
  echo.
  echo Failed to start the dashboard.
  echo Make sure Python is installed and run: python --version
  echo.
  pause
)
