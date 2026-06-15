@echo off
setlocal
cd /d "%~dp0"

echo.
echo Starting MLB Games Dashboard (offline ESPN sample for 2026-06-16)...
echo.

python mlb_sbr.py dashboard --fixture tests/fixtures/espn_scoreboard_20260616.json --no-odds
if errorlevel 1 (
  echo.
  echo Failed to start the dashboard.
  echo.
  pause
)
