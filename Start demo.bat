@echo off
rem Double-click to start the demo page. Close this window to stop it.
title Malicious URL detector - demo (close this window to stop)
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo The project's Python environment is missing.
  echo Set it up first: see "Quick start" in README.md.
  echo.
  pause
  exit /b 1
)

".venv\Scripts\python.exe" src\serve.py %*
if errorlevel 1 (
  echo.
  pause
)
