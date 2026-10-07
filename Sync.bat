@echo off
REM Double-click: get the latest code + data (git pull + dvc pull)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\sync.ps1"
echo.
pause
