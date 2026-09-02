@echo off
REM Double-click: fetches the up to date code and data (git pull + dvc pull)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\sync.ps1"
echo.
pause
