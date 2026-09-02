@echo off
REM Double-click: publishes your data (code and pointers to GitHub, data to the store, EPM View up to date)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\publish.ps1"
echo.
pause
