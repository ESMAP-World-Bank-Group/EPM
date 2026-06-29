@echo off
REM Double-clic : publie tes donnees (code+pointeurs sur GitHub, donnees sur le store, EPM View a jour)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\publish.ps1"
echo.
pause
