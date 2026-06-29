@echo off
REM Double-clic : recupere code + donnees a jour (git pull + dvc pull)
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\sync.ps1"
echo.
pause
