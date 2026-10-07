@echo off
echo ============================================================
echo   EPM Run  : Run 2026-03-15 14:55
echo   Folder   : data_eapp
echo   Started  : %DATE% %TIME%
echo ============================================================
echo.
conda activate gams_env
cd /d "C:\Users\wb590892\Documents\EPM_Models\EPM\epm"
python -u epm.py --folder_input data_eapp --modeltype RMIP --scenarios scenarios.csv --selected_scenarios TEST
set _RC=%ERRORLEVEL%
echo.
echo ============================================================
echo   Finished with exit code: %_RC%
echo ============================================================
echo %_RC%> "C:\Users\wb590892\Documents\EPM_Models\EPM\dashboard\assets\job_887e3700.done"
exit /b %_RC%