@echo off
echo ============================================================
echo   EPM Run  : Run 2026-03-16 12:44
echo   Folder   : data_test
echo   Started  : %DATE% %TIME%
echo ============================================================
echo.
call conda activate gams_env
cd /d "C:\Users\wb590892\Documents\EPM_Models\EPM\epm"
python -u epm.py --folder_input data_test --scenarios scenarios.csv --selected_scenarios baseline
set _RC=%ERRORLEVEL%
echo.
echo ============================================================
echo   Finished with exit code: %_RC%
echo ============================================================
(echo %_RC%) > "C:\Users\wb590892\Documents\EPM_Models\EPM\dashboard\assets\job_05dc5550.done"
exit /b %_RC%