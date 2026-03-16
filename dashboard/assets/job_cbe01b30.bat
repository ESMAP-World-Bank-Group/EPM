@echo off
echo ============================================================
echo   EPM Run  : Run 2026-03-15 15:51
echo   Folder   : data_eapp
echo   Started  : %DATE% %TIME%
echo ============================================================
echo.
call conda activate gams_env
cd /d "C:\Users\wb590892\Documents\EPM_Models\EPM\epm"
python -u epm.py --folder_input data_eapp --modeltype RMIP --scenarios scenarios.csv
set _RC=%ERRORLEVEL%
echo.
echo ============================================================
echo   Finished with exit code: %_RC%
echo ============================================================
(echo %_RC%) > "C:\Users\wb590892\Documents\EPM_Models\EPM\dashboard\assets\job_cbe01b30.done"
exit /b %_RC%