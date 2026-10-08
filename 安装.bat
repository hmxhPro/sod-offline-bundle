@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"
echo.
echo  ============================================================
echo   SOD Windows Install - first run needs Internet access
echo  ============================================================
echo.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0windows\install.ps1"
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" echo  [X] Install FAILED with exit code %RC%. Check the red messages above.
echo  ------------------------------------------------------------
echo   If a red [X] error appears above, please take a screenshot
echo   and report it together with the folder windows\logs
echo  ------------------------------------------------------------
pause
