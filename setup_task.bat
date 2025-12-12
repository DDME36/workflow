@echo off
chcp 65001 >nul
echo ============================================
echo Setup Daily Task - Fear Greed Alert
echo ============================================
echo.

set SCRIPT_DIR=%~dp0
set PY_FILE=%SCRIPT_DIR%daily_check_v2.py

echo Creating scheduled task to run at 22:00 daily...
echo.

schtasks /create /tn "FearGreedDailyCheck" /tr "pythonw \"%SCRIPT_DIR%run_daily.pyw\"" /sc daily /st 22:00 /f

echo.
echo Done! Task "FearGreedDailyCheck" created.
echo Schedule: Every day at 22:00
echo.
echo Commands:
echo   View:   schtasks /query /tn "FearGreedDailyCheck"
echo   Run:    schtasks /run /tn "FearGreedDailyCheck"
echo   Delete: schtasks /delete /tn "FearGreedDailyCheck" /f
echo.
pause
