@echo off
chcp 65001 >nul
echo ============================================
echo Setup Fear Greed Alert - Windows Startup
echo ============================================
echo.

:: Get current directory
set SCRIPT_DIR=%~dp0
set PYW_FILE=%SCRIPT_DIR%run_daily.pyw

:: Create shortcut in Startup folder
set STARTUP_FOLDER=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup
set SHORTCUT=%STARTUP_FOLDER%\FearGreedAlert.lnk

:: Create VBS script to make shortcut
echo Set oWS = WScript.CreateObject("WScript.Shell") > CreateShortcut.vbs
echo sLinkFile = "%SHORTCUT%" >> CreateShortcut.vbs
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> CreateShortcut.vbs
echo oLink.TargetPath = "pythonw.exe" >> CreateShortcut.vbs
echo oLink.Arguments = """%PYW_FILE%""" >> CreateShortcut.vbs
echo oLink.WorkingDirectory = "%SCRIPT_DIR%" >> CreateShortcut.vbs
echo oLink.Description = "Fear Greed Daily Alert" >> CreateShortcut.vbs
echo oLink.Save >> CreateShortcut.vbs

cscript //nologo CreateShortcut.vbs
del CreateShortcut.vbs

echo.
echo Done! Shortcut created at:
echo %SHORTCUT%
echo.
echo The script will run automatically when you start Windows.
echo It will check Fear/Greed and send Discord alert, then close.
echo.
pause
