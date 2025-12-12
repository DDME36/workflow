@echo off
chcp 65001 >nul
echo ============================================
echo INSTALL Fear Greed Alert
echo ============================================
echo.
echo This will:
echo 1. Copy files to AppData\Local\FearGreedAlert
echo 2. Create startup shortcut (run when PC starts)
echo.
pause

set INSTALL_DIR=%USERPROFILE%\AppData\Local\FearGreedAlert
set SOURCE_DIR=%~dp0

echo.
echo Creating install directory...
mkdir "%INSTALL_DIR%" 2>nul
mkdir "%INSTALL_DIR%\models" 2>nul

echo Copying files...
xcopy "%SOURCE_DIR%*.py" "%INSTALL_DIR%\" /Y /Q
xcopy "%SOURCE_DIR%*.pyw" "%INSTALL_DIR%\" /Y /Q
xcopy "%SOURCE_DIR%*.txt" "%INSTALL_DIR%\" /Y /Q
xcopy "%SOURCE_DIR%.env" "%INSTALL_DIR%\" /Y /Q
xcopy "%SOURCE_DIR%models\*" "%INSTALL_DIR%\models\" /Y /Q

echo.
echo Creating startup shortcut...

set STARTUP_FOLDER=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup
set SHORTCUT=%STARTUP_FOLDER%\FearGreedAlert.lnk

echo Set oWS = WScript.CreateObject("WScript.Shell") > "%TEMP%\CreateShortcut.vbs"
echo sLinkFile = "%SHORTCUT%" >> "%TEMP%\CreateShortcut.vbs"
echo Set oLink = oWS.CreateShortcut(sLinkFile) >> "%TEMP%\CreateShortcut.vbs"
echo oLink.TargetPath = "pythonw.exe" >> "%TEMP%\CreateShortcut.vbs"
echo oLink.Arguments = """%INSTALL_DIR%\run_daily.pyw""" >> "%TEMP%\CreateShortcut.vbs"
echo oLink.WorkingDirectory = "%INSTALL_DIR%" >> "%TEMP%\CreateShortcut.vbs"
echo oLink.Description = "Fear Greed Daily Alert" >> "%TEMP%\CreateShortcut.vbs"
echo oLink.Save >> "%TEMP%\CreateShortcut.vbs"

cscript //nologo "%TEMP%\CreateShortcut.vbs"
del "%TEMP%\CreateShortcut.vbs"

echo.
echo ============================================
echo INSTALLATION COMPLETE!
echo ============================================
echo.
echo Installed to: %INSTALL_DIR%
echo.
echo The alert will run automatically when you start Windows.
echo It checks once per day and sends Discord notification.
echo.
echo You can safely delete this folder now.
echo The installed version is in AppData\Local\FearGreedAlert
echo.
pause
