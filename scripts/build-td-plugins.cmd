@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build-td-plugins.ps1" %*
exit /b %ERRORLEVEL%
