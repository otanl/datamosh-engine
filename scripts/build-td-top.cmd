@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build-td-top.ps1" %*
exit /b %ERRORLEVEL%
