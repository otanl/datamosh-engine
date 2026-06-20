@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build-td-dct-top.ps1" %*
exit /b %ERRORLEVEL%
