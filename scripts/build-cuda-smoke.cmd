@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build-cuda-smoke.ps1" %*
exit /b %ERRORLEVEL%
