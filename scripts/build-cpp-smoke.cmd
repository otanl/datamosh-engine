@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build-cpp-smoke.ps1" %*
exit /b %ERRORLEVEL%
