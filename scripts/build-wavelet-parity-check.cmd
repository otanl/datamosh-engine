@echo off
setlocal
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0build-wavelet-parity-check.ps1" %*
exit /b %ERRORLEVEL%
