@echo off
echo Starting SRAG...
echo.
echo API Server: http://localhost:8000
echo Frontend:   http://localhost:3000 (dev) or http://localhost:8000 (production)
echo.

start "SRAG API" cmd /c "cd /d %~dp0 && python api_server.py"
timeout /t 3 /nobreak >nul
start "SRAG Frontend" cmd /c "cd /d %~dp0\web && npm run dev"

echo.
echo Servers starting...
echo API:   http://localhost:8000
echo Frontend: http://localhost:3000
echo.
pause
