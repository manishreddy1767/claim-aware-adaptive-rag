@echo off
rem Starts the Claim-Aware RAG application and opens it in the browser.
rem Extra options are passed through, e.g.  start.bat --port 9000  or  start.bat --no-signup
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\carag-app.exe" (
    echo The application is not installed yet. Run install.bat first.
    pause
    exit /b 1
)
".venv\Scripts\carag-app.exe" %*
