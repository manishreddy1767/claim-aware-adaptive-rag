@echo off
rem Claim-Aware RAG installer for Windows. Everything runs on this laptop: its GPU
rem (or CPU) does the work and documents are saved on its disk.
rem
rem   install.bat                  install, prepare the models, create a desktop shortcut
rem   install.bat --no-shortcut    without the desktop shortcut
rem   install.bat --no-pause       do not wait for a key press at the end
setlocal
cd /d "%~dp0"
set "SHORTCUT=1"
set "PAUSE_AT_END=1"
:args
if "%~1"=="" goto :start
if /i "%~1"=="--no-shortcut" set "SHORTCUT=0"
if /i "%~1"=="--no-pause" set "PAUSE_AT_END=0"
shift
goto :args

:start
echo.
echo   Claim-Aware RAG - installation
echo   ==============================
echo.

rem ---- 1. Python 3.10 or newer ---------------------------------------------------------
call :find_python
if not defined PY (
    echo Python 3.10 or newer was not found on this computer.
    where winget >nul 2>nul || goto :no_python
    echo Installing Python 3.12 for this user with winget. This can take a few minutes...
    winget install -e --id Python.Python.3.12 --scope user --accept-source-agreements --accept-package-agreements
    call :find_python
)
if not defined PY goto :no_python
echo [1/5] Python found.

rem ---- 2. Private environment for the application ------------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo [2/5] Creating the application environment...
    %PY% -m venv .venv || goto :error
) else (
    echo [2/5] Application environment already exists.
)
set "VPY=.venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip --quiet || goto :error

rem ---- 3. PyTorch: the NVIDIA GPU build when a GPU is present -------------------------------
"%VPY%" -c "import torch" >nul 2>nul
if errorlevel 1 (
    nvidia-smi >nul 2>nul
    if errorlevel 1 (
        echo [3/5] No NVIDIA GPU found: installing PyTorch for the CPU. Answers will be slower.
        "%VPY%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu || goto :error
    ) else (
        echo [3/5] NVIDIA GPU found: installing PyTorch with GPU support. This is a large download.
        "%VPY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu126 || goto :error
    )
) else (
    echo [3/5] PyTorch already installed.
)

rem ---- 4. The application ---------------------------------------------------------------------
echo [4/5] Installing Claim-Aware RAG...
"%VPY%" -m pip install -e ".[ocr]" --quiet || goto :error

rem ---- 5. Models, so the first start is quick and later starts work offline ------------------
echo [5/5] Preparing the models...
"%VPY%" -m carag.server.prefetch
if errorlevel 1 echo Warning: the models could not be prepared now; they will download on the first start.

if "%SHORTCUT%"=="1" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command ^
      "$d=[Environment]::GetFolderPath('Desktop'); $s=(New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $d 'Claim-Aware RAG.lnk')); $s.TargetPath='%~dp0start.bat'; $s.WorkingDirectory='%~dp0'; $s.Description='Ask questions about your documents'; $s.IconLocation='%SystemRoot%\System32\imageres.dll,76'; $s.Save()" ^
      && echo Created the desktop shortcut "Claim-Aware RAG".
)

echo.
echo   Installed. Start it from the desktop shortcut or with start.bat.
echo   Your documents and accounts are saved in %LOCALAPPDATA%\ClaimAwareRAG
echo.
if "%PAUSE_AT_END%"=="1" pause
exit /b 0

:find_python
set "PY="
set "CHECK=import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)"
py -3 -c "%CHECK%" >nul 2>nul && (set "PY=py -3" & exit /b 0)
python -c "%CHECK%" >nul 2>nul && (set "PY=python" & exit /b 0)
set "WINGET_PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if exist "%WINGET_PY%" "%WINGET_PY%" -c "%CHECK%" >nul 2>nul && (set PY="%WINGET_PY%" & exit /b 0)
exit /b 0

:no_python
echo.
echo Please install Python 3.10 or newer from https://www.python.org/downloads/
echo (tick "Add python.exe to PATH" during setup), then run install.bat again.
goto :error_end

:error
echo.
echo Installation failed. See the messages above.
:error_end
if "%PAUSE_AT_END%"=="1" pause
exit /b 1
