@echo off
rem One-time setup: creates .venv, installs PyTorch (GPU build when an NVIDIA GPU is present)
rem and the application. Afterwards start the app with start.bat.
setlocal
cd /d "%~dp0"

set "PY="
where py >nul 2>nul && set "PY=py -3"
if not defined PY (
    where python >nul 2>nul && set "PY=python"
)
if not defined PY (
    echo Python 3.10 or newer is required: https://www.python.org/downloads/
    goto :error
)

if not exist ".venv\Scripts\python.exe" (
    echo Creating the virtual environment...
    %PY% -m venv .venv || goto :error
)
set "VPY=.venv\Scripts\python.exe"
"%VPY%" -m pip install --upgrade pip || goto :error

"%VPY%" -c "import torch" >nul 2>nul
if errorlevel 1 (
    nvidia-smi >nul 2>nul
    if errorlevel 1 (
        echo Installing PyTorch ^(CPU^)...
        "%VPY%" -m pip install torch --index-url https://download.pytorch.org/whl/cpu || goto :error
    ) else (
        echo Installing PyTorch ^(NVIDIA GPU, CUDA 12.6^)...
        "%VPY%" -m pip install torch --index-url https://download.pytorch.org/whl/cu126 || goto :error
    )
)

echo Installing Claim-Aware RAG...
"%VPY%" -m pip install -e ".[ocr]" || goto :error

echo.
echo Installed. Start the application with start.bat
echo (the first start downloads the models, about 1 GB).
exit /b 0

:error
echo.
echo Installation failed. See the messages above.
exit /b 1
