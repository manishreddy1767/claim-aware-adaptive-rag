@echo off
rem Removes the application environment and the desktop shortcut. Your accounts,
rem documents and question history are kept unless you choose to delete them.
rem
rem   uninstall.bat                 asks whether to delete your data
rem   uninstall.bat --keep-data     keeps it without asking
rem   uninstall.bat --delete-data   deletes it without asking
setlocal
cd /d "%~dp0"
set "DATA=%LOCALAPPDATA%\ClaimAwareRAG"
if defined CARAG_DATA_DIR set "DATA=%CARAG_DATA_DIR%"
set "DELETE_DATA="
if /i "%~1"=="--delete-data" set "DELETE_DATA=Y"
if /i "%~1"=="--keep-data" set "DELETE_DATA=N"

rem Only this installation's programs count (another copy elsewhere may be running).
powershell -NoProfile -Command "$v=[IO.Path]::GetFullPath('%~dp0.venv'); if (Get-CimInstance Win32_Process | Where-Object { $_.ExecutablePath -and $_.ExecutablePath.StartsWith($v, [StringComparison]::OrdinalIgnoreCase) }) { exit 1 }"
if errorlevel 1 (
    echo Claim-Aware RAG is still running. Close its window first, then run uninstall.bat again.
    exit /b 1
)

powershell -NoProfile -Command "$l=Join-Path ([Environment]::GetFolderPath('Desktop')) 'Claim-Aware RAG.lnk'; if (Test-Path $l) { Remove-Item $l }"
if exist ".venv" rmdir /s /q ".venv"
echo Removed the application environment and the desktop shortcut.

if not defined DELETE_DATA if exist "%DATA%" (
    choice /c YN /n /m "Also delete your accounts, documents and questions in %DATA%? [Y/N] "
    if errorlevel 2 (set "DELETE_DATA=N") else (set "DELETE_DATA=Y")
)
if /i "%DELETE_DATA%"=="Y" if exist "%DATA%" (
    rmdir /s /q "%DATA%"
    echo Deleted %DATA%.
)
if /i not "%DELETE_DATA%"=="Y" if exist "%DATA%" echo Your data is still in %DATA%.
echo You can now delete this folder.
exit /b 0
