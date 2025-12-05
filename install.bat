@echo off
setlocal enabledelayedexpansion

set SCRIPT_NAME=%~nx0

echo ===============================
echo Checking Python installation...
echo ===============================

:: Check if python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found.
    goto install_python
) else (
    for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVERSION=%%v
    echo Found Python version: %PYVERSION%
)

:: Extract major.minor
for /f "tokens=1,2 delims=." %%a in ("%PYVERSION%") do (
    set MAJOR=%%a
    set MINOR=%%b
)

:: If version < 3.10 → install Python 3.10.9
if %MAJOR% lss 3 (
    goto install_python
)
if %MAJOR%==3 if %MINOR% lss 10 (
    goto install_python
)

echo Python version is OK.
goto setup_env

:install_python
echo ===============================
echo Installing Python 3.10.9...
echo ===============================

set PYTHON_INSTALLER=python-3.10.9-amd64.exe
set PYTHON_URL=https://www.python.org/ftp/python/3.10.9/python-3.10.9-amd64.exe

echo Downloading Python 3.10.9 from official site...
curl -o %PYTHON_INSTALLER% -L %PYTHON_URL%

if not exist %PYTHON_INSTALLER% (
    echo Failed to download Python installer.
    exit /b 1
)

echo Running installer silently...
start /wait %PYTHON_INSTALLER% /quiet InstallAllUsers=1 PrependPath=1 Include_test=0
del %PYTHON_INSTALLER%

echo Python installed successfully.
echo Restarting script to reload PATH...

:: Restart script after installation
start cmd /c "%SCRIPT_NAME%"
exit /b

:setup_env
echo ===============================
echo Setting up virtual environment...
echo ===============================

python -m venv venv
if errorlevel 1 (
    echo Failed to create virtual environment.
    exit /b 1
)

call venv\Scripts\activate

echo ===============================
echo Installing requirements...
echo ===============================

if exist requirements.txt (
    pip install -r requirements.txt
) else (
    echo requirements.txt not found. Skipping package installation.
)

echo ===============================
echo Setup complete!
echo ===============================

endlocal
pause
