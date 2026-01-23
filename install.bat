@echo off
setlocal EnableDelayedExpansion

rem ------------------------
rem Define paths and URLs
rem ------------------------

set "CONDA_PATH=%USERPROFILE%\miniforge3\scripts\conda.exe"
set "CONDA_BAT_PATH=%USERPROFILE%\miniforge3\condabin\conda.bat"
set "ENVIRONMENT_DIR=%USERPROFILE%\miniforge3\envs\PurePlay-Anti-Cheat"

set "INSTALLER_URL=https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe"
set "INSTALLER_PATH=%~dp0\miniforge-installer.exe"

rem ------------------------
rem Check for Miniforge
rem ------------------------

if not exist "%CONDA_PATH%" (
    echo Miniforge not found. Checking for installer...
    if not exist "%INSTALLER_PATH%" (
        echo Installer not found. Downloading...
        powershell -Command "Invoke-WebRequest -Uri '%INSTALLER_URL%' -OutFile '%INSTALLER_PATH%'"
        if errorlevel 1 (
            echo Failed to download the installer.
            pause
            exit /b
        )
    )
    echo Running the miniforge installer.
    echo Do not change the default options.
    echo After installation, please restart this script.
    start "" "%INSTALLER_PATH%"
    pause
    exit /b
)

rem ------------------------
rem Delete Miniforge installer
rem ------------------------

if exist "%INSTALLER_PATH%" (
    echo Deleting Miniforge installer...
    del "%INSTALLER_PATH%"
    if errorlevel 1 (
        echo Failed to delete the installer.
        pause
        exit /b
    )
)

rem ------------------------
rem Create and set up the conda environment
rem ------------------------

echo Checking if the environment already exists...
call "%CONDA_PATH%" env list | findstr /C:"PurePlay-Anti-Cheat" >nul
if %errorlevel%==0 (
    set /p OVERWRITE_ENV="Environment already exists. Do you want to overwrite it? (y/n): "
    if /i "!OVERWRITE_ENV!"=="y" (
        echo Deleting existing environment...
        call "%CONDA_PATH%" env remove -y -n PurePlay-Anti-Cheat
        if errorlevel 1 (
            echo Failed to delete the existing environment.
            pause
            exit /b
        )
    ) else (
        echo Updating existing environment...
        goto :SKIP_ENV_CREATION
    )
)

echo Creating new environment...
call "%CONDA_PATH%" create -y -q -n PurePlay-Anti-Cheat python=3.11
if errorlevel 1 (
    echo Failed to create the environment.
    pause
    exit /b
)

:SKIP_ENV_CREATION

echo Activating environment...
call "%CONDA_BAT_PATH%" activate PurePlay-Anti-Cheat
if errorlevel 1 (
    echo Failed to activate the environment.
    pause
    exit /b
)

echo Installing conda packages...
call "%CONDA_PATH%" install -y pydantic pywin32 pyarrow polars pytorch-gpu lightning lightning[extra] optuna optuna-integration plotly scikit-learn psutil onnx
if errorlevel 1 (
    echo Failed to install conda packages.
    pause
    exit /b
)

echo Installing pip packages...
call pip install XInput-Python keyboard mouse
if errorlevel 1 (
    echo Failed to install pip packages.
    pause
    exit /b
)

echo Setup complete!
echo Run start.bat to begin!
pause
endlocal