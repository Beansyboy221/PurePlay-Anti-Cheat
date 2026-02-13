@echo off
setlocal EnableDelayedExpansion

cd /d "%~dp0"

set "ENV_YML=%~dp0environment.yml"
set "CONDA_BAT_PATH=%USERPROFILE%\miniforge3\condabin\conda.bat"

rem -------------------------------------------------------
rem Extract environment name from environment.yml
rem -------------------------------------------------------
if not exist "%ENV_YML%" (
    echo ERROR: environment.yml not found.
    pause
    exit /b
)

for /f "tokens=2 delims=: " %%A in ('findstr /B "name:" "%ENV_YML%"') do (
    set "ENV_NAME=%%A"
)

set "ENV_NAME=%ENV_NAME:"=%"
set "ENV_NAME=%ENV_NAME: =%"

if "%ENV_NAME%"=="" (
    echo ERROR: Could not detect environment name from environment.yml.
    pause
    exit /b
)

rem -------------------------------------------------------
rem Activate and Run
rem -------------------------------------------------------
echo Activating environment: %ENV_NAME%...
call "%CONDA_BAT_PATH%" activate %ENV_NAME%
if errorlevel 1 (
    echo.
    echo ERROR: Environment "%ENV_NAME%" not found. 
    echo Please run install.bat first.
    pause
    exit /b
)

echo Starting PurePlay...
python source\main.py

if errorlevel 1 (
    echo.
    echo PurePlay closed with an error.
    pause
)

endlocal