@echo off
setlocal EnableDelayedExpansion

set "ENV_YML=%~dp0environment.yml"
set "CONDA_PATH=%USERPROFILE%\miniforge3\scripts\conda.exe"

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
rem Delete environment
rem -------------------------------------------------------
echo Deleting conda environment...
call "%CONDA_PATH%" env remove -y -q -n %ENV_NAME%
if errorlevel 1 (
    echo ERROR: Failed to remove environment: "%ENV_NAME%".
    pause
    exit /b
)
echo Environment removed successfully.
echo Please delete the Miniforge installation manually if desired.
pause

endlocal