@echo off
setlocal

set "CONDA_PATH=%USERPROFILE%\miniforge3\scripts\conda.exe"
echo Deleting conda environment...
call "%CONDA_PATH%" env remove -y -q -n PurePlay-Anti-Cheat
if errorlevel 1 (
    echo Failed to remove the PurePlay-Anti-Cheat environment.
    pause
    exit /b
)
echo Environment removed successfully.
pause

endlocal