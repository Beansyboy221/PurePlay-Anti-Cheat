@echo off
setlocal

set "CONDA_BAT_PATH=%USERPROFILE%\miniforge3\condabin\conda.bat"
call "%CONDA_BAT_PATH%" activate PurePlay-Anti-Cheat
if errorlevel 1 (
    echo No environment found. Please run install.bat first.
    pause
    exit /b
)
call python source\main.py
pause

endlocal