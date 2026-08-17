@echo off
cd /d "%~dp0"
title Plant Disease GUI

set PY=python
where python >nul 2>nul
if errorlevel 1 set PY=py -3

%PY% -c "import torch, tkinter" >nul 2>nul
if errorlevel 1 goto try_venv

echo [OK] Use system Python
%PY% gui\app.py
goto end

:try_venv
if not exist ".venv\Scripts\python.exe" goto setup

echo [OK] Use project venv
".venv\Scripts\python.exe" gui\app.py
goto end

:setup
echo [SETUP] Creating .venv and installing dependencies...
echo [SETUP] First run may take a few minutes.
%PY% -m venv .venv
if errorlevel 1 goto error
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto error
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto error

echo [OK] Setup done, starting GUI...
".venv\Scripts\python.exe" gui\app.py
goto end

:error
echo [ERROR] Failed to start.
echo Please install Python 3.10+ and add it to PATH.
pause
exit /b 1

:end
pause
