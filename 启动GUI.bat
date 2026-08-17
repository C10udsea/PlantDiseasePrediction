@echo off
chcp 65001 >nul
cd /d "%~dp0"
title Plant Disease GUI

echo ============================================
echo   Plant Disease - 智慧农业病虫害识别 GUI
echo ============================================
echo.

rem --- 1. 优先使用系统 Python(已有 torch+tkinter 则直接跑) ---
set "PY=python"
where python >nul 2>nul
if errorlevel 1 set "PY=py -3"

%PY% -c "import torch, tkinter" >nul 2>nul
if not errorlevel 1 (
    echo [OK] 检测到系统 Python 已具备 torch/tkinter
    echo [RUN] %PY% gui\app.py
    %PY% gui\app.py
    goto :end
)

rem --- 2. 否则使用项目虚拟环境 ---
if exist ".venv\Scripts\python.exe" (
    echo [OK] 使用项目虚拟环境 .venv
    echo [RUN] .venv\Scripts\python.exe gui\app.py
    ".venv\Scripts\python.exe" gui\app.py
    goto :end
)

rem --- 3. 都没有则自动创建虚拟环境并安装依赖 ---
echo [SETUP] 未检测到可用环境,首次运行将创建 .venv 并安装依赖...
echo [SETUP] 请耐心等待,可能需要数分钟(首次需下载 PyTorch)。
%PY% -m venv .venv
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :error
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :error
echo [OK] 环境准备完成,启动 GUI...
".venv\Scripts\python.exe" gui\app.py
goto :end

:error
echo.
echo [ERROR] 启动失败。请确认:
echo   1. 已安装 Python 3.10+ (https://www.python.org/downloads/)
echo   2. 安装时勾选 "Add Python to PATH"
echo   3. 网络可访问 PyPI(首次安装依赖需要联网)
echo.
pause
exit /b 1

:end
echo.
pause
