@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion
title VAE-LSTM-Sync WebUI

cd /d "%~dp0"

echo ========================================
echo   VAE-LSTM-Sync WebUI  Launcher
echo ========================================
echo.

REM ============================================================
REM  Optional: set USER_PORT to a fixed value (e.g. 9000).
REM  Leave empty to be prompted at start.
REM ============================================================
set "USER_PORT="
REM ============================================================

REM ===== 0. ask for port =====
set "INPUT_PORT=%USER_PORT%"
if defined INPUT_PORT goto port_ready
set /p "INPUT_PORT=Enter port (press Enter for default 7860): "
if not defined INPUT_PORT set "INPUT_PORT=7860"
:port_ready
echo.

REM ===== 1. venv =====
if not exist "env\Scripts\python.exe" (
    echo [ERROR] venv not found: env\Scripts\python.exe
    goto end
)
echo [INFO] venv OK

REM ===== 2. local ffmpeg to PATH =====
set "PATH=%~dp0ffmpeg\bin;%PATH%"
echo [INFO] ffmpeg bin: %~dp0ffmpeg\bin

REM ===== 3. Gradio =====
.\env\Scripts\python.exe -c "import gradio" 2>nul
if errorlevel 1 (
    echo [INFO] Gradio not installed, installing from TUNA mirror...
    .\env\Scripts\pip.exe install "gradio>=4.0.0,<5.0.0" -i https://pypi.tuna.tsinghua.edu.cn/simple
    if errorlevel 1 (
        echo [ERROR] Gradio install failed
        goto end
    )
)
echo [INFO] Gradio OK

REM ===== 4. kill leftover webui.py processes =====
echo [INFO] Killing leftover webui.py processes (if any)...
set "KILLED=0"
set "WPID="

REM 1) find PIDs whose CommandLine contains webui.py
for /f "usebackq tokens=2 delims==" %%P in (`
    powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -and $_.CommandLine -match 'webui.py' } | Select-Object -ExpandProperty ProcessId"
`) do (
    if not "%%P"=="" if not "%%P"==" " (
        set "KILLED=1"
        echo    killing PID %%P
        taskkill /F /PID %%P >nul 2>&1
    )
)

if "%KILLED%"=="0" echo    (none)

REM ===== 5. validate port =====
set "FREE_PORT="
set "CHK="
for /f "tokens=*" %%C in ('echo !INPUT_PORT!^| findstr /R "^[0-9][0-9]*$"') do set "CHK=%%C"
if not defined CHK (
    echo [WARN] "!INPUT_PORT!" is not a number, fallback to 7860
    set "INPUT_PORT=7860"
) else (
    if !INPUT_PORT! LSS 1024 (
        echo [WARN] Port !INPUT_PORT! too small, fallback to 7860
        set "INPUT_PORT=7860"
    )
    if !INPUT_PORT! GTR 65535 (
        echo [WARN] Port !INPUT_PORT! too large, fallback to 7860
        set "INPUT_PORT=7860"
    )
)
set "FREE_PORT=!INPUT_PORT!"

REM ===== 6. detect LISTENING process on the port =====
set "BUSY=0"
for /f "tokens=5" %%P in ('
    netstat -ano ^| findstr /C:":!FREE_PORT! " ^| findstr "LISTENING"
') do (
    set "BUSY=1"
    echo [INFO] Listener on port !FREE_PORT! is PID=%%P
)
if "%BUSY%"=="1" (
    echo.
    echo  ============================================================
    echo   [PORT BUSY] Port !FREE_PORT! is currently LISTENING.
    echo              Aborting. Re-run with another port,
    echo              or close the existing process.
    echo  ============================================================
    echo.
    set "EXITCODE=2"
    goto end
)

echo [INFO] Using port: !FREE_PORT!
echo [INFO] URL:     http://127.0.0.1:!FREE_PORT!
echo [INFO] Browser will open automatically when WebUI is ready.

echo.
echo ========================================
echo  Press Ctrl+C to stop WebUI
echo ========================================
echo.

REM ===== 7. start WebUI; port via env var =====
set "WEBUI_HOST=127.0.0.1"
set "WEBUI_PORT=!FREE_PORT!"
".\env\Scripts\python.exe" "webui.py"
set "EXITCODE=%errorlevel%"

echo.
echo ========================================
echo  WebUI exited (code=%EXITCODE%)
echo ========================================

:end
echo.
echo [DONE] Press any key to close...
pause >nul
exit /b %EXITCODE%
