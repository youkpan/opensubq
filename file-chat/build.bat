@echo off
REM File-Chat 启动脚本
REM API Key 配置方式：设置环境变量 DEEPSEEK_API_KEY

if "%DEEPSEEK_API_KEY%"=="" (
    echo ERROR: DEEPSEEK_API_KEY is not set
    echo.
    echo Please set it before running:
    echo   set DEEPSEEK_API_KEY=sk-your-key-here
    echo   start.bat
    echo.
    echo Or edit this file to set the key directly.
    pause
    exit /b 1
)

echo Starting file-chat server...
file-chat.exe

REM Other config (uncomment to customize):
REM set DEEPSEEK_BASE_URL=https://api.deepseek.com
REM set MODEL=deepseek-v4-flash
REM set PORT=8080
REM set JOBS_DIR=./jobs
REM set MARKITDOWN_CMD=markitdown
REM set MAX_RETRIEVE=20
REM set SMALL_FILE_SIZE=15360
