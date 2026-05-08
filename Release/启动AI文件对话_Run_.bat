@echo off
chcp 65001 >nul
cd /d "%~dp0"

REM ====== API Key 检查 ======
if "%DEEPSEEK_API_KEY%"=="" (
    echo.
    echo ============================================
    echo    API Key 未配置
    echo ============================================
    echo.
    echo 请设置 DeepSeek API Key 后再启动：
    echo.
    echo   方式 1 - 设置环境变量（推荐）：
    echo     set DEEPSEEK_API_KEY=sk-your-key-here
    echo     启动AI文件对话_Run_.bat
    echo.
    echo   方式 2 - 直接编辑此 bat 文件：
    echo     在第 20 行填入你的 API Key
    echo.
    echo   获取 API Key：https://platform.deepseek.com/
    echo.
    pause
    exit /b 1
)

REM ====== 可选：直接在此填入 API Key ======
REM set DEEPSEEK_API_KEY=sk-your-key-here

REM ====== 启动服务 ======
echo.
echo ============================================
echo    AI 文件对话服务正在启动...
echo ============================================
echo.
echo 浏览器将自动打开 http://localhost:8080
echo.

start "" "http://localhost:8080"
file-chat.exe
