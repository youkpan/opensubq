@echo off
chcp 65001 >nul
echo ============================================
echo    AI 文件对话系统 - Release 编译打包
echo ============================================
echo.

REM ====== 1. 编译 NextChat 前端 ======
echo [1/4] 正在编译 NextChat 前端...
cd /d "%~dp0NextChat"
if errorlevel 1 (
    echo ERROR: 无法进入 NextChat 目录
    pause
    exit /b 1
)

yarn export
if errorlevel 1 (
    echo ERROR: NextChat 编译失败
    pause
    exit /b 1
)
echo [1/4] NextChat 编译完成
echo.

REM ====== 2. 准备 Release 目录 ======
echo [2/4] 正在准备 Release 目录...
cd /d "%~dp0"
if not exist Release mkdir Release
if exist Release\dist rmdir /S /Q Release\dist
if exist file-chat\dist rmdir /S /Q file-chat\dist
mkdir Release\dist

REM 复制 NextChat 静态文件到 Release/dist/
xcopy /E /I /Y "NextChat\out\*" "Release\dist\" >nul
xcopy /E /I /Y "NextChat\out\*" "file-chat\dist\" >nul
echo [2/4] 静态文件已复制到 Release\dist\
echo.

REM ====== 3. 编译 Go 后端 ======
echo [3/4] 正在编译 Go 后端...
cd /d "%~dp0file-chat"
set GOOS=windows
set GOARCH=amd64
go build -ldflags "-s -w" -o "%~dp0Release\file-chat.exe"
if errorlevel 1 (
    echo ERROR: Go 编译失败
    pause
    exit /b 1
)
echo [3/4] Go 后端编译完成
echo.

REM ====== 4. 复制启动文件 ======
echo [4/4] 正在复制启动文件...
cd /d "%~dp0"
copy /Y "Release\启动AI文件对话_Run_.bat" "Release\启动AI文件对话_Run_.bat" >nul 2>nul
copy /Y "Release\使用说明.txt" "Release\使用说明.txt" >nul 2>nul
echo [4/4] 启动文件已就绪
echo.

echo ============================================
echo    打包完成！
echo ============================================
echo.
echo Release 目录结构：
echo   file-chat.exe              Go 后端（含静态文件服务）
echo   dist\                      NextChat 前端静态文件
echo   启动AI文件对话_Run_.bat    启动脚本
echo   使用说明.txt               使用文档
echo.
echo 运行方式：
echo   1. 配置 DEEPSEEK_API_KEY 环境变量
echo   2. 双击 Release\启动AI文件对话_Run_.bat
echo.
pause
