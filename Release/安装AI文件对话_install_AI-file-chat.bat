@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo    AI 文件对话系统 - 安装向导
echo ============================================
echo.

REM ====== 1. 检查 Python ======
echo [1/4] 检查 Python 环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo.
    echo ============================================
    echo    Python 未安装
    echo ============================================
    echo.
    echo 即将安装 Python 3.13.4...
    echo.
    pause

    start /wait python-3.13.4-amd64.exe

    echo.
    echo Python 安装完成，验证中...
    python --version >nul 2>&1
    if errorlevel 1 (
        echo.
        echo 错误：Python 安装验证失败！
        echo 请手动安装 Python 后重试。
        echo.
        pause
        exit /b 1
    )
) else (
    echo Python 已安装
)
echo [1/4] Python 环境检查完成
echo.

REM ====== 2. 安装 markitdown ======
echo [2/4] 正在安装 markitdown 库...
python -m pip install markitdown -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
    echo.
    echo 错误：markitdown 安装失败！
    echo.
    pause
    exit /b 1
)
echo [2/4] markitdown 安装完成
echo.

REM ====== 3. 配置 DeepSeek API Key ======
echo [3/4] 配置 DeepSeek API Key...
echo.
echo ============================================
echo    获取 DeepSeek API Key
echo ============================================
echo.
echo 请按以下步骤操作：
echo.
echo   1. 浏览器将自动打开 DeepSeek 平台
echo   2. 登录你的账号
echo   3. 在 API Keys 页面创建新的 Key
echo   4. 复制你的 API Key（格式：sk-xxxxx）
echo.
pause

REM 打开 DeepSeek 平台
start "" "https://platform.deepseek.com/"

:input_apikey
echo.
echo 请输入你的 DeepSeek API Key （点击鼠标右键 粘贴）：
echo.
set /p API_KEY="API Key: "

REM 验证 API Key 格式
echo %API_KEY% | findstr /B /C:"sk-" >nul
if errorlevel 1 (
    echo.
    echo 错误：API Key 格式不正确！
    echo 正确的格式应该以 "sk-" 开头
    echo.
    goto input_apikey
)

echo.
echo API Key 已设置：%API_KEY%
echo [3/4] API Key 配置完成
echo.

REM ====== 4. 更新启动脚本 ======
echo [4/4] 正在配置启动脚本...

REM 使用 Python 替换 API Key
python -c "import codecs; content = codecs.open('启动AI文件对话_Run_.bat', 'r', 'utf-8').read(); codecs.open('启动AI文件对话_Run_.bat', 'w', 'utf-8').write(content.replace('set DEEPSEEK_API_KEY=sk-your-key-here', 'set DEEPSEEK_API_KEY=%API_KEY%'))"

echo [4/4] 启动脚本配置完成
echo.

REM ====== 完成 ======
echo ============================================
echo    安装完成！
echo ============================================
echo.
echo 安装内容：
echo   ✓ Python 3.13.4
echo   ✓ markitdown 库
echo   ✓ DeepSeek API Key 已配置
echo.
echo 下一步：
echo   双击 [启动AI文件对话_Run_.bat] 启动服务
echo.
pause
