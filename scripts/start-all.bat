@echo off
REM NextChat + file-chat 联合测试启动脚本

REM 启动 file-chat 后端（如果未运行）
start "file-chat" cmd /c "cd /d F:\github\subq\file-chat && set DEEPSEEK_API_KEY= && file-chat.exe"

REM 等待后端启动
timeout /t 3 /nobreak >nul

REM 启动 NextChat
cd /d F:\github\subq\NextChat
set PATH=F:\Program Files\nodejs;%PATH%
set OPENAI_API_KEY=
npm run dev
