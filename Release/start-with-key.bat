@echo off
REM File-Chat 启动脚本 (带 API Key)
REM 请将下面的 sk-xxx 替换为你的 DeepSeek API Key

set DEEPSEEK_API_KEY=sk-...

echo Starting file-chat server on http://localhost:8880 ...
file-chat.exe
