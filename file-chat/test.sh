#!/bin/bash
# file-chat 测试脚本

BASE_URL="http://localhost:8080"

echo "=== Test 1: 普通对话（无文件引用）==="
curl -s "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "你好，请简单介绍一下你自己"}],
    "stream": false
  }' | python -m json.tool 2>/dev/null || echo "Request failed"

echo ""
echo "=== Test 2: 引用文件 ===="
curl -s "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test" \
  -H "X-Conversation-Id: test-conv-001" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "总结下这个文件的主要内容 @F:/github/subq/file-chat/testfile.txt"}],
    "stream": false
  }' | python -m json.tool 2>/dev/null || echo "Request failed"

echo ""
echo "=== Test 3: 查看生成的 jobs 目录 ===="
find jobs/ -type f 2>/dev/null | head -20

echo ""
echo "=== Test 4: 查看 files.json ===="
cat jobs/files.json 2>/dev/null || echo "No files.json"

echo ""
echo "=== Test 5: 查看 files_summary.xml ===="
cat jobs/test-conv-001/files_summary.xml 2>/dev/null || echo "No files_summary.xml"

echo ""
echo "=== Test 6: 查看大纲 ===="
cat jobs/test-conv-001/outline 2>/dev/null || echo "No outline"

echo ""
echo "=== Test 7: 查看 per-file outline ===="
ls jobs/test-conv-001/outlines/ 2>/dev/null || echo "No outlines"

echo ""
echo "=== Test 8: 追问（同对话）==="
curl -s "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test" \
  -H "X-Conversation-Id: test-conv-001" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [
      {"role": "user", "content": "总结下这个文件的主要内容 @F:/github/subq/file-chat/testfile.txt"},
      {"role": "assistant", "content": "这是一个测试文件。"},
      {"role": "user", "content": "性能指标部分说了什么？"}
    ],
    "stream": false
  }' | python -m json.tool 2>/dev/null || echo "Request failed"

echo ""
echo "=== Test 9: GET /v1/models ===="
curl -s "$BASE_URL/v1/models" | python -m json.tool 2>/dev/null || echo "Request failed"

echo ""
echo "=== All tests done ===="
