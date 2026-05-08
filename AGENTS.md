# AGENTS.md

This file provides guidance to Qian Shou when working with code in this repository.

## Project Overview

SubQ is a collection of long-context and intelligent Q&A projects. The main active project is **File-Chat**, a Go-based long-document intelligent Q&A system that provides an OpenAI-compatible API for seamless integration with chat frontends like NextChat.

### Architecture

```
NextChat (前端) ──HTTP/SSE──▶ file-chat (Go 后端) ──HTTP/SSE──▶ DeepSeek API
                            │
                            ├── 全局文件索引（跨对话共享）
                            ├── LLM 语义分片（20 并发）
                            ├── 基于 hash 的文件存储（去重）
                            └── markitdown 文档转换
```

## File-Chat Development

### Build Commands

```bash
# 编译 Go 后端
cd file-chat && go build -o file-chat

# Windows 快捷编译
cd file-chat && build.bat

# 安装 markitdown 依赖
pip install markitdown

# 运行测试
cd file-chat && go test ./...
```

### Configuration

环境变量配置（通过 `set` 设置或在启动脚本中配置）：

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `DEEPSEEK_API_KEY` | - | DeepSeek API 密钥（必填） |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | API 地址 |
| `MODEL` | `deepseek-v4-flash` | 使用模型 |
| `PORT` | `8880` | HTTP 服务端口 |
| `DATA_DIR` | `./data` | 数据存储目录 |
| `MARKITDOWN_CMD` | `markitdown` | markitdown 命令路径 |
| `CHUNK_TOKENS` | `2000` | 默认 chunk token 数 |
| `MAX_RETRIEVE` | `20` | 最大检索 chunk 数 |
| `SMALL_FILE_SIZE` | `15360` | 15KB，小文件阈值 |

### Running the Server

```bash
# 方式1：设置环境变量后运行
set DEEPSEEK_API_KEY=your-key-here
cd file-chat
file-chat.exe

# 方式2：使用启动脚本
cd file-chat
start-with-key.bat  # 需要先编辑填入 API Key
```

### Data Storage Structure

```
data/
├── files.json                    # 全局文件注册表
├── files/{hash[:2]}/{hash[2:4]}/{name}/
│   ├── outline                   # Per-file 大纲
│   ├── source                    # 转换后文本（文档文件）
│   └── chunks/                   # Chunk 文件
├── chats/{conversationID}/
│   └── chat-files.json           # 对话关联文件列表
└── global/
    ├── global_outline            # 全局大纲（所有文件）
    └── global_files_summary.xml  # 全局文件摘要
```

## Code Architecture

### Layer Structure

```
file-chat/
├── main.go                 # HTTP server + 路由
├── config.go               # 配置加载 + 硬件 ID + 许可证检查
├── handler/
│   ├── chat.go             # POST /v1/chat/completions 处理
│   └── models.go           # GET /v1/models 处理
├── service/
│   ├── chat.go             # 主流程编排（并行文件处理）
│   ├── chunker.go          # LLM 语义分片（并行 30KB/段，20并发）
│   ├── retriever.go        # 检索 + 上下文构建
│   ├── all_handler.go      # @全部 两级检索 + 全局大纲检索
│   ├── outline.go          # 大纲读写（全局 + per-file）
│   ├── summary.go          # files_summary.xml 管理 + LLM 摘要生成
│   ├── path_extractor.go   # @路径 + @全部 提取
│   └── converter.go        # 调用 markitdown 转换文档
├── llm/
│   ├── client.go           # DeepSeek API 客户端
│   └── sse.go              # SSE 流式写入 + 流代理
├── model/
│   ├── openai.go           # OpenAI API 请求/响应类型
│   └── types.go            # Chunk, Outline, FileRegistry 等核心类型
└── store/
    ├── storage_paths.go    # DataPaths — 基于 hash 的文件存储路径管理
    ├── filestore.go        # 文件系统操作（EnsureDir, WriteFile 等）
    ├── registry.go         # files.json 读写 + hash 计算
    ├── chat_files.go       # chat-files.json 对话文件关联读写
    ├── global_index.go     # 全局大纲和摘要读写 + 截断
    └── filelock.go         # 文件锁（并发安全）
```

### Request Flow

1. **接收请求** (`handler/chat.go`)
   - 解析 OpenAI 兼容的 `/v1/chat/completions` 请求
   - 从 `X-Conversation-Id` header 获取对话 ID

2. **提取路径** (`service/path_extractor.go`)
   - 从消息中提取 `@路径` 和 `@全部` 标记
   - 清理消息中的路径标记

3. **文件处理** (`service/chat.go`)
   - 并行处理文件（最多 20 并发）
   - 每个文件加文件锁，避免并发冲突
   - 文档格式 → markitdown 转换
   - < 15KB → LLM 生成摘要，存为单 chunk
   - ≥ 15KB → 并行 LLM 语义分片

4. **语义分片** (`service/chunker.go`)
   - 按 30KB 分段
   - 最多 20 个 goroutine 并行处理各段
   - LLM 返回 `chunk_id|摘要|起始行|结束行`
   - 按段顺序拼接，统一重编号 chunk ID（chunk_001, chunk_002...）

5. **检索** (`service/retriever.go`, `service/all_handler.go`)
   - `@全部` → 两级检索（全局摘要 → per-file outline）
   - `@path` → 全局大纲筛选 → LLM 排序 chunk
   - 取 top 20，按原顺序排列

6. **上下文构建** (`service/retriever.go`)
   - 读取 chunk 文件 + 从 source 扩展上下文
   - 上下各扩展 500~1200 byte
   - 截断规则：500B遇`\n\n` / 800B遇`\n` / 1200B强截

7. **LLM 调用** (`llm/client.go`)
   - 构造最终 prompt（system + context + history + query）
   - SSE 流式转发 DeepSeek 回复

### Key Data Structures

**ChunkMeta** (存储在 chunks.json):
```go
type ChunkMeta struct {
    ID        string // chunk_001
    StartByte int64  // 起始字节偏移
    EndByte   int64  // 结束字节偏移（不含）
    Head30    string // 前30字节指纹（hex）
    Tail30    string // 后30字节指纹（hex）
    Hash      string // 内容 MD5
    Summary   string // 30~150字摘要
}
```

**Chunk** (运行时使用):
```go
type Chunk struct {
    ID        string
    FilePath  string // 绝对路径
    Summary   string // LLM 生成的摘要
    StartByte int64
    EndByte   int64
}
```

**Outline 格式** (每行一条，`|` 分隔):
```
chunk_001|摘要内容|起始字节|结束字节
```

### Incremental Update Algorithm

文件变更时，基于位置对齐的增量更新（`service/chunker.go:IncrementalUpdateFile`）：

1. 逐 chunk 检查 hash
2. Hash 不一致 → 搜索对齐点（通过 Head30 在新内容中查找）
3. 对齐成功 → 局部重新处理变化区域
4. 对齐失败 → 从变化点全量重新拆解

### Important Constants

| 常量 | 值 | 说明 |
|-----|-----|------|
| `segmentSizeChars` | 30KB | 大文件并行分片的段大小 |
| `maxConcurrency` | 20 | 并行处理 goroutine 上限 |
| `maxChunkBytesNew` | 8000 | 新文件 chunk 超过此值按行边界拆分 |
| `maxChunkBytesUpdate` | 6000 | 增量更新时 chunk 拆分阈值 |

### File Change Detection

三级检测（`service/all_handler.go:IsFileChanged`）：
1. size 变化 → 需处理
2. modTime 变化 → 计算 MD5 hash
3. hash 变化 → 需处理

### File Lock Mechanism

并发处理同一文件时：
1. 创建 `{filedir}.lock` 文件（O_EXCL 原子操作）
2. 获取成功 → 执行处理
3. 获取失败 → 轮询等待（100ms 间隔，最多 10s）
4. 处理完成 → 删除 lock 文件

## CRBSA Project (Python)

CRBSA (Codebook-Routed Block-Sparse Attention) 是一个面向 1M~10M Token 超长上下文的注意力架构。

### Install

```bash
cd crbsa
pip install -e .
```

### Verification

```bash
# 不加载模型，纯模块测试
python scripts/verify.py --seq-len 2048 --debug

# 完整模型测试（需要 GPU）
python scripts/verify.py --model Qwen/Qwen3.6-35B-A3B --seq-len 4096 --debug
```

## Frontend (NextChat)

NextChat 通过 `X-Conversation-Id` header 传递对话 ID。

### Start Frontend

```bash
cd NextChat
npx next dev -p 3000
```

### NextChat Configuration

在 http://localhost:3000 设置页面：
- 自定义接口地址：`http://localhost:8880`
- API Key：填写你的 DeepSeek API Key
- 模型名称：`deepseek-v4-flash`

### Usage

新建对话，输入 `@文件绝对路径\文件名` + 你的提示词：
```
@F:\github\data\report.pdf 帮我分析关键数据
```

使用 `@全部` 搜索所有已索引文件：
```
总结所有文件的关键信息 @全部
```
npm: /f/Program Files/nodejs/

---