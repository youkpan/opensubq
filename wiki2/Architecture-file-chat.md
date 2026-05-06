# File-Chat 技术架构

## 1. 系统架构

```
┌──────────┐       ┌───────────────────────────────┐       ┌──────────────┐
│          │  HTTP │     file-chat (Go)             │  HTTP │  DeepSeek    │
│ NextChat │──────→│                                │──────→│  v4 Flash    │
│          │←──────│  ┌──────────┐  ┌────────────┐  │←──────│  API         │
│          │  SSE  │  │ API层    │  │ 服务层     │  │  SSE  │              │
└──────────┘       │  │ (handler)│  │ (service)  │  │       └──────────────┘
                   │  └──────────┘  └────────────┘  │
                   │       │              │          │       ┌──────────────┐
                   │  ┌──────────┐  ┌────────────┐  │       │ markitdown   │
                   │  │ 存储层   │  │ 外部工具   │──│──────→│ (Python CLI) │
                   │  │ (store)  │  └────────────┘  │       └──────────────┘
                   │  └──────────┘                   │
                   └───────────────────────────────┘
```

**自定义 Header**：NextChat 通过 `X-Conversation-Id` 传递对话 ID

## 2. 项目结构

```
file-chat/
├── main.go                 # 入口：HTTP server + 路由
├── config.go               # 配置加载（环境变量）
├── start.bat               # 通用启动脚本
├── start-with-key.bat      # 带 API Key 的启动脚本
├── handler/
│   ├── chat.go             # POST /v1/chat/completions
│   └── models.go           # GET  /v1/models
├── service/
│   ├── chat.go             # 主流程编排（并行文件处理）
│   ├── chunker.go          # LLM 语义分片（并行30KB/段，20并发）
│   ├── retriever.go        # 检索 + 上下文构建
│   ├── all_handler.go      # @全部 两级检索
│   ├── outline.go          # 大纲读写（全局 + per-file）
│   ├── summary.go          # files_summary.xml 管理 + LLM 摘要生成
│   ├── path_extractor.go   # @路径 + @全部 提取
│   └── converter.go        # 调用 markitdown 转换文档
├── llm/
│   ├── client.go           # DeepSeek API 客户端
│   └── sse.go              # SSE 流式写入 + 流代理
├── model/
│   ├── openai.go           # OpenAI API 请求/响应类型
│   └── types.go            # Chunk, Outline, FileRegistry, FileEntry
└── store/
    ├── filestore.go        # 文件系统操作 + JobPaths + SafeFileName
    └── registry.go         # files.json 读写 + hash 计算
```

## 3. 数据模型

### 3.1 目录结构

```
jobs/
├── files.json                                    # 全局文件注册表
└── {jobId}/
    ├── outline                                   # 全局聚合大纲
    ├── files_summary.xml                         # 文件摘要索引
    ├── outlines/                                 # per-file outline
    │   ├── F__github_subq_data_report.pdf
    │   └── F__github_subq_data_supplement.xlsx
    ├── sources/                                  # 转换后源文本
    │   ├── F__github_subq_data_report.pdf.txt
    │   └── F__github_subq_data_supplement.xlsx.txt
    └── chunks/                                   # 分片文件
        ├── F__github_subq_data_report.pdf_chunk_001
        ├── F__github_subq_data_report.pdf_chunk_002
        └── F__github_subq_data_supplement.xlsx_chunk_001
```

文件名通过 `SafeFileName` 生成：替换 `\`、`/`、`:` 为 `_`（兼容 Windows 路径）。

### 3.2 大纲格式

每行一条 chunk 记录，字段以 `|` 分隔：

```
chunk_001|F:\github\data\report.pdf|关于Q3营收数据的分析|10|50
chunk_002|F:\github\data\report.pdf|关于Q4市场展望的策略|51|120
```

| 字段 | 说明 |
|------|------|
| chunk_id | 片段唯一标识（chunk_NNN） |
| file_path | 文件绝对路径 |
| summary | LLM 生成的片段摘要（50~150字） |
| start_line | 起始行号 |
| end_line | 结束行号 |

### 3.3 files.json

```json
{
  "files": {
    "F:\\github\\data\\report.pdf": {
      "hash": "a1b2c3d4e5f6",
      "mod_time": "2026-05-07T10:30:00+08:00",
      "size": 153600,
      "processed_at": "2026-05-07T10:31:00+08:00"
    }
  }
}
```

### 3.4 files_summary.xml

```xml
<files>
  <file path="F:\github\data\report.pdf">该文件是Q3季度财务报告，包含营收分析、市场展望等...</file>
</files>
```

### 3.5 核心类型

```go
type Chunk struct {
    ID        string // chunk_001
    FilePath  string
    Summary   string // LLM 生成的摘要
    StartLine int
    EndLine   int
}

type Outline struct {
    JobID  string
    Chunks []Chunk
}

type FileEntry struct {
    Hash        string
    ModTime     string
    Size        int64
    ProcessedAt string
}

type FileRegistry struct {
    Files map[string]*FileEntry
}
```

## 4. 核心流程

### 4.1 主流程（chat handler）

```
收到 /v1/chat/completions 请求（含 X-Conversation-Id header）
    │
    ├─ 1. 提取 @路径 和 @全部 标记
    │
    ├─ 2. 确定 job ID（X-Conversation-Id 或 MD5 fallback）
    │
    ├─ 3. 加载 files.json，检测文件变更
    │      ├─ size+modTime 未变 → 跳过
    │      ├─ hash 未变 → 跳过
    │      └─ 文件已变化 → 清理旧数据
    │
    ├─ 4. 并行处理文件（最多 20 并发）
    │      ├─ 文档格式 → markitdown 转换
    │      ├─ < 15KB → LLM 生成摘要，存为单 chunk
    │      └─ ≥ 15KB → 并行 LLM 语义分片
    │
    ├─ 5. 顺序组装结果
    │      ├─ 追加全局 outline
    │      ├─ 写 per-file outline
    │      ├─ LLM 生成文件摘要 → files_summary.xml
    │      └─ 更新 files.json
    │
    ├─ 6. 检索
    │      ├─ @全部 → 两级检索（files_summary → per-file outline）
    │      └─ @path → 大纲 + query → LLM 排序 chunk
    │
    ├─ 7. 读取 top 20 chunk + 扩展上下文 → 拼接 prompt
    │
    └─ 8. SSE 流式转发 DeepSeek 回复
```

### 4.2 并行语义分片（chunker）

```
输入：大文件（≥ 15KB）
    │
    ├─ 1. 按 30KB 分段（按行边界切分）
    │
    ├─ 2. 启动最多 20 个 goroutine 并行处理
    │      每个 goroutine:
    │      ├─ 构造带行号的文本
    │      ├─ LLM prompt → 输出 1~3 个片段
    │      └─ 返回 chunks（临时 ID）
    │
    ├─ 3. 等待全部完成
    │
    ├─ 4. 按段顺序拼接，统一重编号 chunk ID
    │
    └─ 5. 写入 chunk 文件
    （失败段自动 fallback 为固定切分）
```

### 4.3 @全部 两级检索（all_handler）

```
输入：query + @全部
    │
    ├─ 1. 读取 files_summary.xml
    │
    ├─ 2. LLM 一级检索：文件摘要 + query → 选择相关文件（≤5个）
    │
    ├─ 3. 加载相关文件的 per-file outline
    │
    ├─ 4. LLM 二级检索：合并 outline + query → 选择 top 20 chunks
    │
    └─ 5. BuildContext 拼接上下文
```

### 4.4 检索 + 上下文构建（retriever）

```
输入：大纲 + 用户 query
    │
    ├─ 1. 大纲全文 + query → LLM 排序 chunk
    │
    ├─ 2. 取 top 20，按原顺序排列
    │
    ├─ 3. 读取 chunk 文件 + 从 sources/ 扩展上下文
    │      - 上下各扩展 500~1200 byte
    │      - 截断：500B遇\n\n / 800B遇\n / 1200B强截
    │
    └─ 4. 非连续 chunk 用 <chunk-{id}> 包裹
```

### 4.5 文件变更检测

```
处理文件前：
    │
    ├─ files.json 无记录 → 新文件，需处理
    │
    ├─ size + modTime 都未变 → 跳过
    │
    ├─ modTime 变了 → 计算 MD5 hash
    │   ├─ hash 未变 → 跳过
    │   └─ hash 变了 → 清理旧数据 + 重新处理
    │
    └─ 处理完成后更新 files.json
```

### 4.6 最终 Prompt 组装

```
system: 文档问答助手角色设定

user: 以下是相关文档内容：
  <context>
  <chunk-001>... 内容 + 扩展上下文 ...</chunk-001>
  <chunk-010>... 内容 + 扩展上下文 ...</chunk-010>
  </context>

assistant: 好的，我已经阅读了文档内容，请提出你的问题。

[对话历史（清理 @path 后的内容）]

user: [当前 query]
```

## 5. API 设计

### 5.1 POST /v1/chat/completions

完全兼容 OpenAI Chat Completion API。

**自定义 Header**：
| Header | 必填 | 说明 |
|--------|------|------|
| `X-Conversation-Id` | 否 | 对话 ID，为空时自动生成 |

**请求**：
```json
{
  "model": "deepseek-v4-flash",
  "messages": [
    {"role": "user", "content": "分析 @/data/report.pdf 的关键数据"}
  ],
  "stream": true
}
```

**响应（stream=true）**：SSE 格式
```
data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"根"},"index":0}]}
data: {"id":"chatcmpl-xxx","choices":[{"delta":{"content":"据"},"index":0}]}
...
data: [DONE]
```

### 5.2 GET /v1/models

```json
{
  "object": "list",
  "data": [
    {"id": "deepseek-v4-flash", "object": "model", "owned_by": "file-chat"}
  ]
}
```

## 6. 并行处理设计

### 6.1 跨文件并行（chat.go）

- 所有待处理文件放入 goroutine 池
- 信号量控制最多 20 并发
- 文件处理完成后顺序组装 outline、summary、registry

### 6.2 文件内并行（chunker.go）

- 大文件按 30KB 分段（`splitSegments`）
- 每段独立 goroutine 调用 LLM
- 全部完成后按段顺序拼接，统一重编号 chunk ID
- 失败段自动 fallback

## 7. 配置项

| 配置 | 环境变量 | 默认值 | 说明 |
|------|---------|--------|------|
| API Key | `DEEPSEEK_API_KEY` | - | DeepSeek API 密钥（必填） |
| API URL | `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | API 地址 |
| 模型 | `MODEL` | `deepseek-v4-flash` | 使用模型 |
| 端口 | `PORT` | `8080` | HTTP 服务端口 |
| Job 目录 | `JOBS_DIR` | `./jobs` | job 存储目录 |
| markitdown | `MARKITDOWN_CMD` | `markitdown` | markitdown 命令路径 |
| 检索数量 | `MAX_RETRIEVE` | `20` | 最大检索 chunk 数 |
| 小文件阈值 | `SMALL_FILE_SIZE` | `15360` | 15KB，低于此直接存 chunk |
| 分片段大小 | - | `30KB` | 大文件并行分片的段大小 |
| 最大并发数 | - | `20` | 并行处理 goroutine 上限 |

## 8. 错误处理

| 场景 | 处理 |
|------|------|
| @路径不存在 | 日志记录，继续处理其他路径 |
| markitdown 转换失败 | 日志记录，跳过该文件 |
| DeepSeek API 调用失败 | SSE 发送错误事件 |
| 分片段 LLM 输出格式错误 | fallback 为固定大小切分 |
| 分片段 LLM 调用失败 | fallback 为单 chunk |
| 大纲为空 | 直接透传请求给 DeepSeek |
| LLM 摘要生成失败 | 使用截断内容作为 fallback |

## 9. NextChat 修改

### 9.1 app/client/api.ts
`getHeaders()` 函数中添加 `X-Conversation-Id` header，值为当前 session ID。

### 9.2 app/api/common.ts
`requestOpenai()` 中透传 `X-Conversation-Id` header 到上游 API。

## 10. 部署

```
1. 安装 markitdown: pip install markitdown
2. 编译: go build -o file-chat
3. 配置 API Key:
   - 方式1: set DEEPSEEK_API_KEY=xxx && start.bat
   - 方式2: 编辑 start-with-key.bat 填入 key，双击启动
4. 部署 NextChat，API 地址指向 http://localhost:8080
```
