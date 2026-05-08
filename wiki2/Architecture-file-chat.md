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
│   └── types.go            # Chunk, Outline, FileRegistry, FileEntry, ChatFiles
└── store/
    ├── storage_paths.go    # DataPaths — 基于 hash 的文件存储路径管理
    ├── filestore.go        # 文件系统操作（EnsureDir, WriteFile, SafeFileName 等）
    ├── registry.go         # files.json 读写 + hash 计算
    ├── chat_files.go       # chat-files.json 对话文件关联读写
    ├── global_index.go     # 全局大纲和摘要读写 + 截断
    └── filelock.go         # 文件锁（并发安全）
```

## 3. 数据模型

### 3.1 目录结构

```
data/
├── files.json                                    # 全局文件注册表
├── files/                                        # 文件数据（按路径hash分桶）
│   ├── 75/ba/F__github_subq_file-chat_饮马流花河.txt/
│   │   ├── outline                               # per-file 大纲
│   │   ├── source                                # 转换后文本
│   │   └── chunks/
│   │       ├── chunk_001
│   │       └── chunk_002
│   └── ...
├── chats/                                        # 对话数据
│   └── {conversationID}/
│       └── chat-files.json                       # 对话关联文件列表
└── global/                                       # 全局索引
    ├── global_outline                            # 全局大纲汇总
    └── global_files_summary.xml                  # 全局文件摘要
```

文件存储路径规则：`data/files/{路径hash前2位}/{路径hash 3~4位}/{SafeFileName}/`
- hash 使用文件**绝对路径**的 MD5（不是文件内容 hash）
- 文件名通过 `SafeFileName` 生成：替换 `\`、`/`、`:` 为 `_`

### 3.2 大纲格式

每行一条 chunk 记录，字段以 `|` 分隔：

```
chunk_001|F:\github\data\report.pdf|关于Q3营收数据的分析|10|50
chunk_002|F:\github\data\report.pdf|关于Q4市场展望的策略|51|120
```

| 字段 | 说明 |
|------|------|
| chunk_id | 片段唯一标识（chunk_NNN，全局递增，无后缀） |
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
      "processed_at": "2026-05-07T10:31:00+08:00",
      "chunk_count": 12,
      "summary": "该文件是Q3季度财务报告，包含营收分析、市场展望等..."
    }
  }
}
```

### 3.4 global_files_summary.xml

```xml
<files>
  <file path="F:\github\data\report.pdf">该文件是Q3季度财务报告，包含营收分析、市场展望等...</file>
</files>
```

### 3.5 chat-files.json

```json
{
  "conversation_id": "abc123",
  "files": [
    "F:\\github\\data\\report.pdf",
    "F:\\github\\data\\supplement.xlsx"
  ],
  "folders": [
    "F:\\github\\data"
  ],
  "updated_at": "2026-05-08T10:30:00+08:00"
}
```

### 3.6 核心类型

```go
type ChunkMeta struct {
    ID        string // chunk_001（简单递增，无后缀）
    StartByte int64  // 起始字节偏移
    EndByte   int64  // 结束字节偏移（不含）
    Head30    string // 前30字节指纹（hex）
    Tail30    string // 后30字节指纹（hex）
    Hash      string // 内容 MD5
    Summary   string // 30~150字摘要
}

type Chunk struct {
    ID        string // chunk_001（全局唯一）
    FilePath  string // 绝对路径
    Summary   string // LLM 生成的摘要
    StartByte int64
    EndByte   int64
}

type Outline struct {
    Chunks []Chunk
}

type FileEntry struct {
    Hash        string
    ModTime     string
    Size        int64
    ProcessedAt string
    ChunkCount  int
    Summary     string
}

type FileRegistry struct {
    Files map[string]*FileEntry
}

type ChatFiles struct {
    ConversationID string
    Files          []string
    Folders        []string
    UpdatedAt      string
}
```

## 4. 核心流程

### 4.1 主流程（chat handler）

```
收到 /v1/chat/completions 请求（含 X-Conversation-Id header）
    │
    ├─ 1. 确定 conversation ID（header 或 MD5 fallback）
    │
    ├─ 2. 提取 @路径 和 @全部 标记
    │
    ├─ 3. 初始化 data/ 目录结构
    │
    ├─ 4. 加载全局注册表 data/files.json + 全局大纲
    │
    ├─ 5. 解析路径，检测文件变更
    │      ├─ registry 无记录 → 新文件，需处理
    │      ├─ size+modTime 未变 → 跳过
    │      ├─ hash 未变 → 跳过
    │      └─ 文件已变化 → 清理旧数据（加文件锁）
    │
    ├─ 6. 并行处理文件（最多 20 并发，每文件加锁）
    │      ├─ 文档格式 → markitdown 转换
    │      ├─ < 15KB → LLM 生成摘要，存为单 chunk
    │      └─ ≥ 15KB → 并行 LLM 语义分片
    │
    ├─ 7. 顺序组装结果
    │      ├─ 追加到全局大纲 data/global/global_outline
    │      ├─ 写 per-file outline（data/files/{hash}/outline）
    │      ├─ LLM 生成文件摘要 → data/global/global_files_summary.xml
    │      ├─ 更新全局注册表 data/files.json
    │      └─ 更新对话文件列表 data/chats/{id}/chat-files.json
    │
    ├─ 8. 检索
    │      ├─ @全部 → 两级检索（全局摘要 → per-file outline）
    │      │         fallback: 全局大纲直接检索（超1MB截断）
    │      └─ @path → 全局大纲筛选 → LLM 排序 chunk
    │
    ├─ 9. 读取 top 20 chunk + 扩展上下文 → 拼接 prompt
    │
    └─ 10. SSE 流式转发 DeepSeek 回复
```

### 4.2 文件存储路径

```
文件: F:\github\data\report.pdf
路径 hash (MD5): 75ba443e5ea3c06e...
SafeFileName: F__github_data_report.pdf

存储路径: data/files/75/ba/F__github_data_report.pdf/
          ├── outline          # per-file 大纲
          ├── source           # 转换后文本
          └── chunks/
              ├── chunk_001    # 分片内容
              └── chunk_002
```

### 4.3 并行语义分片（chunker）

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
    ├─ 4. 按段顺序拼接，统一重编号 chunk ID（全局递增，格式 chunk_001, chunk_002...）
    │
    ├─ 5. 超过 8000 字节的 chunk 按行边界拆分，分配新的连续 ID
    │
    └─ 6. 写入 chunk 文件到 data/files/{hash}/chunks/
    （失败段自动 fallback 为固定切分）
```

### 4.4 @全部 两级检索（all_handler）

```
输入：query + @全部
    │
    ├─ 1. 读取全局摘要 data/global/global_files_summary.xml
    │
    ├─ 2. LLM 一级检索：全局摘要 + query → 选择相关文件（≤5个）
    │
    ├─ 3. 加载相关文件的 per-file outline（从 data/files/{hash}/outline）
    │
    ├─ 4. LLM 二级检索：合并 outline + query → 选择 top 20 chunks
    │
    ├─ 5. 如果两级检索无结果 → fallback 到全局大纲直接检索
    │      ├─ 读取 data/global/global_outline
    │      ├─ 超过 1MB → 截断到最后 1MB（按换行截断）
    │      └─ LLM 检索 top 20 chunks
    │
    └─ 6. BuildContext 拼接上下文
```

### 4.5 检索 + 上下文构建（retriever）

```
输入：大纲 + 用户 query
    │
    ├─ 1. 大纲全文 + query → LLM 排序 chunk
    │
    ├─ 2. 取 top 20，按原顺序排列
    │
    ├─ 3. 读取 chunk 文件 + 从 source 扩展上下文
    │      - 上下各扩展 500~1200 byte
    │      - 截断：500B遇\n\n / 800B遇\n / 1200B强截
    │
    └─ 4. 非连续 chunk 用 <chunk-{id}> 包裹
```

### 4.6 增量更新（IncrementalUpdateFile）

文件变更时，基于位置对齐的增量更新算法，避免全量重新处理：

```
输入：旧 chunks.json + 新文件内容
    │
    ├─ 1. 逐 chunk 检查 hash
    │      计算 contentBytes[old.StartByte:old.EndByte] 的 hash
    │      ├─ hash 一致 → chunk 未变，保留（仅调整 ID）
    │      └─ hash 不一致 → 变化点，进入步骤 2
    │
    ├─ 2. 搜索对齐点
    │      从变化点的下一个 chunk 开始，搜索 Head30 在新内容中的位置
    │      ├─ chunk[i+1] 找到 → 对齐成功
    │      ├─ chunk[i+1] 未找到 → 尝试 chunk[i+2], chunk[i+3]...
    │      └─ 全部未找到 → 进入步骤 4（全量重新拆解）
    │
    ├─ 3. 对齐成功 → 局部重新处理
    │      ├─ 重新处理 [changeStart, alignPos) 区域
    │      │   ├─ ≤ 6000 字节 → LLM 生成摘要，单 chunk
    │      │   └─ > 6000 字节 → 按行边界拆分 + LLM 生成摘要
    │      ├─ 新 chunk ID 按顺序命名（如 chunk_010 拆出 3 个 → chunk_010/011/012）
    │      ├─ 后续 chunk ID 顺延（原 chunk_011 → chunk_013）
    │      ├─ 后续 chunk 的 byte offset 按偏移量调整（原地修改）
    │      └─ 从 alignIdx 继续循环，逐个验证后续 chunk
    │
    ├─ 4. 全量重新拆解（对齐失败时）
    │      └─ 从变化点开始，调用 processLargeFileParallel 完整流程
    │
    └─ 5. 更新 chunks.json + outline
```

**示例**：
```
旧 chunks: chunk_001(A) chunk_002(B) chunk_003(C) chunk_004(D)
B 区域内容变化，C 的起始位置找到对齐 → B 拆为 2 个 chunk

新 chunks: chunk_001(A) chunk_002(B') chunk_003(B'') chunk_004(C) chunk_005(D)
                                         ↑ 原chunk_003  ↑ 原chunk_004
                                         ID 顺延 +2     ID 顺延 +2
```

### 4.7 文件变更检测

```
处理文件前：
    │
    ├─ files.json 无记录 → 新文件，需处理
    │
    ├─ size + modTime 都未变 → 跳过
    │
    ├─ modTime 变了 → 计算 MD5 hash
    │   ├─ hash 未变 → 跳过
    │   └─ hash 变了 → 加文件锁 → 清理旧数据 + 重新处理
    │
    └─ 处理完成后更新 files.json
```

### 4.8 文件锁机制

```
并发处理同一文件时：
    │
    ├─ 创建 {filedir}.lock 文件（O_EXCL 原子操作）
    ├─ 获取成功 → 执行处理
    ├─ 获取失败 → 轮询等待（100ms间隔，最多10s）
    └─ 处理完成 → 删除 lock 文件
```

### 4.9 最终 Prompt 组装

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
- 每个文件加文件锁，避免并发冲突
- 文件处理完成后顺序组装全局大纲、摘要、注册表

### 6.2 文件内并行（chunker.go）

- 大文件按 30KB 分段（`splitSegments`）
- 每段独立 goroutine 调用 LLM
- 全部完成后按段顺序拼接，统一重编号 chunk ID（全局递增，格式 chunk_001, chunk_002...）
- 超过 8000 字节的 chunk 按行边界拆分，分配新的连续 ID
- 失败段自动 fallback

## 7. 配置项

| 配置 | 环境变量 | 默认值 | 说明 |
|------|---------|--------|------|
| API Key | `DEEPSEEK_API_KEY` | - | DeepSeek API 密钥（必填） |
| API URL | `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | API 地址 |
| 模型 | `MODEL` | `deepseek-v4-flash` | 使用模型 |
| 端口 | `PORT` | `8880` | HTTP 服务端口 |
| 数据目录 | `DATA_DIR` | `./data` | 数据存储目录 |
| markitdown | `MARKITDOWN_CMD` | `markitdown` | markitdown 命令路径 |
| 检索数量 | `MAX_RETRIEVE` | `20` | 最大检索 chunk 数 |
| 小文件阈值 | `SMALL_FILE_SIZE` | `15360` | 15KB，低于此直接存 chunk |
| 新文件拆分阈值 | - | `8000` | 新文件 chunk 超过此值按行边界拆分 |
| 增量更新拆分阈值 | - | `6000` | 增量更新时 chunk 超过此值拆分 |
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
| 文件锁超时 | 返回错误，跳过该文件 |
| 全局大纲超 1MB | 截断到最后 1MB（按换行截断） |
| 增量更新对齐失败 | 从变化点全量重新拆解 |
| 增量更新无旧 chunks | fallback 到新文件处理流程 |

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
4. 部署 NextChat，API 地址指向 http://localhost:8880
```
