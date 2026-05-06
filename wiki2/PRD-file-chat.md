# File-Chat 产品需求文档 (PRD)

## 1. 项目概述

**File-Chat** 是一个基于上下文工程的长文本智能问答系统。用户在对话中通过 `@路径` 引用服务器上的文件或文件夹，系统自动将长文本按语义拆分为片段并建立大纲索引；查询时，LLM 根据大纲智能检索相关片段，拼接上下文后生成回答。

后端使用 Go，调用 DeepSeek v4 Flash；前端直接部署 NextChat，后端提供 OpenAI 兼容 API。

## 2. 核心特性

| 特性 | 说明 |
|------|------|
| 文件引用 | 对话中使用 `@路径` 引用服务器文件/文件夹 |
| 语义分片 | LLM 驱动的语义拆分，自动生成摘要与大纲 |
| 智能检索 | LLM 读取大纲 + query，排序 chunk 相关性，取 top 20 |
| 多轮对话 | 追问时增量更新大纲，不重新处理已有文件 |
| 文档支持 | doc/pdf/excel/ppt 等通过 markitdown 自动转换 |
| 流式响应 | SSE 转发 LLM 输出，实时返回 |
| OpenAI API 兼容 | 可直接接入 NextChat |
| 文件变更检测 | 基于 MD5 hash + 修改时间的变更检测，避免重复处理 |
| @全部 | 引用所有已处理文件，两级检索（文件→片段） |
| 并行处理 | 文件间和大文件内均支持 20 并发并行分片 |
| 对话 ID | NextChat 通过 `X-Conversation-Id` header 传递会话 ID |

## 3. 用户流程

### 3.1 添加文件并提问

```
用户: 帮我分析一下 @/data/report.pdf 里的关键数据

1. 后端提取路径 /data/report.pdf
2. markitdown 转换为文本
3. ≥15KB → LLM 语义分片（30KB/段，20并发） → 生成大纲 + chunk 文件
4. LLM 为每个 chunk 生成摘要
5. LLM 为文件生成整体摘要，写入 files_summary.xml
6. 大纲 + query → LLM 选出 top 20 chunk
7. 读取 chunk + 扩展上下文 → 拼接 prompt
8. 调用 DeepSeek → SSE 流式返回回答
```

### 3.2 追问（同一对话）

```
用户: 再看看 @/data/supplement.xlsx，和之前的数据对比下

1. 提取新路径，检查 files.json（hash 不变则跳过）
2. 处理新文件（增量分片）
3. 更新大纲、files_summary.xml
4. 新大纲 + 新 query → 检索 → 回答
```

### 3.3 @全部 检索

```
用户: 总结下所有文件的关键信息 @全部

1. 读取 files_summary.xml（所有文件摘要）
2. LLM 根据查询匹配相关文件（一级检索）
3. 加载匹配文件的 per-file outline
4. LLM 从 outline 中选择相关 chunks（二级检索）
5. 拼接上下文，生成回答
```

### 3.4 普通对话（无 @文件）

```
用户: 你好，帮我写个函数

→ 直接透传给 DeepSeek，不触发分片和检索流程
```

## 4. 功能需求

### 4.1 文件处理

- 文本文件（txt/md/go/py/java/json 等）：直接读取
- 文档文件（doc/docx/pdf/xls/xlsx/ppt/pptx）：调用 markitdown CLI 转换
- **< 15KB**：LLM 生成摘要后存为单个 chunk
- **≥ 15KB**：按 30KB 分段，最多 20 并发 LLM 语义分片
- **文件夹**：递归扫描目录下所有支持的文件，并行处理
- 不支持的文件格式：跳过并提示用户

### 4.2 文件变更检测

- 全局注册表 `files.json` 记录文件 hash、修改时间、大小
- 三级检测：size → modTime → MD5 hash
- 文件未变化：跳过处理
- 文件已变化：自动清理旧 chunks/outline/summary，重新处理

### 4.3 LLM 语义分片（并行）

- 大文件按 30KB 分段
- 最多 20 个 goroutine 并行处理各段
- 每段 LLM 输出 1~3 个片段（chunk_id|路径|摘要|起始行|结束行）
- 全部完成后按段顺序拼接，统一重编号 chunk ID
- 每段处理失败时自动 fallback 为固定切分

### 4.4 大纲管理

- 全局大纲：`jobs/{jobId}/outline`，所有 chunk 的聚合视图
- 文件大纲：`jobs/{jobId}/outlines/{safeName}`，每文件独立
- 文件摘要：`jobs/{jobId}/files_summary.xml`，文件路径 + LLM 生成的整体摘要
- 所有 summary 均由 LLM 生成（50~150 字）

### 4.5 智能检索

1. 读取大纲全文
2. 构造 prompt：大纲 + 用户 query
3. LLM 返回按相关性排序的 chunk ID 列表
4. 取 **top 20**
5. 按 chunk 在原文中的顺序排列
6. 读取每个 chunk 文件内容
7. 从原始文件扩展上下文（每侧 500~1200 byte）

#### 上下文扩展规则
从 chunk 起止行向上下扩展时：
- 读到 500 byte 后遇到 `\n\n` → 截取
- 否则读到 800 byte 后遇到 `\n` → 截取
- 否则到 1200 byte 强制截取

#### 非连续 chunk 拼接
不连续的 chunk 内容用 XML 标签包裹：
```xml
<chunk-001>
... chunk 001 内容 + 扩展上下文 ...
</chunk-001>
```

### 4.6 @全部 支持

- 用户输入 `@全部` 时触发两级检索
- 一级：files_summary.xml + query → LLM 选择相关文件（最多5个）
- 二级：选中文件的 per-file outline + query → LLM 选择相关 chunks
- 复用 BuildContext 拼接最终上下文

### 4.7 API 接口

提供 OpenAI Chat Completion 兼容接口：

| 接口 | 方法 | 说明 |
|------|------|------|
| `/v1/chat/completions` | POST | 聊天补全，支持 stream |
| `/v1/models` | GET | 返回可用模型列表 |

- 支持 `stream: true`，返回 SSE 格式（`data: {...}\n\n`，以 `data: [DONE]` 结尾）
- 自定义 header：`X-Conversation-Id`（NextChat 传入对话 ID，为空时自动生成）

### 4.8 对话 ID

- NextChat 通过 `X-Conversation-Id` header 传递 session ID
- 后端优先使用此 ID 作为 job ID
- 为空时 fallback 为首条 user message 的 MD5 前6位
- 已修改 NextChat `app/client/api.ts`（getHeaders）和 `app/api/common.ts`（代理转发）

## 5. 非功能需求

| 需求 | 说明 |
|------|------|
| 单用户 | 无需认证鉴权，本地/内网使用 |
| 简洁设计 | 不过度工程化，不引入复杂框架 |
| 纯文件存储 | 无数据库依赖 |
| 并行处理 | 文件间和大文件内均支持 20 并发 |
| 响应速度 | 分片并行处理，查询响应取决于 LLM 调用 |

## 6. 数据管理

### 6.1 存储结构
```
jobs/
├── files.json                         # 全局文件注册表（hash+modTime+size）
└── {jobId}/
    ├── outline                        # 全局聚合大纲
    ├── files_summary.xml              # 文件摘要索引
    ├── outlines/                      # 每文件独立大纲
    │   └── {safeFileName}
    ├── sources/                       # 转换后源文本
    │   └── {safeFileName}.txt
    └── chunks/                        # 分片文件
        └── {safeFileName}_{chunkId}
```

### 6.2 files.json
```json
{
  "files": {
    "/path/to/file.go": {
      "hash": "a1b2c3d4",
      "mod_time": "2026-05-07T10:30:00Z",
      "size": 15360,
      "processed_at": "2026-05-07T10:31:00Z"
    }
  }
}
```

### 6.3 files_summary.xml
```xml
<files>
  <file path="/path/to/file.go">LLM 生成的文件摘要...</file>
</files>
```

## 7. 约束

- 本项目方向与 wiki/ 下的 CRBSA 项目无关
- 后端语言：Go
- LLM：DeepSeek v4 Flash
- 前端：NextChat（外部部署，已修改支持 X-Conversation-Id）
- 文档转换：markitdown（Go 调用 CLI 脚本）

## 8. 部署

```
1. 安装 markitdown: pip install markitdown
2. 编译: go build -o file-chat
3. 配置: set DEEPSEEK_API_KEY=xxx
4. 运行: start.bat 或 start-with-key.bat
5. 部署 NextChat，API 地址指向 http://localhost:8080
```
