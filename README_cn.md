# SubQ

[English](./README.md)

长上下文与智能问答项目集合。

---

## 子项目

### 1. File-Chat — 长文档智能问答系统

[![Go 1.22+](https://img.shields.io/badge/Go-1.22+-00ADD8.svg)](https://go.dev/)
[![OpenAI Compatible](https://img.shields.io/badge/API-OpenAI_Compatible-green.svg)]()

基于 Go 的长文本智能问答系统，采用 **Context Engineering** 技术结合 DeepSeek LLM。提供 OpenAI 兼容 API，可与 NextChat 等前端无缝集成。

**核心特性：**
- LLM 语义分片 — 自动将长文档拆分为有意义的片段并生成摘要
- 两级检索：文件摘要 → 片段大纲 → Top-K 片段
- `@路径` 引用指定文件，`@全部`/`@all` 搜索所有已索引文件
- 全局文件索引 — 所有对话共享全局大纲和摘要，`@all` 可跨对话检索
- 文件变更检测（size → modTime → MD5 hash），避免重复处理
- 文件锁机制 — 并发处理同一文件时加锁，避免数据冲突
- 并行处理：20 并发 goroutine，每 worker 处理 30KB
- 通过 OpenAI 兼容的 `/v1/chat/completions` API 提供 SSE 流式输出
- 支持 PDF、Excel、Word 等，通过 markitdown 转换

**架构：**
```
NextChat ──HTTP/SSE──▶ file-chat (Go) ──HTTP/SSE──▶ DeepSeek API
                            │
                            ├── 全局文件索引（跨对话共享）
                            ├── LLM 语义分片（20 并发）
                            ├── 基于 hash 的文件存储（去重）
                            └── markitdown 文档转换
```

**数据存储：**
```
data/
├── files.json                    # 全局文件注册表
├── files/{hash[:2]}/{hash[2:4]}/{name}/
│   ├── outline                   # 每文件大纲
│   ├── source                    # 转换后文本
│   └── chunks/                   # 分片文件
├── chats/{conversationID}/
│   └── chat-files.json           # 对话关联文件列表
└── global/
    ├── global_outline            # 全局大纲（所有文件）
    └── global_files_summary.xml  # 全局文件摘要
```

**快速开始：**
```bash
# 安装 markitdown
pip install markitdown

# 编译
cd file-chat && go build -o file-chat

# 配置 API Key 并运行
set DEEPSEEK_API_KEY=your-key-here
file-chat.exe

# 部署 NextChat，API 地址设为 http://localhost:8880
```

详见 [技术架构](./wiki2/Architecture-file-chat.md) 和 [PRD](./wiki2/PRD-file-chat.md)。

**使用方法（NextChat）：**
1. 启动后端：双击 `file-chat/start-with-key-pan.bat`
2. 启动前端：双击 `scripts/start-nextchat.bat`
3. 打开 http://localhost:3000，进入设置：
   - 自定义接口地址：`http://localhost:8880`
   - API Key：填写你的 DeepSeek API Key
   - 模型名称：`deepseek-v4-flash`，模型提供商选择 **deepseek-v4-flash**
4. 新建对话，输入 `@文件绝对路径\文件名` + 你的提示词，发送即可

**截图：**

![截图](./ScreenShot.png)

---

### 2. CRBSA — 基于密码本路由的块稀疏注意力

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![Triton](https://img.shields.io/badge/Kernel-Triton-lightgrey)]()

**CRBSA (Codebook-Routed Block-Sparse Attention)** 是一种面向 1M~10M Token 超长上下文的注意力架构。通过引入固定大小的全局语义密码本，将路由复杂度从 $O(N^2)$ 降至 $O(M)$（$M=1024$，常数），选中 Block 后执行**精确** FlashAttention —— 零信息模糊，无 RNN 隐状态，无近似。

## 为什么选择 CRBSA

| | 稠密注意力 | DeepSeek V4 | 线性 RNN | **CRBSA** |
|:---|:---|:---|:---|:---|
| **复杂度** | $O(N^2)$ | $O((N/m)^2)$ | $O(N)$ | **$O(N)$** |
| **1M Prefill 加速** | 1× | ~10× | ~8× | **>50×** |
| **近处偏差** | 无 | 低 | 高 | **无** |
| **多跳检索** | 极强 | 强 | 弱 | **极强** |

## 架构

```
输入 ──▶ Q/K/V 投影 ──▶ Block 摘要 ──▶ 密码本倒排索引
                                    │              │
                                    │    Query × 密码本 (O(M))
                                    │              │
                                    ▼              ▼
                              局部滑动窗口  +  路由召回 Block
                                             │
                                  Triton 块稀疏 FlashAttention
                                             │
                                           输出
```

四步并行计算流：
1. **Block 摘要** — 按 $B=128$ 切块，Key 平均池化为摘要向量
2. **密码本索引** — 将每个 Block 分配到 $M=1024$ 个可学习语义聚类中心之一
3. **$O(1)$ Query 路由** — Query 与密码本打分，取 Top-$K$ 聚类 → 召回 Block IDs
4. **精确稀疏注意力** — Triton/Flex/Dense 算子仅对选中 Block 执行 FlashAttention

## 快速开始

### 安装

```bash
git clone https://github.com/your-org/CRBSA.git
cd CRBSA
pip install -e .
```

### 验证模块

```bash
# 不加载模型，纯模块测试
python scripts/verify.py --seq-len 2048 --debug

# 完整模型测试（需要 GPU）
python scripts/verify.py --model Qwen/Qwen3.6-35B-A3B --seq-len 4096 --debug
```

### 推理

```python
import torch
from crbsa.config import CRBSAConfig
from crbsa.models import apply_crbsa_to_qwen3

# 配置
config = CRBSAConfig.from_pretrained("Qwen/Qwen3.6-35B-A3B")
config.block_size = 128
config.codebook_size = 1024
config.max_routed_blocks = 6

# 加载 CRBSA 模型
model = apply_crbsa_to_qwen3("Qwen/Qwen3.6-35B-A3B", config)
model.eval()

# 长上下文前向传播
input_ids = torch.randint(0, config.vocab_size, (1, 100000)).cuda()
result = model(input_ids=input_ids)
print(result["logits"].shape)
```

### 调试模式

所有调试开关集中在 `CRBSAConfig` 中，关闭时零开销。

```python
config = CRBSAConfig(
    debug_enabled=True,               # 总开关
    debug_log_routing=True,           # Top-K 密码本 ID、分数
    debug_log_block_assignment=True,  # 密码本分布、熵
    debug_check_numerics=True,        # NaN/Inf 检测
    debug_profile_kernel=True,        # 每步计时
    debug_save_intermediates=True,    # 保存中间张量到磁盘
)
```

## 训练流程

三阶段训练策略，避免索引坍塌（`scripts/train/`）：

**Stage 1 — 路由器蒸馏** (`stage1_distill.py`)

冻结主干，用稠密注意力 Ground Truth 训练密码本路由器。

```bash
python scripts/train/stage1_distill.py \
    --model Qwen/Qwen3.6-35B-A3B \
    --seq-len 131072 --epochs 3 --debug
```

**Stage 2 — 截断式稀疏微调** (`stage2_sft.py`)

全参解冻。路由稀疏前向，但**截断**路由器梯度与 LM Loss 的连接。

```bash
python scripts/train/stage2_sft.py \
    --model Qwen/Qwen3.6-35B-A3B \
    --stage1-dir outputs/stage1 --epochs 2 --debug
```

**Stage 3 — GRPO 强化学习** (`stage3_grpo.py`)

在 NIAH/SWE-bench 任务上用 RL 奖励长程检索行为。

```bash
python scripts/train/stage3_grpo.py \
    --model Qwen/Qwen3.6-35B-A3B --debug
```

一键运行：

```bash
bash scripts/train/run_all.sh
```

## 评测

```bash
# NIAH：不同长度和深度的大海捞针测试
python scripts/eval/eval_niah.py --model Qwen/Qwen3.6-35B-A3B --debug

# 路由命中率：对比稠密注意力 Ground Truth
python scripts/eval/eval_routing.py --seq-len 32768 --debug

# Kernel 性能：CRBSA vs Dense 在不同序列长度下的对比
python scripts/eval/benchmark_kernel.py --seq-lengths 2048 4096 8192 16384
```

## 项目结构

```
crbsa/
├── config.py                  # CRBSAConfig — 所有超参 + 调试开关
├── debug.py                   # DebugContext / DebugCollector / CRBSAProfiler
├── nn/
│   ├── block_summarizer.py    # Step 1: Block 摘要 (池化 → 投影)
│   ├── codebook_router.py     # Step 2+3: 倒排索引 + Query 路由
│   ├── sparse_attention.py    # Step 4: Triton / Flex / Dense 三后端
│   └── crbsa_layer.py         # 完整 Layer: 4 步流水线 + RoPE + Detach
├── kernels/
│   └── block_sparse_attn.py   # Triton Kernel + fallback 实现
├── models/
│   └── qwen_crbsa.py          # Qwen3.6-35B-A3B (MoE) 适配器
├── utils/
│   ├── distributed.py         # Ulysses 序列并行 + P2P KV 拉取
│   └── profiling.py           # BenchmarkResult + 显存测量
scripts/
├── verify.py                  # 快速模块 + 模型验证
├── train/                     # 三阶段训练脚本 + run_all.sh
└── eval/                      # NIAH、路由命中率、Kernel 性能评测
wiki/
├── Architecture.md            # 完整架构文档
└── Code-Design.md             # 代码设计 + 调试优化路线
```

## 支持模型

CRBSA 面向 **Qwen3.6-35B-A3B**（MoE 架构，35B 总参 / 3B 激活）设计和测试。适配器保留 MoE MLP 层，仅替换 Attention 层。

架构通用，可适配任何带 GQA 的 Transformer 模型。

## 核心参数

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `block_size` | 128 | 每 Block Token 数（匹配 Triton 最优吞吐） |
| `codebook_size` | 1024 | 语义聚类中心数 |
| `route_dim` | 64 | 路由投影维度 |
| `topk_codebooks` | 4 | 每个 Query 召回的密码本聚类数 |
| `max_routed_blocks` | 6 | 每个 Query 最大远距离 Block 数 |
| `local_blocks` | 2 | 局部滑动窗口 Block 数 |
| `route_temperature` | 1.0 | 密码本选择的 Softmax 温度 |

## 许可

MIT
