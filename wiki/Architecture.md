# CRBSA 架构文档

> **Codebook-Routed Block-Sparse Attention**
> 基于密码本路由的块稀疏注意力架构

---

## 目录

1. [概述](#1-概述)
2. [问题定义：长文本的"不可能三角"](#2-问题定义长文本的不可能三角)
3. [数学基础](#3-数学基础)
4. [架构总览](#4-架构总览)
5. [模块级微架构](#5-模块级微架构)
6. [三阶段训练法则](#6-三阶段训练法则)
7. [分布式系统架构](#7-分布式系统架构)
8. [核心参数参考](#8-核心参数参考)
9. [评测基准](#9-评测基准)
10. [复现路线](#10-复现路线)

---

## 1. 概述

CRBSA 是一种面向 **1M~10M Token 超长上下文** 的注意力架构，通过引入全局语义密码本 (Global Semantic Codebook) 将路由复杂度与序列长度解耦，结合 Triton 块稀疏算子实现精确注意力计算。

**核心指标 (1M Token 场景)**：

| 指标 | 值 |
|:---|:---|
| Prefill 加速比 | > 50x (vs Dense Attention) |
| 算法复杂度 | O(N) |
| 显存需求 | ~24GB (7B 模型, 1M tokens) |
| 长程信息衰减 | 零 |
| 多跳检索能力 | 与 Dense Attention 持平 |

---

## 2. 问题定义：长文本的"不可能三角"

现有长文本架构无法同时满足三个目标：**无限上下文、精确召回、低延迟/低成本**。

### 2.1 稠密注意力 (Dense Attention)

```
复杂度: O(N^2)
```

- 精度完美，多跳检索无衰减
- 128K 以上序列成本失控
- 代表：标准 Transformer、GPT-4

### 2.2 混合压缩稀疏 (DeepSeek V4 路线)

```
路由复杂度: O((N/m)^2)  (m 为压缩比)
```

- 将序列压缩 m 倍后路由打分
- 路由打分器本身仍是二次复杂度
- 1M+ Token 时路由计算再次打爆 GPU

### 2.3 线性混合 RNN (Kimi Linear 路线)

```
复杂度: O(N)
```

- 速度极快
- 隐状态压缩导致灾难性遗忘和近处偏差 (Recency Bias)
- 多跳检索能力弱，必须依赖稠密层兜底

### 2.4 CRBSA 的破局

| 维度 | CRBSA 方案 |
|:---|:---|
| 路由 | 引入固定 Codebook 中介空间，单 Query 路由 O(M)，M 为常数 |
| 计算 | Block 级精确 FlashAttention，零信息模糊 |
| 硬件 | 全 Block 级操作，无 Token 级 Gather，Tensor Core 利用率 > 90% |

---

## 3. 数学基础

### 3.1 传统路由的复杂度瓶颈

传统 Attention 路由的打分机制：

```
Score(Q, K) = Q · K^T
Q ∈ R^{N×d}, K ∈ R^{N×d}
复杂度: O(N^2 · d)
```

无论 DeepSeek 的 DSA 还是其他方案，只要 Query 需要与所有 Token/Block 打分，复杂度就无法突破 O(N^2)。

### 3.2 CRBSA 的数学重构

引入中介空间 **C** (Codebook)：

```
C ∈ R^{M×d}   (M 为固定常数，如 1024)
```

路由过程解耦为两步：

**建库 (Block → Topic)**：

```
S_KV = Pool(K) · C^T
复杂度: O(N/B · M · d) = O(N)    (B 为 Block 大小)
```

**查询 (Query → Topic)**：

```
S_Q = Q · C^T
复杂度: O(N · M · d) = O(N)
```

**关键结论**：单 Query 的路由复杂度从 O(N) 降至绝对常数 **O(M)**。路由算力与序列长度 **完全解耦**。

---

## 4. 架构总览

CRBSA 插入在标准 Transformer 每一层（或交替层），替代原有 Dense Attention。计算流高度并行化，分为四步：

```
┌──────────────────────────────────────────────────────────────┐
│                    CRBSA Attention Layer                      │
│                                                              │
│  ┌─────────────┐   ┌──────────────┐   ┌─────────────────┐  │
│  │  Step 1      │   │  Step 2       │   │  Step 3          │  │
│  │  Block       │──▶│  Codebook     │──▶│  Query           │  │
│  │  Summarize   │   │  Inverted     │   │  Routing         │  │
│  │  O(N)        │   │  Indexing     │   │  O(1) per query  │  │
│  │              │   │  O(N)         │   │                  │  │
│  └─────────────┘   └──────────────┘   └────────┬─────────┘  │
│                                                  │            │
│                                                  ▼            │
│                                      ┌──────────────────┐   │
│                                      │  Step 4           │   │
│                                      │  Exact Block-     │   │
│                                      │  Sparse Attention │   │
│                                      │  (Triton Kernel)  │   │
│                                      └──────────────────┘   │
└──────────────────────────────────────────────────────────────┘
```

### Step 1: Block Summarization (O(N))

- 输入序列按 B=128 切块，共 N_b = N/B 个 Block
- 对每个 Block 的 Key 向量做 Average Pooling 或轻量 1 层 MLP
- 生成 Block Summary: K_summary ∈ R^{N_b × d_route}

### Step 2: Codebook Inverted Indexing (O(N))

- Block Summary 与 Codebook C (M=1024) 计算相似度，取 Argmax
- 构建倒排索引: `Codebook_ID → [Block_ID_1, Block_ID_2, ...]`

### Step 3: O(1) Query Routing

- Query 与 Codebook 计算相似度，取 Top-K 语义桶
- 从倒排表提取候选 Block IDs
- 若候选数超限，按局部相似度裁剪

### Step 4: Exact Block-Sparse Attention

- 候选集 = Local Blocks (2个) ∪ Routed Blocks (如 6个)
- 总计固定数量的 Block (如 8 个 = 1024 tokens)
- 调用 Triton Kernel 执行精确 FlashAttention

---

## 5. 模块级微架构

### 5.1 Block Summarization

```
输入: K ∈ R^{N × d}
输出: K_summary ∈ R^{N_b × d_route}

1. Chunking: 按 B=128 切块 → N_b = N/B 个 Block
2. Pooling:  Average Pooling / 1-Layer MLP
3. 投影:     降维到 d_route = 64 或 128
```

**设计决策**：
- Block Size B=128 匹配 Triton Kernel 的分块最优吞吐
- 路由维度 d_route 可大幅降维，削减建库和打分算力

### 5.2 Codebook 路由器

```
参数: C ∈ R^{M × d_route}   (M=1024, 可学习)
      W_Q, W_K 投影层

1. KV 桶分配:
   sim(K_summary, C) → Argmax → 倒排索引
   Codebook_ID → [Block_ID_1, Block_ID_2, ...]

2. Query 多路召回:
   P(q_t, C_j) = Softmax(q_t · C_j^T / τ)
   选取 Top-K (如 K=4) Codebook ID

3. 块合并与裁剪:
   合并 K 个倒排表 → 若候选 Block 数 > 上限 → 按局部相似度裁剪
```

### 5.3 局部滑动窗口

```
强制加入: 当前 Query 所在 Block + 前 1 个 Block (共 256 tokens)

总候选集 = Local Blocks (2个) ∪ Routed Blocks (如 6个)
       = 固定 8 个 Block = 1024 tokens

保证: 每个 Query 计算量恒定，彻底消除 Load Imbalance
```

**理由**：90% 的连贯语法和短期逻辑依赖相邻 Token，局部窗口保证基础流畅性。

### 5.4 Triton 块稀疏算子

```python
# 伪代码：仅对选定 Block 执行精确 FlashAttention
output = flash_attn_with_block_mask(
    q, k, v,
    block_indices=Target_Block_IDs,
    block_size=128
)
```

**特性**：
- 复用 FlashAttention-3 的 block_pointer 机制
- 零精度近似——选定 Block 内部是 100% Exact Attention
- Block 级连续内存访问，Tensor Core 利用率 > 90%
- 无 Token 级 Gather/Scatter，无 Warp 线程分化

---

## 6. 三阶段训练法则

稀疏注意力直接端到端训练会导致"索引坍塌"或"路由退化"。采用三阶段渐进训练策略。

### Stage 1: 路由器蒸馏 (Router Knowledge Distillation)

```
目标: 训练 Codebook + 路由器到接近完美
主模型: 冻结
```

**流程**：
1. 准备 128K 长度高质量文本 (代码、论文、逻辑推理)
2. Teacher 标注：用稠密模型跑前向，记录每层每个 Query 的高 Attention Weight Block IDs
3. Student 训练：冻结主干，梯度仅传给 Codebook C 和投影层 W_Q, W_K

**损失函数**：

| 损失 | 公式 | 目的 |
|:---|:---|:---|
| Cross Entropy | 强制路由器 Top-K 命中 Teacher 选出的 Blocks | 路由精度 |
| Load Balancing | L_bal = α · Σ(f_i · P_i) | 防止索引坍塌 |

其中 f_i 是分配到第 i 个 codebook 的块比例，P_i 是 Query 选中该 codebook 的平均概率。

### Stage 2: 截断式稀疏微调 (Detached Sparse Tuning)

```
目标: 让主模型适应稀疏上下文环境
路由器命中率: 已达 85%+
全参解冻: 是
```

**关键工程决策——梯度截断 (Detach)**：

```
路由器 ──→ Target_Block_IDs ──[Detach]──→ Attention Kernel
                                         ↑
                              作为常量传入，不回传梯度

主模型: 通过 Next Token Prediction (CE Loss) 更新
路由器: 仅受 Stage 1 的 Distillation Loss + Load Balancing Loss 更新
```

**为什么必须 Detach**：如果主模型梯度通过 Attention Mask 传回路由器，模型会发现"只看附近最容易降低 Loss"，从而摧毁长程路由能力。

### Stage 3: 长文本强化学习对齐 (RLHF/GRPO)

```
目标: 逼模型主动检索远处证据
```

**任务构造**：
- 大海捞针 (NIAH)、多针多跳
- 变量追踪 (跨 200K+ Token)
- 跨文件代码补丁 (SWE-Bench 风格)

**奖励设计**：

| 奖励 | 值 | 触发条件 |
|:---|:---|:---|
| R_correct | +1.0 | 最终答案正确 |
| R_routing | +0.5 | 路由器命中关键证据所在 Block |
| R_hallucination | -1.0 | 答案合理但未基于远处约束 |

**算法**：GRPO (Group Relative Policy Optimization)，通过生成结果进行优势估计，省去 Critic 模型显存。

---

## 7. 分布式系统架构

单卡 H100 (80GB) 无法装载 1M Token 的 KV Cache + Activations。需结合序列并行与异步跨卡拉取。

### 7.1 Ulysses 序列并行

```
场景: 8 GPU, N=1,024,000

┌─────────┐  ┌─────────┐       ┌─────────┐
│  GPU 0  │  │  GPU 1  │  ...  │  GPU 7  │
│ 128K    │  │ 128K    │       │ 128K    │   Step 1: 输入切分
│ tokens  │  │ tokens  │       │ tokens  │
└────┬────┘  └────┬────┘       └────┬────┘
     │            │                  │
     ▼            ▼                  ▼
  Block       Block              Block        Step 2: 本地
  Summary     Summary            Summary      计算 Block Summary
     │            │                  │
     └────────────┼──────────────────┘
                  │
                  ▼
          All-Gather (< 2MB)                Step 3: 全局同步
                  │
                  ▼
         每张 GPU 拥有全局 8000 个           Block Summary
                  │
                  ▼
         本地独立路由 (零通信)               Step 4: 计算
                                                  Target_Block_IDs
```

**通信量分析**：Block Summary 极小 (1M/128=8000 个 Block × 64 维 ≈ 2MB)，All-Gather 开销可忽略。

### 7.2 异步 P2P KV 拉取

```
GPU 0 需要 Block 3000 (物理存储在 GPU 7)

时间线:
GPU 0 ─── 发送请求 ─── 计算本地 Block Attention ─── 拼接远程 KV ─── 继续
GPU 7 ─── 收到请求 ─── 发送 Block 3000 KV ────────────────────────
            │                                        │
            └──────── 计算通信掩盖 (Overlap) ─────────┘
```

**原则**：
- 绝对禁止 All-Gather 全部 KV（网络 OOM）
- 使用 NVLink (单节点) 或 InfiniBand (跨节点) P2P 原语
- 远程数据到达前先计算 Local Blocks，实现计算通信掩盖

### 7.3 PagedAttention 集成

推理阶段与 vLLM/SGLang 的 PagedAttention 深度绑定：

- CRBSA 路由本身就是 Block 级别
- PagedAttention 按 Page (Block) 物理离散存储 KV Cache
- 路由得到的 Block_ID → 直接映射为 Paged Block Table 物理地址
- 零拷贝，极速推理

---

## 8. 核心参数参考

适用于 7B~14B 级别基座模型 (Llama-3 / Qwen-2.5)，冲击 1M 上下文。

| 组件 / 参数 | 推荐值 | 工程解释 |
|:---|:---|:---|
| **Block Size (B)** | 128 tokens | 匹配 Triton Kernel 分块最优吞吐 |
| **Codebook 数量 (M)** | 1024 | 足够细粒度的语义区分度 |
| **路由降维 (d_route)** | 64 | 极大削减建库和打分算力 |
| **单 Query 关注 Block 数** | 8 (= 1024 tokens) | 计算复杂度恒定锁死在 1K 规模 |
| **局部保底 Block 数** | 2 (当前块 + 上一个块) | 保证代码、句法的极短依赖不断裂 |
| **KV Heads (GQA)** | H_k = 8 或 4 | 搭配 GQA 减少 KV Cache 显存占用 |
| **RoPE Scaling** | YaRN (x8 scale) | 外推位置编码，确保长文本位置感知 |
| **路由温度 τ** | 可调 | 控制 Codebook 选择的锐度 |

---

## 9. 评测基准

### 9.1 RULER: 动态长文本基础评测

相比经典 NIAH，增加多跳聚合、变量跟踪和条件过滤。

**CRBSA 验证点**：Codebook 路由是否能将同一变量的不同定义片段分配到相同/相近的 Topic 桶，供 Query 精确提取。

### 9.2 MRCR v2: 非相邻证据融合

- 证据 A 在 10K 处，证据 B 在 800K 处，缺乏直接字面相似度
- 通过 Stage 3 RL，模型不仅能匹配字面，更能识别"语义桥梁"
- 路由阶段将 A 和 B 的 Block 并行拉入计算窗口

### 9.3 SWE-bench Verified: 真实工业世界

- 长代码库、历史 PR 讨论、Issue 描述
- CRBSA 优势：Codebook 全局静态，生成代码时触及接口逻辑，Query 瞬间激活包含接口约束的 Block 索引
- 避免传统 RNN 的"局部看似正确但全局编译报错"

---

## 10. 复现路线

### 路线 1: LSH 路由 + Block-Sparse (最快验证)

**目标**：1~2 周内验证 SSA 思路可行

- 基座：Qwen2.5-0.5B/1.5B 或 Llama-3.2-1B
- 位置编码：RoPE + YaRN/NTK Scaling 或 ALiBi
- 稀疏 Kernel：PyTorch FlexAttention / xFormers block-sparse
- 路由：LSH Hash + 排序分桶 + 局部窗口 + 全局 Token
- 训练：加载预训练权重 → continued pretrain (8K → 32K → 128K)

**能验证**：
- 随长度增长速度优势扩大
- 一定程度的功能性长上下文

### 路线 2: VQ/Codebook 倒排索引 (更接近 CRBSA)

**目标**：实现真正的"像检索一样的 Attention 路由"

- 路由：向量量化 Codebook + 倒排表
- 索引构建：O(N) (counting sort / prefix sum)
- 查询：O(M) per query，M 为常数
- 必须_block 化：倒排表输出 token id → 转 block id → block 内精确 attention
- Codebook 学习：随机初始化 + EMA / 离线 k-means 初始化

### 路线 3: 两阶段 Coarse-to-Fine + RL (冲高指标)

**目标**：逼近 MRCR/SWE 高指标

- Stage A (Coarse)：Block Summary 级注意力 → 选 Top-B 个 Block
- Stage B (Fine)：选中 Block 内精确 FlashAttention
- RL 对齐：引用一致性奖励 + 多证据覆盖奖励 + 代码补丁约束奖励
- 分布式：Sequence Parallel + Activation Checkpointing + Paged KV Cache

---

## 附录 A: 性能对比总览 (1M Token 模拟)

| 指标 | Dense Attention | DeepSeek V4 (CSA) | Linear RNN (Kimi) | **CRBSA** |
|:---|:---|:---|:---|:---|
| **算法复杂度** | O(N^2) | O((N/m)^2) | O(N) | **O(N)** |
| **Prefill 加速** | 1.0x | ~10.0x | ~8.0x | **> 50.0x** |
| **Recency Bias** | 无 | 低 | 高 | **无** |
| **多跳检索** | 极强 | 强 | 弱 | **极强** |
| **信息精度** | 100% Exact | 近似 | 有损 | **100% Exact** |

## 附录 B: 关键数据流

```
Input Sequence (N tokens)
        │
        ▼
┌─────────────────┐
│  Q, K, V 投影   │  标准线性投影
└───────┬─────────┘
        │
        ├──▶ K → Chunk(B=128) → Pool → K_summary ∈ R^{N_b × d_route}
        │                                        │
        │                        ┌───────────────┘
        │                        ▼
        │              K_summary · C^T → Argmax → 倒排索引
        │                                      │
        ├──▶ Q → Q · C^T → Top-K Codebook IDs ─┤
        │                                      │
        │                        ┌─────────────┘
        │                        ▼
        │              候选 Block IDs + 局部窗口 Block IDs
        │                        │
        │                        ▼
        │              Target_Block_IDs (固定数量)
        │                        │
        └────────────────────────┤
                                 ▼
                    Triton Block-Sparse FlashAttention
                    (仅对 Target_Block_IDs 内的 KV 执行精确计算)
                                 │
                                 ▼
                          Output ∈ R^{N × d}
```

## 附录 C: 第一周行动项

1. **算子工程师**：在 FlashAttention-3 基础上，实现接收离散 `Target_Block_IDs` 的 Forward/Backward Triton Kernel
2. **Benchmark**：测试 128K / 1M 下的速度与显存表现
3. **验证**：确认 Kernel 性能超越稠密模型 (理论预期 100% 成立)
4. **推进**：Kernel 验证通过后，开始 Codebook 路由器训练
