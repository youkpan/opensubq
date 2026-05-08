# CRBSA 代码设计文档

> 优雅、简洁、可调试的模块化实现

---

## 目录

1. [项目结构](#1-项目结构)
2. [配置系统](#2-配置系统)
3. [核心模块设计](#3-核心模块设计)
4. [调试系统](#4-调试系统)
5. [调试与优化路线](#5-调试与优化路线)

---

## 1. 项目结构

```
crbsa/
├── __init__.py
├── config.py              # 全局配置 + 调试开关
├── debug.py               # 调试工具 (日志/可视化/profiler)
├── nn/
│   ├── __init__.py
│   ├── block_summarizer.py    # Step 1: Block 摘要提取
│   ├── codebook_router.py     # Step 2+3: Codebook 倒排索引 + Query 路由
│   ├── sparse_attention.py    # Step 4: Triton 块稀疏注意力
│   └── crbsa_layer.py         # CRBSA Attention Layer (串联以上模块)
├── kernels/
│   ├── __init__.py
│   └── block_sparse_attn.py   # Triton Kernel 实现
├── models/
│   ├── __init__.py
│   └── llama_crbsa.py         # Llama 适配层
├── trainer/
│   ├── __init__.py
│   ├── stage1_distill.py      # Stage 1: 路由器蒸馏
│   ├── stage2_sft.py          # Stage 2: 截断式稀疏微调
│   └── stage3_grpo.py         # Stage 3: GRPO 强化学习
├── eval/
│   ├── __init__.py
│   ├── niah.py                # Needle-In-A-Haystack
│   ├── ruler.py               # RULER 动态评测
│   └── swe_bench.py           # SWE-bench 评测
└── utils/
    ├── __init__.py
    ├── distributed.py         # 分布式通信 (Ulysses + P2P)
    └── profiling.py           # 性能分析工具
```

---

## 2. 配置系统

所有超参和调试开关集中管理，通过一个 dataclass 定义。

```python
# crbsa/config.py

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CRBSAConfig:
    """CRBSA 核心配置。所有参数集中在此，便于实验管理。"""

    # ── 架构参数 ──────────────────────────────────
    block_size: int = 128              # Block 大小 (tokens)
    codebook_size: int = 1024          # Codebook 聚类数 M
    route_dim: int = 64                # 路由降维维度 d_route
    topk_codebooks: int = 4            # Query 召回的 Codebook 数
    max_routed_blocks: int = 6         # 最大远距离 Block 数
    local_blocks: int = 2              # 局部保底 Block 数 (当前块+前一块)
    route_temperature: float = 1.0     # Codebook 选择温度 τ

    # ── GQA 参数 ──────────────────────────────────
    num_attention_heads: int = 32
    num_key_value_heads: int = 8       # GQA 压缩
    head_dim: int = 128

    # ── 位置编码 ──────────────────────────────────
    rope_scaling_type: str = "yarn"    # "yarn" | "ntk" | "none"
    rope_scaling_factor: float = 8.0   # YaRN x8

    # ── 调试开关 ──────────────────────────────────
    debug_enabled: bool = False             # 总开关
    debug_log_routing: bool = False         # 记录路由决策
    debug_log_block_assignment: bool = False # 记录 Block 分配
    debug_log_attention_weights: bool = False # 记录 Attention 权重
    debug_visualize_codebook: bool = False   # 可视化 Codebook 分布
    debug_profile_kernel: bool = False       # Triton Kernel profiling
    debug_check_numerics: bool = False       # NaN/Inf 检测
    debug_save_intermediates: bool = False   # 保存中间张量到磁盘
    debug_intermediate_dir: str = "./debug_outputs"

    # ── Kernel 参数 ───────────────────────────────
    kernel_backend: str = "triton"      # "triton" | "flex_attention" | "xformers"

    # ── 分布式参数 ────────────────────────────────
    sequence_parallel: bool = False
    sequence_parallel_world_size: int = 1
    async_kv_fetch: bool = False        # 异步跨卡 KV 拉取

    @property
    def total_blocks_per_query(self) -> int:
        return self.local_blocks + self.max_routed_blocks

    @property
    def tokens_per_query(self) -> int:
        return self.total_blocks_per_query * self.block_size
```

---

## 3. 核心模块设计

### 3.1 Block Summarizer

```python
# crbsa/nn/block_summarizer.py

import torch
import torch.nn as nn
from crbsa.config import CRBSAConfig
from crbsa.debug import DebugContext


class BlockSummarizer(nn.Module):
    """Step 1: 将 KV Cache 按 Block 池化为摘要向量。"""

    def __init__(self, config: CRBSAConfig):
        super().__init__()
        self.config = config
        self.pool_type = "avg"  # "avg" | "learned"

        if self.pool_type == "learned":
            # 轻量 MLP: d → d_route
            self.projection = nn.Sequential(
                nn.Linear(config.head_dim, config.route_dim, bias=False),
            )
        else:
            self.projection = nn.Linear(config.head_dim, config.route_dim, bias=False)

    def forward(self, key: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Args:
            key: [batch, num_kv_heads, seq_len, head_dim]
        Returns:
            summary: [batch, num_kv_heads, num_blocks, route_dim]
            debug_info: dict (仅 debug 模式下有内容)
        """
        cfg = self.config
        B = cfg.block_size
        debug = DebugContext(cfg)

        batch, num_heads, seq_len, head_dim = key.shape

        # Pad to block boundary
        pad_len = (B - seq_len % B) % B
        if pad_len > 0:
            key = torch.nn.functional.pad(key, (0, 0, 0, pad_len))

        # Reshape into blocks: [batch, heads, num_blocks, block_size, head_dim]
        num_blocks = key.shape[2] // B
        key_blocks = key.view(batch, num_heads, num_blocks, B, head_dim)

        # Average pooling → [batch, heads, num_blocks, head_dim]
        summary = key_blocks.mean(dim=3)

        # Project to route_dim
        summary = self.projection(summary)

        debug.log_tensor("block_summary", summary)
        debug.log_scalar("num_blocks", num_blocks)
        debug.log_scalar("pad_length", pad_len)

        return summary, debug.flush()


# crbsa/debug.py

from crbsa.config import CRBSAConfig
from typing import Any
import torch
import os
import json


class DebugContext:
    """轻量调试上下文。仅在 debug_enabled=True 时收集信息。"""

    def __init__(self, config: CRBSAConfig):
        self.config = config
        self._data: dict[str, Any] = {}

    def log_tensor(self, name: str, t: torch.Tensor):
        if not self.config.debug_enabled:
            return
        info = {"shape": list(t.shape), "dtype": str(t.dtype)}
        if self.config.debug_check_numerics:
            info["has_nan"] = bool(torch.isnan(t).any())
            info["has_inf"] = bool(torch.isinf(t).any())
            info["mean"] = t.mean().item()
            info["std"] = t.std().item()
        self._data[name] = info

        if self.config.debug_save_intermediates:
            path = os.path.join(self.config.debug_intermediate_dir, f"{name}.pt")
            torch.save(t.detach().cpu(), path)

    def log_scalar(self, name: str, value: float):
        if not self.config.debug_enabled:
            return
        self._data[name] = value

    def log_routing(self, query_ids: torch.Tensor, block_ids: torch.Tensor, scores: torch.Tensor):
        if not self.config.debug_enabled or not self.config.debug_log_routing:
            return
        self._data["routing"] = {
            "query_ids_shape": list(query_ids.shape),
            "block_ids_shape": list(block_ids.shape),
            "scores_shape": list(scores.shape),
            "top_scores": scores.topk(min(5, scores.shape[-1])).values.tolist(),
        }

    def log_codebook_stats(self, assignment: torch.Tensor, codebook: torch.Tensor):
        if not self.config.debug_enabled or not self.config.debug_log_block_assignment:
            return
        counts = assignment.bincount(minlength=codebook.shape[0])
        self._data["codebook_stats"] = {
            "assignment_entropy": self._entropy(counts.float()),
            "max_bucket_size": int(counts.max()),
            "min_bucket_size": int(counts.min()),
            "empty_buckets": int((counts == 0).sum()),
            "distribution": counts.tolist(),
        }

    def flush(self) -> dict:
        data = self._data
        self._data = {}
        return data

    @staticmethod
    def _entropy(probs: torch.Tensor) -> float:
        probs = probs / probs.sum()
        probs = probs[probs > 0]
        return float(-(probs * probs.log()).sum())
```

### 3.2 Codebook Router

```python
# crbsa/nn/codebook_router.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from crbsa.config import CRBSAConfig
from crbsa.debug import DebugContext


class CodebookRouter(nn.Module):
    """Step 2+3: Codebook 倒排索引 + Query 路由。"""

    def __init__(self, config: CRBSAConfig):
        super().__init__()
        self.config = config

        # 全局语义密码本 (可学习)
        self.codebook = nn.Parameter(
            torch.randn(config.codebook_size, config.route_dim) * 0.02
        )
        # Query 投影
        self.query_proj = nn.Linear(config.head_dim, config.route_dim, bias=False)

    def build_inverted_index(
        self, block_summary: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        Step 2: 将 Block Summary 分配到 Codebook，构建倒排索引。

        Args:
            block_summary: [batch, heads, num_blocks, route_dim]
        Returns:
            assignment: [batch, heads, num_blocks] — 每个 Block 的 Codebook ID
            debug_info: dict
        """
        cfg = self.config
        debug = DebugContext(cfg)

        # [batch, heads, num_blocks, M]
        similarity = torch.matmul(block_summary, self.codebook.T)
        assignment = similarity.argmax(dim=-1)  # [batch, heads, num_blocks]

        debug.log_tensor("codebook_assignment", assignment)
        debug.log_codebook_stats(assignment.view(-1), self.codebook)
        debug.log_routing(None, assignment, similarity)

        return assignment, debug.flush()

    def route(
        self,
        query: torch.Tensor,
        assignment: torch.Tensor,
        num_total_blocks: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Step 3: Query 路由到 Top-K Codebook，返回目标 Block IDs。

        Args:
            query: [batch, num_heads, seq_len, head_dim]
            assignment: [batch, heads, num_blocks] — Block 的 Codebook ID
            num_total_blocks: int
        Returns:
            target_block_ids: [batch, num_heads, seq_len, total_blocks]
            block_mask: [batch, num_heads, seq_len, num_total_blocks] bool
            debug_info: dict
        """
        cfg = self.config
        debug = DebugContext(cfg)

        batch, num_heads, seq_len, _ = query.shape

        # Query → route_dim
        q_route = self.query_proj(query)  # [batch, heads, seq, route_dim]

        # Query × Codebook → [batch, heads, seq, M]
        scores = torch.matmul(q_route, self.codebook.T)
        scores = scores / cfg.route_temperature
        probs = F.softmax(scores, dim=-1)

        # Top-K Codebook IDs: [batch, heads, seq, K]
        topk_scores, topk_ids = probs.topk(cfg.topk_codebooks, dim=-1)

        # 从倒排索引中提取候选 Block IDs
        # assignment: [batch, heads, num_blocks]
        # topk_ids: [batch, heads, seq, K]
        # → 构建 block_mask: [batch, heads, seq, num_total_blocks]
        target_block_ids, block_mask = self._expand_blocks(
            topk_ids, assignment, num_total_blocks, cfg
        )

        # 加入局部滑动窗口
        block_mask = self._add_local_window(block_mask, seq_len, num_total_blocks, cfg)

        # 裁剪到固定数量
        target_block_ids, block_mask = self._trim_to_fixed_count(
            target_block_ids, block_mask, cfg
        )

        debug.log_routing(topk_ids, target_block_ids, topk_scores)
        debug.log_scalar("avg_selected_blocks", block_mask.sum().item() / (batch * num_heads * seq_len))

        return target_block_ids, block_mask, debug.flush()

    @staticmethod
    def _expand_blocks(topk_ids, assignment, num_blocks, cfg):
        """将 Top-K Codebook ID 展开为具体的 Block ID。"""
        batch, heads, seq, K = topk_ids.shape

        # 构建 block_mask: 标记每个 query 应该 attend 的 block
        # assignment: [batch, heads, num_blocks], values ∈ [0, M)
        # topk_ids: [batch, heads, seq, K], values ∈ [0, M)
        block_mask = torch.zeros(batch, heads, seq, num_blocks, device=topk_ids.device, dtype=torch.bool)

        for k in range(K):
            # [batch, heads, seq, 1] vs [batch, heads, 1, num_blocks]
            match = (topk_ids[..., k : k + 1] == assignment.unsqueeze(2))
            block_mask |= match

        # 从 mask 提取 block ids
        target_block_ids = block_mask.nonzero()

        return target_block_ids, block_mask

    @staticmethod
    def _add_local_window(block_mask, seq_len, num_blocks, cfg):
        """加入局部滑动窗口: 当前 Block + 前 B 个 Block。"""
        block_size = cfg.block_size
        query_block_ids = torch.arange(seq_len, device=block_mask.device) // block_size

        for offset in range(cfg.local_blocks):
            local_ids = (query_block_ids - offset).clamp(min=0)
            # scatter local window into mask
            block_mask.scatter_(-1, local_ids.unsqueeze(-1), True)

        return block_mask

    @staticmethod
    def _trim_to_fixed_count(target_block_ids, block_mask, cfg):
        """裁剪候选集到固定数量，消除 Load Imbalance。"""
        max_blocks = cfg.total_blocks_per_query
        # 如果超出，按路由分数裁剪 (简化: 随机裁剪)
        return target_block_ids, block_mask


class LoadBalancingLoss(nn.Module):
    """防止 Codebook 索引坍塌的负载均衡损失。"""

    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha

    def forward(self, assignment: torch.Tensor, query_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            assignment: [batch * heads * num_blocks] — 每个 Block 的 Codebook ID
            query_probs: [batch * heads * seq * M] — Query 对每个 Codebook 的概率
        """
        M = query_probs.shape[-1]
        f = assignment.bincount(minlength=M).float() / assignment.numel()
        P = query_probs.mean(dim=0)
        return self.alpha * (f * P).sum()
```

### 3.3 CRBSA Layer

```python
# crbsa/nn/crbsa_layer.py

import torch
import torch.nn as nn
from crbsa.config import CRBSAConfig
from crbsa.nn.block_summarizer import BlockSummarizer
from crbsa.nn.codebook_router import CodebookRouter, LoadBalancingLoss
from crbsa.debug import DebugContext


class CRBSAAttention(nn.Module):
    """CRBSA 完整 Attention Layer，串联所有子模块。"""

    def __init__(self, config: CRBSAConfig):
        super().__init__()
        self.config = config

        self.summarizer = BlockSummarizer(config)
        self.router = CodebookRouter(config)
        self.balance_loss = LoadBalancingLoss()

        # Q/K/V/O 投影 (与标准 Transformer 相同)
        self.q_proj = nn.Linear(config.head_dim * config.num_attention_heads, config.head_dim * config.num_attention_heads, bias=False)
        self.k_proj = nn.Linear(config.head_dim * config.num_key_value_heads, config.head_dim * config.num_key_value_heads, bias=False)
        self.v_proj = nn.Linear(config.head_dim * config.num_key_value_heads, config.head_dim * config.num_key_value_heads, bias=False)
        self.o_proj = nn.Linear(config.head_dim * config.num_attention_heads, config.head_dim * config.num_attention_heads, bias=False)

        # 路由相关损失累加器
        self._routing_losses: list[torch.Tensor] = []
        self._debug_info: dict = {}

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        detach_router: bool = False,
    ) -> tuple[torch.Tensor, dict]:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_dim]
            detach_router: 是否截断路由器梯度 (Stage 2 使用)
        Returns:
            output: [batch, seq_len, hidden_dim]
            debug_info: dict (调试信息)
        """
        cfg = self.config
        debug = DebugContext(cfg)
        batch, seq_len, _ = hidden_states.shape

        # 标准投影
        q = self.q_proj(hidden_states).view(batch, seq_len, cfg.num_attention_heads, cfg.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(batch, seq_len, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(batch, seq_len, cfg.num_key_value_heads, cfg.head_dim).transpose(1, 2)

        debug.log_tensor("query", q)
        debug.log_tensor("key", k)
        debug.log_tensor("value", v)

        # Step 1: Block Summarization
        block_summary, step1_debug = self.summarizer(k)

        # Step 2: Codebook Inverted Indexing
        assignment, step2_debug = self.router.build_inverted_index(block_summary)

        num_blocks = block_summary.shape[2]

        # Step 3: Query Routing
        target_block_ids, block_mask, step3_debug = self.router.route(
            q, assignment, num_blocks
        )

        # 计算负载均衡损失
        q_route = self.router.query_proj(q)
        scores = torch.matmul(q_route, self.router.codebook.T)
        probs = torch.nn.functional.softmax(scores / cfg.route_temperature, dim=-1)

        balance_loss = self.balance_loss(
            assignment.view(-1), probs.view(-1, cfg.codebook_size)
        )
        self._routing_losses.append(balance_loss)

        # Detach: Stage 2 截断路由器梯度
        if detach_router:
            block_mask = block_mask.detach()

        # Step 4: Exact Block-Sparse Attention
        if cfg.kernel_backend == "triton":
            from crbsa.kernels.block_sparse_attn import block_sparse_attention
            attn_output = block_sparse_attention(q, k, v, block_mask, cfg.block_size)
        elif cfg.kernel_backend == "flex_attention":
            attn_output = self._flex_attention_fallback(q, k, v, block_mask)
        else:
            attn_output = self._dense_fallback(q, k, v, block_mask)

        # Output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        output = self.o_proj(attn_output)

        debug.log_tensor("attn_output", output)

        self._debug_info = {
            "step1_block_summary": step1_debug,
            "step2_inverted_index": step2_debug,
            "step3_routing": step3_debug,
            "balance_loss": balance_loss.item(),
            "sparsity_ratio": 1.0 - block_mask.float().mean().item(),
        }

        return output, debug.flush()

    def get_routing_loss(self) -> torch.Tensor:
        """获取累积的路由损失 (用于 backward)。"""
        if not self._routing_losses:
            return torch.tensor(0.0)
        loss = torch.stack(self._routing_losses).sum()
        self._routing_losses.clear()
        return loss

    @staticmethod
    def _flex_attention_fallback(q, k, v, block_mask):
        """PyTorch FlexAttention 后备方案。"""
        import torch.nn.functional as F
        # 将 block_mask 扩展到 token 级别
        B = block_mask.shape[-1]
        token_mask = block_mask.repeat_interleave(128, dim=-1)[..., :q.shape[2]]
        # 标准注意力 + mask
        scale = q.shape[-1] ** -0.5
        attn = torch.matmul(q * scale, k.transpose(-2, -1))
        attn = attn.masked_fill(~token_mask.unsqueeze(-2), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        return torch.matmul(attn, v)

    @staticmethod
    def _dense_fallback(q, k, v, block_mask=None):
        """纯 Dense 注意力 (调试/基准用)。"""
        scale = q.shape[-1] ** -0.5
        attn = torch.matmul(q * scale, k.transpose(-2, -1))
        attn = torch.nn.functional.softmax(attn, dim=-1)
        return torch.matmul(attn, v)
```

### 3.4 Triton Kernel 骨架

```python
# crbsa/kernels/block_sparse_attn.py

import torch
import triton
import triton.language as tl
from crbsa.config import CRBSAConfig


@triton.jit
def _block_sparse_attn_fwd(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    BLOCK_INDICES_ptr,
    stride_qb, stride_qh, stride_qs, stride_qd,
    stride_kb, stride_kh, stride_ks, stride_kd,
    stride_vb, stride_vh, stride_vs, stride_vd,
    stride_ob, stride_oh, stride_os, stride_od,
    BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_SELECTED_BLOCKS: tl.constexpr,
):
    """Triton Kernel: 对选定的 Block 执行精确 FlashAttention。"""
    pid = tl.program_id(0)
    head = tl.program_id(1)

    # 每个 program 处理一个 query token 的所有 selected blocks
    q_offset = pid * stride_qs
    q = tl.load(Q_ptr + q_offset + tl.arange(0, HEAD_DIM) * stride_qd)

    # 累加器
    m_i = tl.full([1], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for block_idx in range(NUM_SELECTED_BLOCKS):
        block_id = tl.load(BLOCK_INDICES_ptr + pid * NUM_SELECTED_BLOCKS + block_idx)
        if block_id < 0:
            continue

        k_start = block_id * BLOCK_SIZE
        # 加载 K block: [BLOCK_SIZE, HEAD_DIM]
        offs = k_start + tl.arange(0, BLOCK_SIZE)
        k = tl.load(K_ptr + offs[:, None] * stride_ks + tl.arange(0, HEAD_DIM)[None, :] * stride_kd)
        v = tl.load(V_ptr + offs[:, None] * stride_vs + tl.arange(0, HEAD_DIM)[None, :] * stride_vd)

        # QK^T
        scale = 1.0 / (HEAD_DIM ** 0.5)
        qk = tl.sum(q[None, :] * k, axis=1) * scale  # [BLOCK_SIZE]

        # FlashAttention 在线 Softmax
        m_new = tl.maximum(m_i, tl.max(qk))
        alpha = tl.exp(m_i - m_new)
        l_new = l_i * alpha + tl.sum(tl.exp(qk - m_new))
        acc = acc * alpha + tl.sum(tl.exp(qk - m_new)[:, None] * v, axis=0)

        m_i = m_new
        l_i = l_new

    # 写回
    out = acc / l_i
    tl.store(O_ptr + pid * stride_os + tl.arange(0, HEAD_DIM) * stride_od, out)


def block_sparse_attention(
    q: torch.Tensor,       # [batch, heads, seq, dim]
    k: torch.Tensor,       # [batch, heads, seq, dim]
    v: torch.Tensor,       # [batch, heads, seq, dim]
    block_mask: torch.Tensor,  # [batch, heads, seq, num_blocks]
    block_size: int,
) -> torch.Tensor:
    """入口函数：调用 Triton Block-Sparse Attention。"""
    batch, heads, seq, dim = q.shape
    num_blocks = k.shape[2] // block_size
    selected = block_mask.sum(dim=-1).max().item()  # 最大选中 Block 数

    output = torch.empty_like(q)

    grid = (seq, heads)
    _block_sparse_attn_fwd[grid](
        q, k, v, output, block_mask,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_SIZE=block_size,
        HEAD_DIM=dim,
        NUM_SELECTED_BLOCKS=int(selected),
    )

    return output
```

---

## 4. 调试系统

### 4.1 调试开关一览

| 开关 | 用途 | 输出内容 |
|:---|:---|:---|
| `debug_enabled` | 总开关 | 所有调试信息的使能 |
| `debug_log_routing` | 路由决策 | Top-K Codebook ID、分数分布 |
| `debug_log_block_assignment` | Block 分配 | 每个 Codebook 的 Block 数量、分布熵 |
| `debug_log_attention_weights` | Attention 权重 | 选定 Block 的 Attention Map |
| `debug_visualize_codebook` | Codebook 可视化 | t-SNE 降维图、分配热力图 |
| `debug_profile_kernel` | Kernel Profiling | Triton Kernel 执行时间、显存占用 |
| `debug_check_numerics` | 数值检查 | NaN/Inf 检测、均值/标准差 |
| `debug_save_intermediates` | 中间张量 | 保存到磁盘的 `.pt` 文件 |

### 4.2 调试输出收集器

```python
# crbsa/debug.py (续)

class DebugCollector:
    """全局调试信息收集器。跨层聚合 debug 信息。"""

    def __init__(self, config: CRBSAConfig):
        self.config = config
        self._layer_info: dict[int, dict] = {}

    def collect(self, layer_id: int, debug_info: dict):
        if not self.config.debug_enabled:
            return
        self._layer_info[layer_id] = debug_info

    def summary(self) -> str:
        if not self._layer_info:
            return "No debug info collected."

        lines = ["=== CRBSA Debug Summary ==="]
        for layer_id, info in sorted(self._layer_info.items()):
            lines.append(f"\n--- Layer {layer_id} ---")
            if "balance_loss" in info:
                lines.append(f"  Balance Loss: {info['balance_loss']:.6f}")
            if "sparsity_ratio" in info:
                lines.append(f"  Sparsity: {info['sparsity_ratio']:.2%}")
            if "codebook_stats" in info.get("step2_inverted_index", {}):
                stats = info["step2_inverted_index"]["codebook_stats"]
                lines.append(f"  Codebook Entropy: {stats['assignment_entropy']:.4f}")
                lines.append(f"  Empty Buckets: {stats['empty_buckets']}")
                lines.append(f"  Max/Min Bucket: {stats['max_bucket_size']}/{stats['min_bucket_size']}")

        return "\n".join(lines)

    def to_json(self, path: str):
        import json
        with open(path, "w") as f:
            json.dump(self._layer_info, f, indent=2, default=str)
```

### 4.3 性能 Profiler

```python
# crbsa/utils/profiling.py

import torch
import time
from contextlib import contextmanager
from crbsa.config import CRBSAConfig


class CRBSAProfiler:
    """各步骤耗时统计。"""

    def __init__(self, config: CRBSAConfig):
        self.enabled = config.debug_enabled and config.debug_profile_kernel
        self.timings: dict[str, list[float]] = {}

    @contextmanager
    def measure(self, step_name: str):
        if not self.enabled:
            yield
            return
        torch.cuda.synchronize()
        start = time.perf_counter()
        yield
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        self.timings.setdefault(step_name, []).append(elapsed)

    def report(self) -> str:
        if not self.timings:
            return "No profiling data."
        lines = ["=== CRBSA Profiling Report ==="]
        for step, times in self.timings.items():
            avg = sum(times) / len(times)
            total = sum(times)
            lines.append(f"  {step}: avg={avg*1000:.2f}ms, total={total*1000:.2f}ms, calls={len(times)}")
        return "\n".join(lines)
```

---

## 5. 调试与优化路线

### Phase 0: 基础验证 (第 1 周)

**目标**：单模块正确性验证

```
验证项:
├── Block Summarizer
│   ├── 输出 shape 正确: [batch, heads, num_blocks, d_route]
│   ├── Padding 逻辑: seq_len 非 block_size 整数倍时正确
│   └── 数值合理性: 无 NaN/Inf, 均值/方差在合理范围
│
├── Codebook Router
│   ├── 倒排索引: 每个 Block 有且仅有一个 Codebook ID
│   ├── Top-K 路由: 返回正确数量的候选 Block
│   ├── 负载均衡: 空桶数 < 10%, 最大桶不超过均值 5x
│   └── 数值稳定性: Softmax 无 overflow
│
├── Local Window
│   ├── 覆盖正确: 当前 Block + 前 N 个 Block
│   └── 边界处理: 序列开头的 Query 不越界
│
└── Triton Kernel (如果可用)
    ├── 正确性: vs Dense Attention 结果差异 < 1e-3
    ├── Shape: 处理各种 seq_len / block_size 组合
    └── 边界: 空 Block、单 Block 序列
```

**调试配置**：

```python
config = CRBSAConfig(
    debug_enabled=True,
    debug_check_numerics=True,
    debug_save_intermediates=True,
    kernel_backend="flex_attention",  # 先用 FlexAttention 验证逻辑
)
```

**预期产出**：
- 每个模块的单元测试通过
- `debug_check_numerics` 报告零异常
- 中间张量保存并可检查

### Phase 1: Kernel 性能验证 (第 2 周)

**目标**：Triton Kernel 跑通并超越 Dense

```
Benchmark 矩阵:
├── 序列长度: 4K, 8K, 32K, 128K, (256K if VRAM allows)
├── Block Size: 64, 128, 256
├── Top-K Blocks: 4, 8, 16
├── 对比基线: Dense Attention, FlexAttention fallback
└── 指标: Wall-clock time, Peak VRAM, TFLOPs utilization
```

**调试配置**：

```python
config = CRBSAConfig(
    debug_enabled=True,
    debug_profile_kernel=True,
    kernel_backend="triton",
)
```

**预期产出**：
- 128K+ 长度下 Triton Kernel 快于 Dense Attention
- 显存占用线性增长验证
- Profiling 报告显示各步骤耗时占比

### Phase 2: 路由质量验证 (第 3-4 周)

**目标**：Codebook 路由器达到 85%+ 命中率

```
验证流程:
├── Teacher 标注: 稠密模型记录高 Attention Block IDs
├── Student 预测: CRBSA 路由器输出 Top-K Block IDs
├── 命中率计算: Intersection / Teacher Ground Truth
└── 对比基线: 随机路由, LSH 路由, 固定窗口
```

**关键指标**：

| 指标 | 目标值 | 不达标时的诊断方向 |
|:---|:---|:---|
| 命中率 (Top-4) | > 85% | 增大 d_route, 增加训练数据多样性 |
| 负载均衡熵 | > 0.9 · log(M) | 增大 balance_loss alpha |
| 空桶率 | < 5% | 减小 M, 增加训练步数 |
| 索引坍塌检测 | 无 | 检查 Codebook 梯度是否正常 |

**调试配置**：

```python
config = CRBSAConfig(
    debug_enabled=True,
    debug_log_routing=True,
    debug_log_block_assignment=True,
    debug_visualize_codebook=True,
)
```

### Phase 3: 端到端训练验证 (第 5-8 周)

**目标**：三阶段训练 Pipeline 跑通

```
Stage 1 → Stage 2 → Stage 3
   │          │          │
   ▼          ▼          ▼
路由器      主模型     RL 对齐
命中率      PPL       NIAH/RULER
>85%       收敛      分数达标
```

**Stage 1 失败诊断**：

| 现象 | 可能原因 | 解决方向 |
|:---|:---|:---|
| 命中率 < 60% | Codebook 初始化差 | k-means 初始化 / 增大 M |
| 索引坍塌 (空桶 > 30%) | balance_loss 太小 | 增大 alpha, 检查梯度 |
| 训练震荡 | 学习率过大 | 降低 LR, 加 warmup |

**Stage 2 失败诊断**：

| 现象 | 可能原因 | 解决方向 |
|:---|:---|:---|
| PPL 不收敛 | 路由器命中率太低 | 回退 Stage 1, 提高训练质量 |
| 路由退化 | Detach 未生效 | 检查梯度流, 确认 block_mask.detach() |
| 模型只看局部 | 局部窗口权重过高 | 减小 local_blocks, 增大 max_routed_blocks |

**Stage 3 失败诊断**：

| 现象 | 可能原因 | 解决方向 |
|:---|:---|:---|
| 奖励不提升 | 任务太难 | 简化 NIAH → 单针, 缩短距离 |
| 模型"偷懒" | R_correct 奖励太高 | 增大 R_routing, 引入 R_hallucination |
| RL 不稳定 | GRPO 超参不当 | 调整 KL 系数, 降低学习率 |

### Phase 4: 分布式扩展 (第 9-12 周)

**目标**：多卡 1M Token 端到端跑通

```
优化顺序:
1. 单卡验证 (128K) ─── 确保逻辑正确
2. Ulysses 序列并行 ── All-Gather Block Summary
3. P2P 异步 KV 拉取 ── 计算通信掩盖
4. PagedAttention ──── 推理集成
```

**性能优化 Checklist**：

```
□ Block Summary All-Gather < 5% 总时间
□ P2P KV 拉取与本地计算 Overlap > 70%
□ 单卡显存峰值 < 80GB (1M tokens, 7B 模型)
□ 多卡线性加速比 > 0.85
□ End-to-end prefill 1M tokens < 30s (8x H100)
```

### 优化路线总览

```
Week 1-2:  Phase 0 + 1  ── 模块验证 + Kernel Benchmark
Week 3-4:  Phase 2      ── 路由质量达标
Week 5-8:  Phase 3      ── 三阶段训练
Week 9-12: Phase 4      ── 分布式 + 1M 跑通

             ↓ 产出物 ↓
Week 2:  单模块单元测试 + Kernel 性能报告
Week 4:  路由命中率报告 + Codebook 可视化
Week 8:  NIAH/RULER 评测结果
Week 12: 1M Token 端到端 Demo
```
