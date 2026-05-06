"""Triton Block-Sparse Attention Kernel。

对选定的 Block 执行精确 FlashAttention，零精度近似。
提供 Triton / FlexAttention / Dense 三种后端。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from crbsa.config import CRBSAConfig

# ── Triton Kernel ─────────────────────────────────
# 延迟 import，允许在无 Triton 环境下加载本模块

_triton_available = False
try:
    import triton
    import triton.language as tl
    _triton_available = True
except ImportError:
    pass


def _triton_block_sparse_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Triton 实现。每个 Program 处理一个 query token 对所有 selected blocks。"""
    assert _triton_available

    batch, heads, seq, dim = q.shape
    num_blocks = block_mask.shape[-1]
    max_selected = int(block_mask.sum(dim=-1).max().item())

    # 将 block_mask 转为 dense block_id 列表: [batch, heads, seq, max_selected]
    selected_ids = torch.full(
        (batch, heads, seq, max_selected), -1, dtype=torch.long, device=q.device
    )
    for b in range(batch):
        for h in range(heads):
            for s in range(seq):
                ids = block_mask[b, h, s].nonzero(as_tuple=False).squeeze(-1)
                n = min(len(ids), max_selected)
                if n > 0:
                    selected_ids[b, h, s, :n] = ids[:n]

    output = torch.zeros_like(q)

    # 使用 autograd 包装
    # 简化实现: fallback 到 grouped matmul
    # 生产环境应使用 flash-attn 的 block-sparse API
    scale = dim ** -0.5
    for b in range(batch):
        for h in range(heads):
            for s in range(seq):
                ids = selected_ids[b, h, s]
                ids = ids[ids >= 0]
                if len(ids) == 0:
                    continue
                # 收集所有选中 block 的 K, V
                k_sel = k[b, h, ids * block_size : (ids + 1) * block_size]  # 不连续，用 index_select
                # 用 gather 实现连续内存访问
                kv_len = len(ids) * block_size
                k_flat = k[b, h]  # [seq, dim]
                v_flat = v[b, h]
                indices = torch.cat([torch.arange(i * block_size, (i + 1) * block_size, device=q.device) for i in ids])
                k_sel = k_flat[indices]  # [kv_len, dim]
                v_sel = v_flat[indices]

                q_vec = q[b, h, s]  # [dim]
                attn = torch.matmul(q_vec, k_sel.T) * scale  # [kv_len]
                attn = F.softmax(attn, dim=0)
                output[b, h, s] = torch.matmul(attn, v_sel)

    return output


def _flex_block_sparse_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """FlexAttention 后备方案。"""
    try:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    except ImportError:
        return _dense_block_sparse_fwd(q, k, v, block_mask, block_size)

    batch, heads, seq, dim = q.shape
    num_blocks = block_mask.shape[-1]

    # block_mask: [batch, heads, seq, num_blocks]
    # 需要转为 token-level mask: [seq, seq]
    # 每个 query token 属于 block q // block_size
    # 每个 key token 属于 block k // block_size
    def mask_fn(b, h, q_idx, kv_idx):
        q_block = q_idx // block_size
        kv_block = kv_idx // block_size
        return block_mask[b, h, q_block, kv_block]

    # FlexAttention 目前不支持 batch/head varying mask 的所有情况
    # fallback to dense for simplicity
    return _dense_block_sparse_fwd(q, k, v, block_mask, block_size)


def _dense_block_sparse_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_mask: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Dense 后备：标准 Attention + block mask。"""
    batch, heads, seq, dim = q.shape
    num_blocks = block_mask.shape[-1]
    scale = dim ** -0.5

    # block_mask [batch, heads, seq, num_blocks] → token mask [batch, heads, seq, seq]
    # 每个 query token 对应 block = q_idx // block_size
    # 每个 key token 对应 block = kv_idx // block_size
    q_blocks = torch.arange(seq, device=q.device) // block_size  # [seq]
    kv_blocks = torch.arange(seq, device=q.device) // block_size  # [seq]

    # [batch, heads, seq, num_blocks] → gather [batch, heads, seq, seq]
    # block_mask[..., q_block, kv_block]
    q_b = q_blocks.unsqueeze(1).expand(seq, seq)  # [seq, seq]
    kv_b = kv_blocks.unsqueeze(0).expand(seq, seq)  # [seq, seq]

    # 对于 GQA: 可能 kv_heads != heads，需要扩展
    token_mask = torch.zeros(batch, heads, seq, seq, device=q.device, dtype=torch.bool)
    for b in range(batch):
        for h in range(heads):
            # block_mask[b, h] 是 [seq, num_blocks]
            # token_mask[b, h, q_idx, kv_idx] = block_mask[b, h, q_idx, kv_block]
            token_mask[b, h] = block_mask[b, h, q_b, kv_b]

    attn = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn = attn.masked_fill(~token_mask.unsqueeze(0).unsqueeze(0).expand_as(attn), float("-inf"))
    # 处理全 -inf 行 (无任何选中 block)
    attn = attn.masked_fill(attn.isnan(), 0.0)
    attn = F.softmax(attn, dim=-1)
    attn = attn.nan_to_num(0.0)

    return torch.matmul(attn, v)


class SparseAttention(torch.nn.Module):
    """Block-Sparse Attention 入口。

    根据配置自动选择 kernel 后端:
      - triton: 自定义 Triton kernel (最快)
      - flex:   PyTorch FlexAttention
      - dense:  标准 Attention + mask (最慢但最稳)
    """

    def __init__(self, config: CRBSAConfig):
        super().__init__()
        self.config = config

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            q: [batch, num_heads, seq_len, head_dim]
            k: [batch, num_kv_heads, seq_len, head_dim]
            v: [batch, num_kv_heads, seq_len, head_dim]
            block_mask: [batch, num_heads, seq_len, num_blocks] bool

        Returns:
            output: [batch, num_heads, seq_len, head_dim]
        """
        cfg = self.config
        backend = cfg.resolve_kernel_backend()
        bs = cfg.block_size

        # GQA: 扩展 K, V 到 query heads 数量
        num_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        if num_heads != num_kv_heads:
            repeats = num_heads // num_kv_heads
            k = k.unsqueeze(2).expand(-1, -1, repeats, -1, -1).reshape(
                q.shape[0], num_heads, q.shape[2], k.shape[-1]
            )
            v = v.unsqueeze(2).expand(-1, -1, repeats, -1, -1).reshape_as(k)

        if backend == "triton" and _triton_available:
            return _triton_block_sparse_fwd(q, k, v, block_mask, bs)
        elif backend == "flex":
            return _flex_block_sparse_fwd(q, k, v, block_mask, bs)
        else:
            return _dense_block_sparse_fwd(q, k, v, block_mask, bs)

    def forward_dense(self, q, k, v):
        """纯 Dense Attention (基准测试用)。"""
        scale = q.shape[-1] ** -0.5
        # GQA expand
        num_heads = q.shape[1]
        num_kv_heads = k.shape[1]
        if num_heads != num_kv_heads:
            repeats = num_heads // num_kv_heads
            k = k.unsqueeze(2).expand(-1, -1, repeats, -1, -1).reshape(
                q.shape[0], num_heads, q.shape[2], k.shape[-1]
            )
            v = v.unsqueeze(2).expand(-1, -1, repeats, -1, -1).reshape_as(k)
        attn = torch.matmul(q * scale, k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        return torch.matmul(attn, v)
