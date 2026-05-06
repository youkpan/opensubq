"""CRBSA Attention Layer。

串联 BlockSummarizer → CodebookRouter → SparseAttention。
可替换标准 Transformer 中的 Attention 层。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from crbsa.config import CRBSAConfig
from crbsa.debug import DebugContext, DebugCollector, CRBSAProfiler
from crbsa.nn.block_summarizer import BlockSummarizer
from crbsa.nn.codebook_router import CodebookRouter, LoadBalancingLoss
from crbsa.nn.sparse_attention import SparseAttention


class CRBSAAttention(nn.Module):
    """CRBSA 完整 Attention Layer。

    用法:
        layer = CRBSAAttention(config)
        output, losses = layer(hidden_states, position_ids=position_ids)
        total_loss = lm_loss + sum(losses.values())
    """

    def __init__(self, config: CRBSAConfig, layer_id: int = 0):
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        hidden = config.hidden_size
        heads = config.num_attention_heads
        kv_heads = config.num_key_value_heads
        head_dim = config.head_dim

        self.hidden_size = hidden
        self.num_heads = heads
        self.num_kv_heads = kv_heads
        self.head_dim = head_dim

        # Q/K/V/O 投影
        self.q_proj = nn.Linear(hidden, heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(heads * head_dim, hidden, bias=False)

        # CRBSA 子模块
        self.summarizer = BlockSummarizer(config)
        self.router = CodebookRouter(config)
        self.sparse_attn = SparseAttention(config)
        self.balance_loss_fn = LoadBalancingLoss()

        # RoPE (如果需要在外部处理则跳过)
        self._use_rope = True

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        output_attentions: bool = False,
        detach_router: Optional[bool] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], dict]:
        """
        Args:
            hidden_states: [batch, seq_len, hidden_size]
            position_ids: [batch, seq_len]
            position_embeddings: (cos, sin) for RoPE
            detach_router: 是否截断路由器梯度 (覆盖 config)

        Returns:
            attn_output: [batch, seq_len, hidden_size]
            attn_weights: None (CRBSA 不输出完整 attention weights)
            losses: {"balance_loss": tensor, "routing_loss": tensor}
        """
        cfg = self.config
        profiler = CRBSAProfiler.get() if CRBSAProfiler._instance else None
        collector = DebugCollector.get() if DebugCollector._instance else None
        debug = DebugContext(cfg, tag=f"L{self.layer_id}")

        batch, seq_len, _ = hidden_states.shape

        should_detach = detach_router if detach_router is not None else cfg.detach_router
        should_freeze_router = cfg.freeze_router

        # ── 投影 ──────────────────────────────────
        with profiler.measure(f"L{self.layer_id}_proj") if profiler else _null_ctx():
            q = self.q_proj(hidden_states)
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)

            q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # ── RoPE ──────────────────────────────────
        if position_embeddings is not None:
            cos, sin = position_embeddings
            q, k = self._apply_rope(q, k, cos, sin)

        # ── Step 1: Block Summarization ────────────
        with profiler.measure(f"L{self.layer_id}_summarize") if profiler else _null_ctx():
            block_summary, step1_dbg = self.summarizer(k)

        # ── Step 2+3: Codebook Routing ────────────
        if should_freeze_router:
            with torch.no_grad():
                assignment, block_mask, topk_scores, step23_dbg = self.router(q, block_summary)
        else:
            assignment, block_mask, topk_scores, step23_dbg = self.router(q, block_summary)

        # ── 裁剪到固定 Block 数 ────────────────────
        block_mask = CodebookRouter.trim_to_fixed(
            block_mask, topk_scores, cfg.total_blocks_per_query
        )

        # ── Detach 路由器梯度 ──────────────────────
        if should_detach:
            block_mask = block_mask.detach()

        # ── Balance Loss ───────────────────────────
        num_blocks = block_summary.shape[2]
        q_route = self.router.query_proj(q)
        scores = torch.matmul(q_route, self.router.codebook.T)
        probs = torch.softmax(scores / cfg.route_temperature, dim=-1)

        balance_loss = self.balance_loss_fn(
            assignment.reshape(-1), probs.reshape(-1, cfg.codebook_size)
        )

        # ── Step 4: Sparse Attention ───────────────
        with profiler.measure(f"L{self.layer_id}_attn") if profiler else _null_ctx():
            # Pad K, V to block boundary
            B = cfg.block_size
            pad_len = (B - seq_len % B) % B
            if pad_len > 0:
                k_pad = torch.nn.functional.pad(k, (0, 0, 0, pad_len))
                v_pad = torch.nn.functional.pad(v, (0, 0, 0, pad_len))
            else:
                k_pad, v_pad = k, v

            attn_output = self.sparse_attn(q, k_pad, v_pad, block_mask)

            # 去 padding
            attn_output = attn_output[:, :, :seq_len, :]

        # ── Output ────────────────────────────────
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        output = self.o_proj(attn_output)

        # ── 调试收集 ──────────────────────────────
        debug.log_tensor("output", output)
        debug.log_scalar("balance_loss", balance_loss.item())
        debug.log_scalar("sparsity", 1.0 - block_mask.float().mean().item())
        debug.log_dict("step1", step1_dbg)
        debug.log_dict("step23", step23_dbg)

        dbg = debug.flush()
        if collector:
            collector.collect(self.layer_id, dbg)

        losses = {"balance_loss": balance_loss}

        return output, None, losses

    @staticmethod
    def _apply_rope(q, k, cos, sin):
        """应用 Rotary Position Embedding。"""
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed


class _null_ctx:
    """空 context manager，用于 profiler 为 None 时。"""
    def __enter__(self): return self
    def __exit__(self, *a): pass
