"""Step 1: Block Summarization。

将 KV Cache 按 Block 池化为摘要向量，用于后续路由。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from crbsa.config import CRBSAConfig
from crbsa.debug import DebugContext


class BlockSummarizer(nn.Module):
    """将 K 向量按 block_size 分块并池化为低维摘要。

    Input:  K [batch, num_kv_heads, seq_len, head_dim]
    Output: summary [batch, num_kv_heads, num_blocks, route_dim]
    """

    def __init__(self, config: CRBSAConfig):
        super().__init__()
        self.config = config
        self.proj = nn.Linear(config.head_dim, config.route_dim, bias=False)

    def forward(self, key: torch.Tensor) -> tuple[torch.Tensor, dict]:
        cfg = self.config
        debug = DebugContext(cfg, tag="summarizer")

        B = cfg.block_size
        batch, num_heads, seq_len, head_dim = key.shape

        # Pad to block boundary
        pad_len = (B - seq_len % B) % B
        if pad_len > 0:
            key = F.pad(key, (0, 0, 0, pad_len))

        padded_len = key.shape[2]
        num_blocks = padded_len // B

        # [batch, heads, num_blocks, B, head_dim]
        blocks = key.view(batch, num_heads, num_blocks, B, head_dim)

        # Pool → [batch, heads, num_blocks, head_dim]
        if cfg.pool_type == "avg":
            summary = blocks.mean(dim=3)
        else:
            # learned: 简单投影 + mean
            summary = self.proj(blocks).mean(dim=3)

        # 如果 avg 模式，也要投影到 route_dim
        if cfg.pool_type == "avg":
            summary = self.proj(summary)

        debug.log_tensor("block_summary", summary)
        debug.log_scalar("num_blocks", num_blocks)
        debug.log_scalar("pad_length", pad_len)

        return summary, debug.flush()
