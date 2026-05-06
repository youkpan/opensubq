"""Step 2+3: Codebook 倒排索引 + Query 路由。

核心思想：固定 Codebook M 个聚类中心，Block 分配到聚类 → 倒排索引。
Query 只需和 M 个聚类中心打分 (O(M))，就能召回候选 Block。
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from crbsa.config import CRBSAConfig
from crbsa.debug import DebugContext


class CodebookRouter(nn.Module):
    """Codebook 路由器。

    可学习参数:
        codebook: [M, d_route] — 全局语义聚类中心
        query_proj: Linear(head_dim, d_route) — Query 投影
        key_proj: Linear(head_dim, d_route) — Key 投影 (与 summarizer 共享时可不设)
    """

    def __init__(self, config: CRBSAConfig):
        super().__init__()
        self.config = config
        M = config.codebook_size
        d = config.route_dim

        self.codebook = nn.Parameter(torch.randn(M, d) * 0.02)
        self.query_proj = nn.Linear(config.head_dim, d, bias=False)

    def forward(
        self,
        query: torch.Tensor,
        block_summary: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """完整路由：建索引 + 查询。

        Args:
            query: [batch, num_heads, seq_len, head_dim]
            block_summary: [batch, num_kv_heads, num_blocks, route_dim]

        Returns:
            assignment: [batch, num_kv_heads, num_blocks] — 每个 Block 的 Codebook ID
            block_mask: [batch, num_heads, seq_len, num_blocks] — bool
            topk_scores: [batch, num_heads, seq_len, topk_codebooks] — 路由分数
            debug_info: dict
        """
        cfg = self.config
        debug = DebugContext(cfg, tag="router")

        batch, num_heads, seq_len, _ = query.shape
        num_kv_heads = block_summary.shape[1]
        num_blocks = block_summary.shape[2]

        # ── Step 2: 倒排索引 ──────────────────────
        # [batch, kv_heads, num_blocks, M]
        sim_kv = torch.matmul(block_summary, self.codebook.T)
        assignment = sim_kv.argmax(dim=-1)  # [batch, kv_heads, num_blocks]

        debug.log_codebook_stats(assignment.reshape(-1), self.codebook)

        # ── Step 3: Query 路由 ─────────────────────
        q_route = self.query_proj(query)  # [batch, heads, seq, d_route]

        # GQA: 扩展 KV-head 维度到 query-head 维度
        if num_heads != num_kv_heads:
            repeats = num_heads // num_kv_heads
            assignment = assignment.unsqueeze(2).expand(-1, -1, repeats, -1, -1).reshape(
                batch, num_heads, num_blocks
            )

        # [batch, heads, seq, M]
        scores = torch.matmul(q_route, self.codebook.T) / cfg.route_temperature
        probs = F.softmax(scores, dim=-1)

        # Top-K Codebook: [batch, heads, seq, K]
        topk_scores, topk_ids = probs.topk(cfg.topk_codebooks, dim=-1)

        # ── 展开 Block Mask ────────────────────────
        # assignment: [batch, heads, num_blocks]
        # topk_ids:   [batch, heads, seq, K]
        # → block_mask: [batch, heads, seq, num_blocks]
        block_mask = self._build_block_mask(assignment, topk_ids, num_blocks, cfg)

        # ── 加入局部滑动窗口 ────────────────────────
        block_mask = self._add_local_window(block_mask, seq_len, num_blocks, cfg)

        debug.log_routing(topk_ids=topk_ids, topk_scores=topk_scores)
        debug.log_tensor("block_mask", block_mask.float())
        debug.log_scalar("avg_selected_blocks",
                         block_mask.sum().item() / max(1, batch * num_heads * seq_len))

        return assignment, block_mask, topk_scores, debug.flush()

    @staticmethod
    def _build_block_mask(
        assignment: torch.Tensor,
        topk_ids: torch.Tensor,
        num_blocks: int,
        cfg: CRBSAConfig,
    ) -> torch.Tensor:
        """将 Top-K Codebook ID 展开为 Block 级 bool mask。"""
        batch, heads, seq, K = topk_ids.shape
        device = topk_ids.device

        # [batch, heads, seq, num_blocks]
        block_mask = torch.zeros(batch, heads, seq, num_blocks, device=device, dtype=torch.bool)

        # 向量化: 对每个 K 检查 assignment 是否匹配
        # assignment: [batch, heads, num_blocks] → [batch, heads, 1, num_blocks]
        a = assignment.unsqueeze(2)
        for k in range(K):
            # [batch, heads, seq, 1] == [batch, heads, 1, num_blocks]
            match = topk_ids[..., k : k + 1] == a
            block_mask |= match

        return block_mask

    @staticmethod
    def _add_local_window(
        block_mask: torch.Tensor,
        seq_len: int,
        num_blocks: int,
        cfg: CRBSAConfig,
    ) -> torch.Tensor:
        """加入局部滑动窗口: 当前 Block + 前 local_blocks-1 个 Block。"""
        device = block_mask.device
        B = cfg.block_size

        # 每个 query token 属于哪个 block
        query_block = torch.arange(seq_len, device=device) // B  # [seq]

        for offset in range(cfg.local_blocks):
            local_block = (query_block - offset).clamp(min=0)  # [seq]
            # [batch, heads, seq, 1] → scatter
            idx = local_block.view(1, 1, seq_len, 1).expand_as(block_mask[..., :1])
            block_mask.scatter_(-1, idx, True)

        return block_mask

    @staticmethod
    def trim_to_fixed(
        block_mask: torch.Tensor,
        scores: torch.Tensor,
        max_blocks: int,
    ) -> torch.Tensor:
        """裁剪候选 Block 到固定数量，消除 Load Imbalance。"""
        current = block_mask.sum(dim=-1)  # [batch, heads, seq]
        over = current > max_blocks
        if not over.any():
            return block_mask

        # 简化: 对超出部分随机裁剪 (实际可用 block-level score 精细排序)
        # 保留前 max_blocks 个 True
        result = block_mask.clone()
        # 按行遍历超出部分
        for b, h, s in over.nonzero(as_tuple=False):
            true_ids = result[b, h, s].nonzero(as_tuple=False).squeeze(-1)
            if len(true_ids) > max_blocks:
                keep = true_ids[:max_blocks]
                result[b, h, s] = False
                result[b, h, s, keep] = True
        return result


class LoadBalancingLoss(nn.Module):
    """MoE 风格负载均衡损失，防止 Codebook 索引坍塌。

    L_bal = α * Σ(f_i · P_i)
    f_i = 分配到第 i 个 codebook 的块比例
    P_i = Query 选中第 i 个 codebook 的平均概率
    """

    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha

    def forward(
        self,
        assignment: torch.Tensor,
        query_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            assignment: [N] 每个 Block 的 Codebook ID (已 flatten)
            query_probs: [N_q, M] Query 对各 Codebook 的概率
        """
        M = query_probs.shape[-1]
        f = assignment.float().bincount(minlength=M) / max(assignment.numel(), 1)
        P = query_probs.reshape(-1, M).mean(dim=0)
        return self.alpha * (f * P).sum()
