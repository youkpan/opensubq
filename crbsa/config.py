"""CRBSA 全局配置。所有超参与调试开关集中管理。"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class CRBSAConfig:
    """CRBSA 核心配置。

    用法:
        cfg = CRBSAConfig()
        cfg = CRBSAConfig.from_json("config.json")
        cfg.save("config.json")
    """

    # ── 架构参数 ──────────────────────────────────
    block_size: int = 128                # Block 大小 (tokens)
    codebook_size: int = 1024            # Codebook 聚类数 M
    route_dim: int = 64                  # 路由降维维度 d_route
    topk_codebooks: int = 4              # Query 召回的 Codebook 数
    max_routed_blocks: int = 6           # 最大远距离 Block 数
    local_blocks: int = 2                # 局部保底 Block 数 (当前块 + 前一块)
    route_temperature: float = 1.0       # Codebook 选择温度 τ
    pool_type: str = "avg"               # "avg" | "learned"

    # ── Transformer 补充 (从基座模型自动填充) ─────
    hidden_size: int = 2048
    num_attention_heads: int = 16
    num_key_value_heads: int = 4         # GQA
    head_dim: int = 128
    num_hidden_layers: int = 28
    intermediate_size: int = 5632
    vocab_size: int = 151936
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-6

    # ── MoE (Qwen3.6-35B-A3B) ────────────────────
    is_moe: bool = True
    num_experts: int = 32
    num_experts_per_tok: int = 4

    # ── 位置编码 ──────────────────────────────────
    rope_scaling_type: str = "yarn"      # "yarn" | "ntk" | "dynamic" | "none"
    rope_scaling_factor: float = 8.0     # YaRN x8 用于 1M 外推

    # ── 训练阶段控制 ──────────────────────────────
    detach_router: bool = False          # Stage 2 设为 True
    freeze_router: bool = False          # Stage 1 设为 True
    freeze_backbone: bool = False        # Stage 1 设为 True

    # ── 调试开关 ──────────────────────────────────
    debug_enabled: bool = False
    debug_log_routing: bool = False
    debug_log_block_assignment: bool = False
    debug_log_attention_weights: bool = False
    debug_visualize_codebook: bool = False
    debug_profile_kernel: bool = False
    debug_check_numerics: bool = False
    debug_save_intermediates: bool = False
    debug_intermediate_dir: str = "./debug_outputs"

    # ── Kernel 参数 ───────────────────────────────
    kernel_backend: str = "auto"         # "auto" | "triton" | "flex" | "dense"

    # ── 分布式参数 ────────────────────────────────
    sequence_parallel: bool = False
    sequence_parallel_world_size: int = 1
    async_kv_fetch: bool = False

    # ── 模型路径 ──────────────────────────────────
    base_model_name_or_path: str = "Qwen/Qwen3.6-35B-A3B"

    # ── 派生属性 ──────────────────────────────────
    @property
    def total_blocks_per_query(self) -> int:
        return self.local_blocks + self.max_routed_blocks

    @property
    def tokens_per_query(self) -> int:
        return self.total_blocks_per_query * self.block_size

    def resolve_kernel_backend(self) -> str:
        """自动检测可用的 kernel 后端。"""
        if self.kernel_backend != "auto":
            return self.kernel_backend
        try:
            import triton  # noqa: F401
            return "triton"
        except ImportError:
            pass
        try:
            from torch.nn.attention.flex_attention import flex_attention  # noqa: F401
            return "flex"
        except ImportError:
            pass
        return "dense"

    # ── 序列化 ────────────────────────────────────
    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str) -> "CRBSAConfig":
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))

    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **overrides) -> "CRBSAConfig":
        """从 HuggingFace 预训练模型配置中提取参数。"""
        from transformers import AutoConfig

        hf_cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        cfg = cls()

        # 自动映射通用字段
        _mapping = {
            "hidden_size": "hidden_size",
            "num_attention_heads": "num_attention_heads",
            "num_key_value_heads": "num_key_value_heads",
            "head_dim": "head_dim",
            "num_hidden_layers": "num_hidden_layers",
            "intermediate_size": "intermediate_size",
            "vocab_size": "vocab_size",
            "rope_theta": "rope_theta",
            "rms_norm_eps": "rms_norm_eps",
        }
        for hf_key, our_key in _mapping.items():
            val = getattr(hf_cfg, hf_key, None)
            if val is not None:
                setattr(cfg, our_key, val)

        # MoE 检测
        cfg.is_moe = getattr(hf_cfg, "decoder_sparse_step", 1) != 1 or getattr(hf_cfg, "num_experts", 0) > 1
        if cfg.is_moe:
            cfg.num_experts = getattr(hf_cfg, "num_experts", 32)
            cfg.num_experts_per_tok = getattr(hf_cfg, "num_experts_per_tok", 4)

        # head_dim 自动推断
        if cfg.head_dim == 0:
            cfg.head_dim = cfg.hidden_size // cfg.num_attention_heads

        cfg.base_model_name_or_path = model_name_or_path
        for k, v in overrides.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        return cfg
