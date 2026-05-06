"""快速验证脚本：单卡测试 CRBSA 各模块正确性。

用法:
    python scripts/verify.py --model Qwen/Qwen3.6-35B-A3B --seq-len 2048 --debug
    python scripts/verify.py --seq-len 4096  # 不加载模型，纯模块测试
"""

from __future__ import annotations

import argparse
import os
import sys


def verify_modules(cfg):
    """验证各模块基本功能。"""
    import torch
    from crbsa.nn.block_summarizer import BlockSummarizer
    from crbsa.nn.codebook_router import CodebookRouter, LoadBalancingLoss
    from crbsa.nn.sparse_attention import SparseAttention
    from crbsa.nn.crbsa_layer import CRBSAAttention

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    batch, heads, kv_heads, seq_len, head_dim = 1, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.seq_len, cfg.head_dim

    print(f"Device: {device}, dtype: {dtype}")
    print(f"Config: heads={heads}, kv_heads={kv_heads}, head_dim={head_dim}, block_size={cfg.block_size}")
    print()

    # ── 1. Block Summarizer ───────────────────────
    print("[1/4] Block Summarizer...")
    summarizer = BlockSummarizer(cfg).to(device).to(dtype)
    k = torch.randn(batch, kv_heads, seq_len, head_dim, device=device, dtype=dtype)
    summary, dbg = summarizer(k)
    num_blocks = seq_len // cfg.block_size
    assert summary.shape == (batch, kv_heads, num_blocks, cfg.route_dim), \
        f"shape mismatch: {summary.shape} != ({batch}, {kv_heads}, {num_blocks}, {cfg.route_dim})"
    print(f"   OK: {summary.shape}, debug={list(dbg.keys())}")

    # ── 2. Codebook Router ────────────────────────
    print("[2/4] Codebook Router...")
    router = CodebookRouter(cfg).to(device).to(dtype)
    q = torch.randn(batch, heads, seq_len, head_dim, device=device, dtype=dtype)
    assignment, block_mask, topk_scores, dbg = router(q, summary)
    assert block_mask.shape == (batch, heads, seq_len, num_blocks), \
        f"block_mask shape mismatch: {block_mask.shape}"
    avg_blocks = block_mask.sum().item() / (batch * heads * seq_len)
    print(f"   OK: mask={block_mask.shape}, avg_selected_blocks={avg_blocks:.1f}, debug={list(dbg.keys())}")

    # ── 3. Sparse Attention ───────────────────────
    print("[3/4] Sparse Attention...")
    sparse = SparseAttention(cfg).to(device).to(dtype)
    v = torch.randn(batch, kv_heads, seq_len, head_dim, device=device, dtype=dtype)
    out = sparse(q, k, v, block_mask)
    assert out.shape == (batch, heads, seq_len, head_dim), f"output shape mismatch: {out.shape}"
    nan_count = torch.isnan(out).sum().item()
    print(f"   OK: {out.shape}, nan={nan_count}")

    # ── 4. Full Layer ─────────────────────────────
    print("[4/4] Full CRBSA Layer...")
    layer = CRBSAAttention(cfg, layer_id=0).to(device).to(dtype)
    hidden = torch.randn(batch, seq_len, cfg.hidden_size, device=device, dtype=dtype)
    output, attn_w, losses = layer(hidden)
    assert output.shape == (batch, seq_len, cfg.hidden_size), f"output shape mismatch: {output.shape}"
    print(f"   OK: {output.shape}, losses={list(losses.keys())}")

    print("\nAll modules verified successfully!")


def verify_model(cfg, model_name):
    """验证完整模型加载和前向传播。"""
    import torch
    from crbsa.debug import DebugCollector, CRBSAProfiler
    from crbsa.models.qwen_crbsa import apply_crbsa_to_qwen3

    print(f"\n=== Loading model: {model_name} ===")

    DebugCollector.init(cfg)
    CRBSAProfiler.init(cfg)

    model = apply_crbsa_to_qwen3(model_name, cfg)
    model.eval()

    print(f"Model loaded. Device: {model.device}")

    # 前向传播
    input_ids = torch.randint(0, cfg.vocab_size, (1, cfg.seq_len)).to(model.device)
    print(f"Running forward pass with {cfg.seq_len} tokens...")

    with torch.no_grad():
        result = model(input_ids=input_ids)

    logits = result["logits"]
    print(f"Output logits shape: {logits.shape}")
    print(f"Balance loss: {result.get('balance_loss', 'N/A')}")

    # 调试输出
    collector = DebugCollector.get()
    print(collector.summary())

    print("\nModel verification complete!")


def main():
    parser = argparse.ArgumentParser(description="CRBSA Quick Verify")
    parser.add_argument("--model", default=None, help="HuggingFace model name (skip if None)")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    from crbsa.config import CRBSAConfig

    cfg = CRBSAConfig(
        seq_len=args.seq_len,
        debug_enabled=args.debug,
        debug_check_numerics=True,
        debug_log_routing=args.debug,
        debug_log_block_assignment=args.debug,
    )
    cfg.seq_len = args.seq_len  # for verify script

    print("=" * 60)
    print("CRBSA Quick Verify")
    print("=" * 60)

    # 模块测试 (不需要模型)
    verify_modules(cfg)

    # 模型测试 (需要模型)
    if args.model:
        verify_model(cfg, args.model)


if __name__ == "__main__":
    main()
