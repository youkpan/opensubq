"""Kernel 性能 Benchmark。

对比 CRBSA (Triton/Flex/Dense) vs 标准 Dense Attention。
测量不同序列长度下的延迟和显存。
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch

from crbsa.config import CRBSAConfig
from crbsa.debug import CRBSAProfiler
from crbsa.nn.crbsa_layer import CRBSAAttention
from crbsa.utils.profiling import BenchmarkResult, measure_vram


def benchmark_layer(args):
    """单层 CRBSA vs Dense Attention benchmark。"""
    cfg = CRBSAConfig(
        debug_enabled=args.debug,
        debug_profile_kernel=True,
        kernel_backend=args.backend,
    )

    profiler = CRBSAProfiler.init(cfg)
    layer = CRBSAAttention(cfg, layer_id=0).cuda().to(torch.bfloat16)
    layer.eval()

    heads = cfg.num_attention_heads
    kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim
    batch = args.batch_size

    benchmark = BenchmarkResult()

    for seq_len in args.seq_lengths:
        print(f"\n=== seq_len={seq_len} ===")

        # 生成输入
        hidden = torch.randn(batch, seq_len, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = layer(hidden)

        torch.cuda.reset_peak_memory_stats()

        # ── CRBSA ──────────────────────────────────
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            with profiler.measure(f"crbsa_{seq_len}"):
                output, _, losses = layer(hidden)

        torch.cuda.synchronize()
        crbsa_ms = (time.perf_counter() - t0) * 1000
        crbsa_vram = measure_vram()

        del output
        torch.cuda.empty_cache()

        # ── Dense Baseline ─────────────────────────
        q = layer.q_proj(hidden).view(batch, seq_len, heads, head_dim).transpose(1, 2)
        k = layer.k_proj(hidden).view(batch, seq_len, kv_heads, head_dim).transpose(1, 2)
        v = layer.v_proj(hidden).view(batch, seq_len, kv_heads, head_dim).transpose(1, 2)

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.no_grad():
            dense_out = layer.sparse_attn.forward_dense(q, k, v)

        torch.cuda.synchronize()
        dense_ms = (time.perf_counter() - t0) * 1000
        dense_vram = measure_vram()

        speedup = dense_ms / max(crbsa_ms, 0.001)

        print(f"  CRBSA: {crbsa_ms:.1f}ms / {crbsa_vram:.0f}MB")
        print(f"  Dense: {dense_ms:.1f}ms / {dense_vram:.0f}MB")
        print(f"  Speedup: {speedup:.1f}x")

        benchmark.add(seq_len, batch, "crbsa", crbsa_ms, crbsa_vram, extra={"backend": cfg.resolve_kernel_backend()})
        benchmark.add(seq_len, batch, "dense", dense_ms, dense_vram, extra={"speedup": round(speedup, 2)})

        del q, k, v, dense_out
        torch.cuda.empty_cache()

    # ── 输出 ──────────────────────────────────────
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{benchmark}")
    print(profiler.report())

    benchmark.save(os.path.join(output_dir, "kernel_benchmark.json"))
    profiler.clear()


def main():
    parser = argparse.ArgumentParser(description="CRBSA Kernel Benchmark")
    parser.add_argument("--backend", default="auto", choices=["auto", "triton", "flex", "dense"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-lengths", nargs="+", type=int, default=[2048, 4096, 8192, 16384, 32768])
    parser.add_argument("--output-dir", default="./outputs/benchmark")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    benchmark_layer(args)


if __name__ == "__main__":
    main()
