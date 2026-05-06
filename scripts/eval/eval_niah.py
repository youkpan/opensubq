"""Needle-In-A-Haystack (NIAH) 评测。

在超长上下文中随机位置插入"针"，测试模型能否精确检索。
支持单针、多针、多跳变量追踪。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from typing import Optional

import torch

from crbsa.config import CRBSAConfig
from crbsa.debug import DebugCollector, CRBSAProfiler
from crbsa.models.qwen_crbsa import apply_crbsa_to_qwen3
from crbsa.utils.profiling import BenchmarkResult, measure_vram


# ── NIAH 数据生成 ─────────────────────────────────

DISTRACTORS = [
    "The quick brown fox jumps over the lazy dog. ",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ",
    "In a hole in the ground there lived a hobbit. ",
    "To be or not to be, that is the question. ",
    "All that glitters is not gold, but sometimes it is. ",
]


def build_haystack(
    context_length: int,
    needle: str,
    needle_depth: float = 0.5,  # 0.0~1.0, 相对位置
) -> tuple[str, int]:
    """构建 NIAH 上下文。

    Returns:
        (context, needle_token_position)
    """
    needle_pos = int(context_length * needle_depth)
    parts = []
    pos = 0
    while pos < context_length:
        if abs(pos - needle_pos) < 50:
            parts.append(needle + " ")
            pos += len(needle.split())
        else:
            chunk = random.choice(DISTRACTORS)
            parts.append(chunk)
            pos += len(chunk.split())
    return "".join(parts)[:context_length], needle_pos


def build_multi_needle(
    context_length: int,
    needles: list[str],
) -> tuple[str, list[int]]:
    """多针 NIAH。"""
    n = len(needles)
    positions = [int(context_length * (i + 1) / (n + 1)) for i in range(n)]

    parts = []
    pos = 0
    needle_idx = 0
    while pos < context_length:
        if needle_idx < n and abs(pos - positions[needle_idx]) < 50:
            parts.append(needles[needle_idx] + " ")
            pos += len(needles[needle_idx].split())
            needle_idx += 1
        else:
            parts.append(random.choice(DISTRACTORS))
            pos += 20
    return "".join(parts)[:context_length], positions


# ── 评测 ──────────────────────────────────────────

def eval_niah(args):
    """运行 NIAH 评测。"""
    cfg = CRBSAConfig.from_pretrained(args.model, debug_enabled=args.debug)
    if args.config:
        cfg = CRBSAConfig.from_json(args.config)

    DebugCollector.init(cfg)
    CRBSAProfiler.init(cfg)
    profiler = CRBSAProfiler.get()

    model = apply_crbsa_to_qwen3(args.model, cfg)
    model.eval()

    results = []

    # 测试矩阵: 不同长度 × 不同深度
    lengths = args.lengths or [8192, 32768, 65536, 131072]
    depths = args.depths or [0.0, 0.25, 0.5, 0.75, 1.0]

    needle = args.needle or "The magic number for today is 42719."
    question = "What is the magic number mentioned in the text?"
    answer = "42719"

    benchmark = BenchmarkResult()

    for seq_len in lengths:
        for depth in depths:
            context, needle_pos = build_haystack(seq_len, needle, depth)

            if args.debug:
                print(f"\n--- NIAH: len={seq_len}, depth={depth:.0%}, needle_pos={needle_pos} ---")

            # Tokenize
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
            input_text = f"{context}\n\nQuestion: {question}\nAnswer:"
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=seq_len + 128)
            input_ids = inputs["input_ids"].to(model.device)
            actual_len = input_ids.shape[1]

            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()

            with torch.no_grad():
                with profiler.measure(f"niah_{seq_len}_{depth}"):
                    outputs = model.hf_model.generate(
                        input_ids,
                        max_new_tokens=64,
                        do_sample=False,
                    )

            latency = (time.perf_counter() - t0) * 1000
            vram = measure_vram()

            generated = tokenizer.decode(outputs[0, input_ids.shape[1]:], skip_special_tokens=True)
            correct = answer in generated

            result = {
                "seq_len": actual_len,
                "target_len": seq_len,
                "depth": depth,
                "needle_pos": needle_pos,
                "correct": correct,
                "latency_ms": latency,
                "vram_mb": vram,
                "generated": generated[:100],
            }
            results.append(result)
            benchmark.add(actual_len, 1, cfg.resolve_kernel_backend(), latency, vram)

            status = "PASS" if correct else "FAIL"
            print(f"  [{status}] len={actual_len} depth={depth:.0%} lat={latency:.0f}ms gen={generated[:50]}")

    # ── 输出 ──────────────────────────────────────
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "niah_results.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 汇总
    total = len(results)
    passed = sum(1 for r in results if r["correct"])
    print(f"\n=== NIAH Summary: {passed}/{total} ({passed/total:.1%}) ===")
    print(benchmark)
    print(profiler.report())

    benchmark.save(os.path.join(output_dir, "niah_benchmark.json"))

    if args.debug:
        DebugCollector.get().to_json(os.path.join(output_dir, "niah_debug.json"))

    return results


def main():
    parser = argparse.ArgumentParser(description="CRBSA NIAH Evaluation")
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--config", default=None, help="CRBSA config JSON")
    parser.add_argument("--output-dir", default="./outputs/eval_niah")
    parser.add_argument("--lengths", nargs="+", type=int, default=None)
    parser.add_argument("--depths", nargs="+", type=float, default=None)
    parser.add_argument("--needle", default=None)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    eval_niah(args)


if __name__ == "__main__":
    main()
