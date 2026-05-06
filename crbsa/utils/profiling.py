"""性能分析工具。"""

from __future__ import annotations

import json
import os
from typing import Any

import torch


class BenchmarkResult:
    """一次 benchmark 的结果。"""

    def __init__(self):
        self.records: list[dict[str, Any]] = []

    def add(
        self,
        seq_len: int,
        batch: int,
        backend: str,
        latency_ms: float,
        peak_vram_mb: float,
        sparsity: float = 0.0,
        extra: dict | None = None,
    ):
        rec = {
            "seq_len": seq_len,
            "batch": batch,
            "backend": backend,
            "latency_ms": round(latency_ms, 3),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "sparsity": round(sparsity, 4),
        }
        if extra:
            rec.update(extra)
        self.records.append(rec)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.records, f, indent=2)

    def __str__(self) -> str:
        if not self.records:
            return "No benchmark results."
        header = f"{'seq_len':>8} {'batch':>5} {'backend':>10} {'lat_ms':>10} {'vram_mb':>10} {'sparsity':>10}"
        lines = [header]
        for r in self.records:
            lines.append(
                f"{r['seq_len']:>8} {r['batch']:>5} {r['backend']:>10} "
                f"{r['latency_ms']:>10.3f} {r['peak_vram_mb']:>10.1f} {r['sparsity']:>10.4f}"
            )
        return "\n".join(lines)


def measure_vram() -> float:
    """返回当前 GPU 已用显存 (MB)。"""
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0.0
