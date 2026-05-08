"""调试系统：DebugContext, DebugCollector, CRBSAProfiler。

debug_enabled=False 时所有方法为空操作，零性能开销。
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from typing import Any, Optional

import torch

from crbsa.config import CRBSAConfig


class DebugContext:
    """轻量调试上下文，附加到每个模块的 forward 中。"""

    def __init__(self, config: CRBSAConfig, tag: str = ""):
        self._on = config.debug_enabled
        self._cfg = config
        self._tag = tag
        self._data: dict[str, Any] = {}

    # ── 基础记录 ──────────────────────────────────

    def log_tensor(self, name: str, t: torch.Tensor):
        if not self._on:
            return
        info: dict[str, Any] = {"shape": list(t.shape), "dtype": str(t.dtype)}
        if self._cfg.debug_check_numerics:
            flat = t.detach().float()
            info.update(
                has_nan=bool(torch.isnan(flat).any()),
                has_inf=bool(torch.isinf(flat).any()),
                mean=float(flat.mean()),
                std=float(flat.std()),
                min=float(flat.min()),
                max=float(flat.max()),
            )
        self._data[name] = info

        if self._cfg.debug_save_intermediates:
            d = self._cfg.debug_intermediate_dir
            os.makedirs(d, exist_ok=True)
            tag = f"{self._tag}_" if self._tag else ""
            torch.save(t.detach().cpu(), os.path.join(d, f"{tag}{name}.pt"))

    def log_scalar(self, name: str, value: float):
        if not self._on:
            return
        self._data[name] = value

    def log_dict(self, name: str, d: dict):
        if not self._on:
            return
        self._data[name] = d

    # ── 专项记录 ──────────────────────────────────

    def log_routing(
        self,
        topk_ids: Optional[torch.Tensor] = None,
        topk_scores: Optional[torch.Tensor] = None,
        target_block_ids: Optional[torch.Tensor] = None,
    ):
        if not self._on or not self._cfg.debug_log_routing:
            return
        info: dict[str, Any] = {}
        if topk_ids is not None:
            info["topk_ids_shape"] = list(topk_ids.shape)
            info["topk_ids_sample"] = topk_ids[0, 0, 0, :].tolist() if topk_ids.numel() else []
        if topk_scores is not None:
            info["topk_scores_sample"] = topk_scores[0, 0, 0, :].tolist() if topk_scores.numel() else []
        if target_block_ids is not None:
            info["target_block_ids_shape"] = list(target_block_ids.shape)
        self._data["routing"] = info

    def log_codebook_stats(self, assignment: torch.Tensor, codebook: torch.Tensor):
        if not self._on or not self._cfg.debug_log_block_assignment:
            return
        M = codebook.shape[0]
        counts = assignment.bincount(minlength=M).float()
        total = counts.sum()
        probs = counts / total if total > 0 else counts
        self._data["codebook_stats"] = {
            "entropy": float(-(probs * (probs + 1e-10).log()).sum()),
            "max_bucket": int(counts.max()),
            "min_bucket": int(counts.min()),
            "empty_buckets": int((counts == 0).sum()),
            "total_blocks": int(total),
        }

    # ── 输出 ──────────────────────────────────────

    def flush(self) -> dict:
        data = self._data
        self._data = {}
        return data


class DebugCollector:
    """跨层聚合 debug 信息，全局单例。"""

    _instance: Optional["DebugCollector"] = None

    def __init__(self, config: CRBSAConfig):
        self._cfg = config
        self._layers: dict[int, dict] = {}
        self._global: dict[str, Any] = {}

    @classmethod
    def init(cls, config: CRBSAConfig) -> "DebugCollector":
        cls._instance = cls(config)
        return cls._instance

    @classmethod
    def get(cls) -> "DebugCollector":
        assert cls._instance is not None, "Call DebugCollector.init() first"
        return cls._instance

    def collect(self, layer_id: int, info: dict):
        if not self._cfg.debug_enabled:
            return
        self._layers[layer_id] = info

    def add_global(self, key: str, value: Any):
        if not self._cfg.debug_enabled:
            return
        self._global[key] = value

    def summary(self) -> str:
        if not self._layers:
            return "No debug info collected."
        lines = ["=== CRBSA Debug Summary ==="]
        for lid, info in sorted(self._layers.items()):
            lines.append(f"\n--- Layer {lid} ---")
            for k, v in _flatten(info, prefix="  "):
                lines.append(f"{k}: {_fmt(v)}")
        if self._global:
            lines.append("\n--- Global ---")
            for k, v in self._global.items():
                lines.append(f"  {k}: {_fmt(v)}")
        return "\n".join(lines)

    def to_json(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"layers": self._layers, "global": self._global}, f, indent=2, default=str)

    def clear(self):
        self._layers.clear()
        self._global.clear()


class CRBSAProfiler:
    """各步骤耗时统计。"""

    _instance: Optional["CRBSAProfiler"] = None

    def __init__(self, config: CRBSAConfig):
        self._on = config.debug_enabled and config.debug_profile_kernel
        self._timings: dict[str, list[float]] = {}

    @classmethod
    def init(cls, config: CRBSAConfig) -> "CRBSAProfiler":
        cls._instance = cls(config)
        return cls._instance

    @classmethod
    def get(cls) -> "CRBSAProfiler":
        assert cls._instance is not None, "Call CRBSAProfiler.init() first"
        return cls._instance

    @contextmanager
    def measure(self, step_name: str):
        if not self._on:
            yield
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        yield
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._timings.setdefault(step_name, []).append(time.perf_counter() - t0)

    def report(self) -> str:
        if not self._timings:
            return "No profiling data."
        lines = ["=== CRBSA Profiling ==="]
        total_all = 0.0
        for step, ts in self._timings.items():
            avg = sum(ts) / len(ts)
            total = sum(ts)
            total_all += total
            lines.append(f"  {step}: avg={avg*1000:.2f}ms  total={total*1000:.1f}ms  n={len(ts)}")
        lines.append(f"  TOTAL: {total_all*1000:.1f}ms")
        return "\n".join(lines)

    def clear(self):
        self._timings.clear()


# ── 工具函数 ──────────────────────────────────────

def _flatten(d: dict, prefix: str = "") -> list[tuple[str, Any]]:
    out = []
    for k, v in d.items():
        if isinstance(v, dict):
            out.extend(_flatten(v, prefix=f"{prefix}{k}."))
        else:
            out.append((f"{prefix}{k}", v))
    return out


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:.6f}"
    if isinstance(v, (list, tuple)) and len(v) <= 10:
        return str(v)
    return str(v)
