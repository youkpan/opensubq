"""Stage 1: 路由器蒸馏。

冻结大模型主干，用稠密 Attention 的 Ground Truth 训练 Codebook 路由器。
"""

from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import Dataset, DataLoader

from crbsa.config import CRBSAConfig
from crbsa.debug import DebugCollector, CRBSAProfiler
from crbsa.models.qwen_crbsa import apply_crbsa_to_qwen3


class LongTextDataset(Dataset):
    """长文本数据集。支持代码、论文、逻辑推理文本。"""

    def __init__(self, data_dir: str, seq_len: int = 131072):
        self.seq_len = seq_len
        self.files = []
        if os.path.isdir(data_dir):
            for root, _, fnames in os.walk(data_dir):
                for f in fnames:
                    if f.endswith((".txt", ".json", ".jsonl")):
                        self.files.append(os.path.join(root, f))

    def __len__(self):
        return max(len(self.files), 1)

    def __getitem__(self, idx):
        # 简化: 返回随机 token IDs
        # 生产环境应从文件读取真实文本并 tokenize
        return torch.randint(0, 151936, (self.seq_len,))


def collect_teacher_labels(
    model,
    input_ids: torch.Tensor,
    top_k_blocks: int = 8,
    block_size: int = 128,
) -> torch.Tensor:
    """用稠密 Attention 的权重作为 Ground Truth。

    记录每个 Query 真正产生高 Attention Weight 的 Block IDs。

    Returns:
        teacher_block_ids: [batch, num_heads, seq_len, top_k_blocks]
    """
    # 在 CRBSA 模型中，临时切换到 dense attention 收集标签
    teacher_labels = []

    def hook_fn(module, input, output):
        # output 是 attention weights [batch, heads, seq, seq]
        # 简化: 这里需要根据实际模型结构调整
        pass

    # 简化实现: 随机生成 teacher labels
    # 生产环境应: model.forward() → 收集每层 attention weights → 找 top-k blocks
    batch, seq = input_ids.shape
    num_blocks = seq // block_size
    # 随机 teacher labels (仅用于框架验证)
    teacher_labels = torch.randint(0, num_blocks, (batch, 32, seq, top_k_blocks))

    return teacher_labels


def distillation_loss(
    student_topk_ids: torch.Tensor,
    teacher_block_ids: torch.Tensor,
) -> torch.Tensor:
    """路由器蒸馏损失: Cross-Entropy。

    强制路由器的 Top-K Blocks 命中 Teacher 选出的 Blocks。

    Args:
        student_topk_ids: [batch, heads, seq, K] — 路由器选出的 Block IDs
        teacher_block_ids: [batch, heads, seq, K'] — Teacher 的 Ground Truth
    """
    # 简化: 计算命中率作为 loss
    # 生产环境应使用真正的 Cross-Entropy
    batch, heads, seq, K = student_topk_ids.shape
    K_t = teacher_block_ids.shape[-1]

    # [batch, heads, seq, K, 1] vs [batch, heads, seq, 1, K_t]
    hit = (student_topk_ids.unsqueeze(-1) == teacher_block_ids.unsqueeze(-2)).any(-1)
    hit_rate = hit.float().mean()
    return 1.0 - hit_rate  # 1 - 命中率 = loss


def train_stage1(args):
    """Stage 1 主训练循环。"""
    # ── 配置 ──────────────────────────────────────
    cfg = CRBSAConfig.from_pretrained(
        args.model,
        debug_enabled=args.debug,
        debug_check_numerics=args.debug,
        debug_log_routing=args.debug,
        debug_log_block_assignment=args.debug,
        freeze_backbone=True,
        freeze_router=False,
        detach_router=False,
    )
    cfg.save(os.path.join(args.output_dir, "crbsa_config.json"))

    # ── 初始化调试 ────────────────────────────────
    DebugCollector.init(cfg)
    CRBSAProfiler.init(cfg)

    # ── 加载模型 ──────────────────────────────────
    model = apply_crbsa_to_qwen3(args.model, cfg)
    model.freeze_backbone()
    model.train()

    # ── 数据 ──────────────────────────────────────
    dataset = LongTextDataset(args.data_dir, seq_len=args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # ── 优化器 ────────────────────────────────────
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── 训练循环 ──────────────────────────────────
    profiler = CRBSAProfiler.get()
    collector = DebugCollector.get()

    for epoch in range(args.epochs):
        for step, input_ids in enumerate(loader):
            input_ids = input_ids.to(model.device)

            with profiler.measure("stage1_total"):
                # Teacher 标注 (简化)
                teacher_labels = collect_teacher_labels(model, input_ids)

                # Student 前向
                result = model(input_ids=input_ids)

                # 蒸馏损失
                # (实际应从 crbsa_layers 收集 student_topk_ids)
                distill_loss = torch.tensor(0.5, device=model.device, requires_grad=True)
                balance_loss = result.get("balance_loss", torch.tensor(0.0))
                total_loss = distill_loss + balance_loss

                # 反向
                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()

            if step % args.log_interval == 0:
                print(
                    f"[Stage1] Epoch {epoch} Step {step} | "
                    f"loss={total_loss.item():.4f} "
                    f"balance={balance_loss.item():.4f}"
                )

                if args.debug and step % (args.log_interval * 10) == 0:
                    print(collector.summary())
                    collector.clear()

        scheduler.step()

    # ── 保存 ──────────────────────────────────────
    model.save_pretrained(os.path.join(args.output_dir, "stage1_final"))
    print(profiler.report())

    if args.debug:
        collector.to_json(os.path.join(args.output_dir, "stage1_debug.json"))


def main():
    parser = argparse.ArgumentParser(description="CRBSA Stage 1: Router Distillation")
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--data-dir", default="./data/train")
    parser.add_argument("--output-dir", default="./outputs/stage1")
    parser.add_argument("--seq-len", type=int, default=131072)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    train_stage1(args)


if __name__ == "__main__":
    main()
