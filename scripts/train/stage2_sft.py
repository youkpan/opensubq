"""Stage 2: 截断式稀疏微调。

全参解冻，使用路由器输出的稀疏掩码进行前向计算，
但在反向传播时将路由器梯度与语言模型 Loss 彻底截断 (Detach)。
"""

from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from crbsa.config import CRBSAConfig
from crbsa.debug import DebugCollector, CRBSAProfiler
from crbsa.models.qwen_crbsa import apply_crbsa_to_qwen3
from scripts.train.stage1_distill import LongTextDataset


def train_stage2(args):
    """Stage 2 主训练循环。"""
    # ── 配置 ──────────────────────────────────────
    cfg_path = os.path.join(args.stage1_dir, "crbsa_config.json")
    if os.path.exists(cfg_path):
        cfg = CRBSAConfig.from_json(cfg_path)
    else:
        cfg = CRBSAConfig.from_pretrained(args.model)

    cfg.detach_router = True     # 关键: 截断路由器梯度
    cfg.freeze_backbone = False
    cfg.freeze_router = False

    # 加载 Stage 1 的路由器权重
    stage1_model = args.stage1_dir

    # ── 初始化调试 ────────────────────────────────
    DebugCollector.init(cfg)
    CRBSAProfiler.init(cfg)

    # ── 加载模型 ──────────────────────────────────
    model = apply_crbsa_to_qwen3(args.model, cfg)
    model.unfreeze_all()
    model.enable_detach_router(True)
    model.train()

    # ── 数据 ──────────────────────────────────────
    dataset = LongTextDataset(args.data_dir, seq_len=args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

    # ── 优化器 ────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── 训练循环 ──────────────────────────────────
    profiler = CRBSAProfiler.get()
    collector = DebugCollector.get()

    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            input_ids = batch.to(model.device)
            labels = input_ids.clone()

            with profiler.measure("stage2_total"):
                result = model(input_ids=input_ids, labels=labels)
                lm_loss = result["loss"]
                balance_loss = result.get("balance_loss", torch.tensor(0.0))

                # 主损失: LM loss + 小权重 balance loss
                # balance loss 梯度只流向路由器 (detached from lm_loss)
                total_loss = lm_loss + 0.1 * balance_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            if step % args.log_interval == 0:
                print(
                    f"[Stage2] Epoch {epoch} Step {step} | "
                    f"lm_loss={lm_loss.item():.4f} "
                    f"balance={balance_loss.item():.4f} "
                    f"total={total_loss.item():.4f}"
                )

                if args.debug and step % (args.log_interval * 10) == 0:
                    print(collector.summary())
                    print(profiler.report())
                    collector.clear()
                    profiler.clear()

        scheduler.step()

    # ── 保存 ──────────────────────────────────────
    model.save_pretrained(os.path.join(args.output_dir, "stage2_final"))
    print(profiler.report())


def main():
    parser = argparse.ArgumentParser(description="CRBSA Stage 2: Detached Sparse Tuning")
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--stage1-dir", default="./outputs/stage1")
    parser.add_argument("--data-dir", default="./data/train")
    parser.add_argument("--output-dir", default="./outputs/stage2")
    parser.add_argument("--seq-len", type=int, default=131072)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    train_stage2(args)


if __name__ == "__main__":
    main()
