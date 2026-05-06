"""Stage 3: GRPO 强化学习对齐。

针对"功能性长上下文"的 RL 训练。
奖励模型: R_correct + R_routing + R_hallucination。
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import torch


def niah_reward(
    generated: str,
    needle: str,
    answer: str,
    block_ids: Optional[list[int]] = None,
    evidence_blocks: Optional[list[int]] = None,
) -> dict[str, float]:
    """NIAH (Needle-In-A-Haystack) 奖励。

    Args:
        generated: 模型生成的文本
        needle: 正确答案
        answer: 标准答案
        block_ids: 路由器选中的 Block IDs
        evidence_blocks: 证据所在的 Block IDs

    Returns:
        {"correct": float, "routing": float, "total": float}
    """
    rewards = {"correct": 0.0, "routing": 0.0, "total": 0.0}

    # 正确性奖励
    if needle.lower() in generated.lower() or answer.lower() in generated.lower():
        rewards["correct"] = 1.0

    # 路由命中奖励
    if block_ids and evidence_blocks:
        hit = set(block_ids) & set(evidence_blocks)
        if hit:
            rewards["routing"] = 0.5

    # 幻觉惩罚
    if rewards["correct"] == 0.0 and len(generated) > 50:
        rewards["total"] = -1.0  # 没命中但生成很多 → 可能是幻觉

    rewards["total"] += rewards["correct"] + rewards["routing"]
    return rewards


def generate_niah_data(
    context_length: int = 100000,
    needle: str = "The secret code is CRBSA-2026.",
    num_distractors: int = 10,
) -> dict:
    """生成 NIAH 测试数据。

    在长上下文中随机插入 needle，要求模型检索。
    """
    import random

    distractor_text = (
        "The quick brown fox jumps over the lazy dog. "
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        * 100
    )

    # 构建上下文
    parts = []
    needle_pos = random.randint(context_length // 4, 3 * context_length // 4)

    pos = 0
    while pos < context_length:
        if abs(pos - needle_pos) < 200:
            parts.append(needle)
            pos += len(needle.split())
        else:
            parts.append(distractor_text[:500])
            pos += 100

    context = " ".join(parts)[:context_length]

    question = "What is the secret code mentioned in the context?"

    return {
        "context": context,
        "needle": needle,
        "question": question,
        "answer": "CRBSA-2026",
        "needle_position": needle_pos,
    }


def train_stage3(args):
    """Stage 3: GRPO 训练。"""
    from crbsa.config import CRBSAConfig
    from crbsa.models.qwen_crbsa import apply_crbsa_to_qwen3

    cfg = CRBSAConfig.from_pretrained(args.model)
    cfg.detach_router = False  # Stage 3 路由器也要学
    model = apply_crbsa_to_qwen3(args.model, cfg)
    model.unfreeze_all()
    model.enable_detach_router(False)
    model.eval()  # GRPO 中 generate 用 eval 模式

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )

    for epoch in range(args.epochs):
        for step in range(args.steps_per_epoch):
            # 生成 NIAH 数据
            niah = generate_niah_data(context_length=args.seq_len)

            # GRPO: 生成多个候选答案
            prompts = [niah["question"]] * args.grpo_group_size
            # (简化: 实际应用 tokenizer + model.generate)
            generated_texts = ["placeholder answer"] * args.grpo_group_size

            # 计算奖励
            rewards = [
                niah_reward(g, niah["needle"], niah["answer"])
                for g in generated_texts
            ]
            reward_values = [r["total"] for r in rewards]

            # GRPO 优势估计
            mean_r = sum(reward_values) / len(reward_values)
            std_r = max((sum((r - mean_r) ** 2 for r in reward_values) / len(reward_values)) ** 0.5, 1e-6)
            advantages = [(r - mean_r) / std_r for r in reward_values]

            # 更新 (简化: 实际 GRPO 需要 importance sampling ratio)
            if step % args.log_interval == 0:
                avg_reward = sum(reward_values) / len(reward_values)
                correct_rate = sum(1 for r in rewards if r["correct"] > 0) / len(rewards)
                print(
                    f"[Stage3] Epoch {epoch} Step {step} | "
                    f"avg_reward={avg_reward:.3f} "
                    f"correct_rate={correct_rate:.2%}"
                )

    model.save_pretrained(os.path.join(args.output_dir, "stage3_final"))


def main():
    parser = argparse.ArgumentParser(description="CRBSA Stage 3: GRPO RL")
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument("--output-dir", default="./outputs/stage3")
    parser.add_argument("--seq-len", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--grpo-group-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--log-interval", type=int, default=10)
    args = parser.parse_args()
    train_stage3(args)


if __name__ == "__main__":
    main()
