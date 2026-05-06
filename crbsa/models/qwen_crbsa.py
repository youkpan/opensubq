"""Qwen3.6-35B-A3B (MoE) 适配 CRBSA。

替换 Qwen3 的 Attention 层为 CRBSAAttention。
保留 MoE MLP 层不变。
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn

from crbsa.config import CRBSAConfig
from crbsa.nn.crbsa_layer import CRBSAAttention


def apply_crbsa_to_qwen3(
    model_name_or_path: str = "Qwen/Qwen3.6-35B-A3B",
    crbsa_config: Optional[CRBSAConfig] = None,
    **config_overrides,
) -> "Qwen3CRBSAForCausalLM":
    """加载 Qwen3 模型并替换 Attention 为 CRBSA。

    Args:
        model_name_or_path: HuggingFace 模型名或本地路径
        crbsa_config: CRBSA 配置 (None 则自动从模型提取)
        **config_overrides: 覆盖 CRBSAConfig 字段

    Returns:
        Qwen3CRBSAForCausalLM 实例
    """
    from transformers import AutoModelForCausalLM, AutoConfig

    if crbsa_config is None:
        crbsa_config = CRBSAConfig.from_pretrained(model_name_or_path, **config_overrides)
    else:
        for k, v in config_overrides.items():
            setattr(crbsa_config, k, v)

    # 加载原始模型
    hf_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        config=hf_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    return Qwen3CRBSAForCausalLM(model, crbsa_config)


class Qwen3CRBSAForCausalLM(nn.Module):
    """Qwen3 + CRBSA 包装器。

    保留 Qwen3 的:
      - Embedding
      - MoE MLP layers
      - RMSNorm
      - LM Head

    替换:
      - Attention layers → CRBSAAttention
    """

    def __init__(self, hf_model, crbsa_config: CRBSAConfig):
        super().__init__()
        self.config = crbsa_config
        self.hf_model = hf_model

        # 替换 Attention 层
        self._crbsa_layers = nn.ModuleList()
        self._replaced_indices = []

        num_layers = crbsa_config.num_hidden_layers
        for i in range(num_layers):
            crbsa_layer = CRBSAAttention(crbsa_config, layer_id=i)

            # 将原始 Qwen3 的 QKV/O 权重复制到 CRBSA 层
            original_attn = hf_model.model.layers[i].self_attn
            self._copy_attn_weights(crbsa_layer, original_attn)

            self._crbsa_layers.append(crbsa_layer)
            self._replaced_indices.append(i)

            # 替换 forward 方法为 CRBSA 版本
            original_attn.forward = self._make_crbsa_forward(i, original_attn)

    def _copy_attn_weights(self, crbsa_layer: CRBSAAttention, original_attn: nn.Module):
        """从 Qwen3 复制 Q/K/V/O 权重到 CRBSA 层。"""
        with torch.no_grad():
            if hasattr(original_attn, "q_proj"):
                crbsa_layer.q_proj.weight.copy_(original_attn.q_proj.weight)
                crbsa_layer.k_proj.weight.copy_(original_attn.k_proj.weight)
                crbsa_layer.v_proj.weight.copy_(original_attn.v_proj.weight)
                crbsa_layer.o_proj.weight.copy_(original_attn.o_proj.weight)

    def _make_crbsa_forward(self, layer_idx: int, original_attn: nn.Module):
        """为原始 Attention 层创建 CRBSA forward 替代。"""
        crbsa_layer = self._crbsa_layers[layer_idx]

        def crbsa_forward(
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_value=None,
            output_attentions=False,
            use_cache=False,
            position_embeddings=None,
            **kwargs,
        ):
            output, attn_weights, losses = crbsa_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                output_attentions=output_attentions,
            )
            # 将 losses 挂到 layer 上供外部收集
            original_attn._crbsa_losses = losses
            return output, attn_weights, past_key_value

        return crbsa_forward

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> dict:
        """前向传播。

        Returns:
            {"loss": ..., "logits": ..., "balance_loss": ..., ...}
        """
        outputs = self.hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            labels=labels,
            **kwargs,
        )

        result = {
            "logits": outputs.logits,
        }
        if outputs.loss is not None:
            result["loss"] = outputs.loss

        # 收集所有层的 balance loss
        total_balance = torch.tensor(0.0, device=input_ids.device)
        for layer in self._crbsa_layers:
            if hasattr(layer, "_crbsa_losses"):
                pass  # losses 在 forward 时收集
        # 从 replaced layers 收集
        for i, idx in enumerate(self._replaced_indices):
            attn = self.hf_model.model.layers[idx].self_attn
            if hasattr(attn, "_crbsa_losses"):
                total_balance = total_balance + attn._crbsa_losses.get("balance_loss", 0.0)

        result["balance_loss"] = total_balance
        return result

    def generate(self, **kwargs):
        """代理到原始模型的 generate。"""
        return self.hf_model.generate(**kwargs)

    def get_routing_losses(self) -> list[torch.Tensor]:
        """获取所有层的路由损失。"""
        losses = []
        for i in self._replaced_indices:
            attn = self.hf_model.model.layers[i].self_attn
            if hasattr(attn, "_crbsa_losses"):
                bl = attn._crbsa_losses.get("balance_loss")
                if bl is not None:
                    losses.append(bl)
        return losses

    def freeze_backbone(self):
        """Stage 1: 冻结主干，只训练路由器。"""
        for param in self.hf_model.parameters():
            param.requires_grad = False
        for layer in self._crbsa_layers:
            # 只解冻路由相关参数
            layer.router.requires_grad_(True)
            layer.summarizer.requires_grad_(True)

    def unfreeze_all(self):
        """Stage 2: 解冻所有参数。"""
        for param in self.hf_model.parameters():
            param.requires_grad = True
        for param in self.parameters():
            param.requires_grad = True

    def enable_detach_router(self, enable: bool = True):
        """Stage 2: 启用路由器梯度截断。"""
        self.config.detach_router = enable
        for layer in self._crbsa_layers:
            layer.config.detach_router = enable

    def save_pretrained(self, path: str):
        """保存模型。"""
        self.hf_model.save_pretrained(path)
        self.config.save(path + "/crbsa_config.json")

    @property
    def device(self):
        return next(self.hf_model.parameters()).device
