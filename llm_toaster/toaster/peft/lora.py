"""Minimal local LoRA implementation with no hard PEFT dependency."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wrap a Linear layer with trainable low-rank adapters."""

    def __init__(self, base: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.05):
        super().__init__()
        if r <= 0:
            raise ValueError("LoRA rank r must be positive")
        self.base = base
        self.r = r
        self.scaling = alpha / r
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.base.weight.requires_grad = False
        if self.base.bias is not None:
            self.base.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adapter = self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()
        return self.base(x) + adapter * self.scaling

    def merge(self) -> nn.Linear:
        self.base.weight.data += (self.lora_B @ self.lora_A) * self.scaling
        return self.base


def inject_lora(model: nn.Module, config) -> nn.Module:
    """Replace target Linear modules with LoRA wrappers and freeze base weights."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    replaced = _inject_recursive(model, set(config.target_modules), config)
    if replaced == 0:
        raise ValueError(f"No LoRA target modules matched {config.target_modules}")
    return model


def lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: value for name, value in model.state_dict().items() if "lora_" in name}


def merge_lora(model: nn.Module) -> nn.Module:
    for module in model.modules():
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                setattr(module, name, child.merge())
    return model


def _inject_recursive(module: nn.Module, targets: set[str], config) -> int:
    count = 0
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in targets:
            setattr(module, name, LoRALinear(child, config.r, config.alpha, config.dropout))
            count += 1
        else:
            count += _inject_recursive(child, targets, config)
    return count
