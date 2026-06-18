"""Optimizer and scheduler factories."""

from __future__ import annotations

import math

import torch


def parameter_groups(model, weight_decay: float) -> list[dict]:
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith("bias"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(model, config):
    optimizer_config = config.optimizer
    fused = optimizer_config.fused or optimizer_config.name == "fused_adamw"
    kwargs = {"lr": optimizer_config.lr, "betas": optimizer_config.betas}
    if fused and "fused" in torch.optim.AdamW.__init__.__code__.co_varnames:
        kwargs["fused"] = True
    return torch.optim.AdamW(parameter_groups(model, optimizer_config.weight_decay), **kwargs)


def build_scheduler(optimizer, config):
    if config.scheduler.name == "constant":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    if config.scheduler.name == "cosine":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, _cosine_lambda(config))
    raise ValueError(f"Unknown scheduler {config.scheduler.name!r}")


def _cosine_lambda(config):
    warmup_steps = config.scheduler.warmup_steps
    max_steps = config.scheduler.max_steps or config.training.max_iter
    min_lr_ratio = config.scheduler.min_lr_ratio

    def schedule(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = min(1.0, (step - warmup_steps) / max(1, max_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return schedule
