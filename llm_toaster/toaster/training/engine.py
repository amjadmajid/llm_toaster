"""TrainingEngine coordinates model, data, optimization, and checkpoints."""

from __future__ import annotations

import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from dataspace import DataLoaderLite

from ..config import ConfigHandler
from ..data.adapters import JsonlSFTDataLoader
from ..models.registry import build_model
from ..peft.lora import inject_lora, lora_state_dict
from ..tokenizers import build_tokenizer
from .checkpointing import load_checkpoint as load_training_checkpoint
from .checkpointing import rotate_checkpoints
from .checkpointing import save_checkpoint as save_training_checkpoint
from .metrics import (
    JsonlMetricsWriter,
    architecture_summary,
    format_architecture_summary,
    format_metrics_line,
)
from .optim import build_optimizer, build_scheduler

logger = logging.getLogger(__name__)


class TrainingEngine:
    """Small but extensible engine for pretraining and supervised finetuning."""

    def __init__(self, config: ConfigHandler | str):
        self.config = ConfigHandler.from_yaml(config) if isinstance(config, str) else config
        self.device = self.config.training.device
        self.global_step = self.config.training.training_step
        self.tokens_seen = 0
        self.best_metric = None
        self.current_shard = self.config.training.current_shard
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.train_loader = None
        self.val_loader = None
        self.last_grad_norm = 0.0
        self._configure_file_logging()

    def setup_tokenizer(self):
        self.tokenizer = build_tokenizer(self.config)
        if self.config.model.vocab_size is None:
            self.config.model.vocab_size = self.tokenizer.vocab_size
        return self.tokenizer

    def setup_model(self):
        if self.tokenizer is None:
            self.setup_tokenizer()
        self.model = build_model(self.config).to(self.device)
        if self.config.peft.enabled:
            inject_lora(self.model, self.config.peft)
        if self.config.distributed.compile and hasattr(torch, "compile"):
            self.model = torch.compile(self.model)
        return self.model

    def setup_dataloaders(self):
        if self.is_finetune_mode:
            self.train_loader = self._setup_sft_loader()
            self.val_loader = None
        else:
            self.train_loader = self._setup_pretrain_loader(split="train", current_shard=self.current_shard)
            self.val_loader = self._try_setup_validation_loader()
        return self.train_loader, self.val_loader

    def setup_optimizer(self):
        self.optimizer = build_optimizer(self.model, self.config)
        return self.optimizer

    def setup_scheduler(self):
        self.scheduler = build_scheduler(self.optimizer, self.config)
        return self.scheduler

    def setup_scaler(self):
        """Create a GradScaler that is only active for CUDA fp16 training.

        fp16 autocast needs loss scaling to avoid gradient underflow/NaNs; bf16
        and CPU paths do not, so the scaler is created disabled there and acts as
        a transparent passthrough around ``scale``/``step``/``update``.
        """
        self.scaler = _build_grad_scaler(self._use_fp16())
        return self.scaler

    def train_step(self) -> float:
        self.model.train()
        if self.scaler is None:
            self.setup_scaler()
        self.optimizer.zero_grad(set_to_none=True)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        total_loss = 0.0

        for _ in range(self.config.training.n_batches):
            x, y, shard = self.train_loader.next_batch()
            x = torch.as_tensor(x, dtype=torch.long, device=self.device)
            y = torch.as_tensor(y, dtype=torch.long, device=self.device)
            with self._autocast_context():
                logits = self.model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / self.config.training.n_batches
            self.scaler.scale(loss).backward()
            total_loss += float(loss.detach().cpu())
            self.tokens_seen += x.numel()
            self.current_shard = shard

        trainable_parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if trainable_parameters:
            self.scaler.unscale_(self.optimizer)
            self.last_grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0))
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.global_step += 1
        return total_loss

    @torch.no_grad()
    def eval_step(self) -> float | None:
        if self.val_loader is None:
            return None
        self.model.eval()
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        losses = []
        for _ in range(self.config.evaluation.eval_steps):
            x, y, _ = self.val_loader.next_batch()
            x = torch.as_tensor(x, dtype=torch.long, device=self.device)
            y = torch.as_tensor(y, dtype=torch.long, device=self.device)
            logits = self.model(x)
            losses.append(float(criterion(logits.view(-1, logits.size(-1)), y.view(-1)).cpu()))
        return sum(losses) / max(1, len(losses))

    def save_checkpoint(self, path: str | None = None, metric: float | None = None) -> None:
        checkpoint_path = path or self.default_checkpoint_path
        save_training_checkpoint(
            checkpoint_path,
            self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            config=self.config,
            global_step=self.global_step,
            tokens_seen=self.tokens_seen,
            best_metric=self.best_metric,
            data_state=self.data_state_dict(),
        )
        self._save_legacy_config_if_needed()
        if self.config.peft.enabled:
            torch.save(lora_state_dict(self.model), checkpoint_path + ".adapter")
        self._save_step_checkpoint(metric)

    def load_checkpoint(self, path: str) -> dict:
        checkpoint = load_training_checkpoint(
            path,
            self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.device,
            strict=False,
        )
        self.global_step = int(checkpoint.get("global_step", self.global_step))
        self.tokens_seen = int(checkpoint.get("tokens_seen", self.tokens_seen))
        self.best_metric = checkpoint.get("best_metric", self.best_metric)
        self.load_data_state_dict(checkpoint.get("data_state", {}))
        return checkpoint

    def train(self):
        seed_everything(self.config.training.seed)
        self.setup_tokenizer()
        self.setup_model()
        self._load_base_checkpoint_for_finetune()
        self.setup_dataloaders()
        self.setup_optimizer()
        self.setup_scheduler()
        self.setup_scaler()
        self._resume_if_requested()

        metrics = JsonlMetricsWriter(self.config.logging.metrics_file)
        summary = architecture_summary(self.model, self.config)
        for line in format_architecture_summary(summary):
            logger.info(line)
        metrics.write({"type": "architecture", **summary})

        start = time.perf_counter()
        last_log_time, last_log_tokens = start, self.tokens_seen
        log_every = max(1, self.config.logging.log_every_steps)
        try:
            while self.global_step < self.config.training.max_iter:
                loss = self.train_step()
                if self.global_step % log_every == 0:
                    now = time.perf_counter()
                    interval = now - last_log_time
                    record = {
                        "step": self.global_step,
                        "max_iter": self.config.training.max_iter,
                        "loss": loss,
                        "lr": self.scheduler.get_last_lr()[0],
                        "grad_norm": self.last_grad_norm,
                        "tokens_per_sec": (self.tokens_seen - last_log_tokens) / interval if interval > 0 else 0.0,
                        "tokens_seen": self.tokens_seen,
                        "elapsed_s": now - start,
                        "eta_s": self._eta_seconds(now - start),
                    }
                    logger.info(format_metrics_line(record))
                    metrics.write({"type": "step", **record})
                    last_log_time, last_log_tokens = now, self.tokens_seen
                metric = self._maybe_evaluate()
                self._maybe_save(metric)
        finally:
            metrics.close()
        return self

    def _eta_seconds(self, elapsed: float) -> float:
        if self.global_step <= 0:
            return 0.0
        remaining = max(0, self.config.training.max_iter - self.global_step)
        return remaining * (elapsed / self.global_step)

    @property
    def is_finetune_mode(self) -> bool:
        return self.config.training.mode in {"finetune", "sft"} or self.config.finetune.enabled

    @property
    def default_checkpoint_path(self) -> str:
        if self.is_finetune_mode:
            return self.config.finetune.output_ckpt
        return self.config.training.ckpt

    def data_state_dict(self) -> dict:
        state = {"current_shard": self.current_shard}
        if hasattr(self.train_loader, "state_dict"):
            state["train_loader"] = self.train_loader.state_dict()
        return state

    def load_data_state_dict(self, state: dict) -> None:
        self.current_shard = int(state.get("current_shard", self.current_shard))
        if self.train_loader is not None and hasattr(self.train_loader, "load_state_dict"):
            self.train_loader.load_state_dict(state.get("train_loader", {}))

    def _setup_sft_loader(self):
        return JsonlSFTDataLoader(
            self.config.training.batch_size,
            self.config.training.seq_len,
            self.config.finetune.dataset_path,
            self.tokenizer,
            fmt=self.config.data.format,
            train_on_prompt=self.config.finetune.train_on_prompt,
            seed=self.config.finetune.seed,
            shuffle=self.config.finetune.shuffle,
        )

    def _setup_pretrain_loader(self, split: str, current_shard: int = 0):
        return DataLoaderLite(
            self.config.training.batch_size,
            self.config.training.seq_len,
            current_shard,
            0,
            1,
            split,
            data_root=self.config.training.data_dir,
        )

    def _try_setup_validation_loader(self):
        if self.config.evaluation.eval_every_steps <= 0:
            return None
        try:
            return self._setup_pretrain_loader(split="val")
        except (AssertionError, FileNotFoundError) as exc:
            logger.info("Validation data unavailable; continuing without validation: %s", exc)
            return None

    def _autocast_context(self):
        enabled = "cuda" in str(self.device) and self.config.distributed.mixed_precision in {"fp16", "bf16"}
        dtype = torch.float16 if self.config.distributed.mixed_precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled)

    def _use_fp16(self) -> bool:
        return "cuda" in str(self.device) and self.config.distributed.mixed_precision == "fp16"

    def _configure_file_logging(self) -> None:
        """Route engine logs to ``logging.log_file`` via a root FileHandler.

        Idempotent: a handler is only added once per log file even when several
        engines are constructed in the same process.
        """
        log_file = self.config.logging.log_file
        if not log_file:
            return
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        target = str(path.resolve())
        root = logging.getLogger()
        for handler in root.handlers:
            if isinstance(handler, logging.FileHandler) and getattr(handler, "baseFilename", None) == target:
                return
        handler = logging.FileHandler(path)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)
        if root.level == logging.NOTSET or root.level > logging.INFO:
            root.setLevel(logging.INFO)

    def _load_base_checkpoint_for_finetune(self) -> None:
        if not self.is_finetune_mode or not self.config.finetune.base_ckpt:
            return
        try:
            load_training_checkpoint(self.config.finetune.base_ckpt, self.model, device=self.device, strict=False)
            logger.info("Loaded base checkpoint for finetuning: %s", self.config.finetune.base_ckpt)
        except FileNotFoundError:
            logger.warning("Base checkpoint not found: %s", self.config.finetune.base_ckpt)

    def _resume_if_requested(self) -> None:
        checkpoint = self.config.checkpointing.resume_from_checkpoint
        if checkpoint:
            self.load_checkpoint(checkpoint)

    def _maybe_evaluate(self) -> float | None:
        if self.val_loader is None:
            return None
        if self.global_step % max(1, self.config.evaluation.eval_every_steps) != 0:
            return None
        metric = self.eval_step()
        if metric is not None:
            logger.info("step %s validation loss %.4f (perplexity %.2f)", self.global_step, metric, perplexity(metric))
            if self.best_metric is None or metric < self.best_metric:
                self.best_metric = metric
                if self.config.checkpointing.save_best:
                    self.save_checkpoint(str(Path(self.config.checkpointing.output_dir) / "best.pt"), metric=metric)
        return metric

    def _maybe_save(self, metric: float | None) -> None:
        should_save = self.global_step % self.config.checkpointing.save_every_steps == 0
        is_last_step = self.global_step >= self.config.training.max_iter
        if should_save or is_last_step:
            self.save_checkpoint(metric=metric)

    def _save_step_checkpoint(self, metric: float | None) -> None:
        output_dir = Path(self.config.checkpointing.output_dir)
        step_path = output_dir / f"step_{self.global_step}.pt"
        if str(step_path) == self.default_checkpoint_path:
            return
        save_training_checkpoint(
            str(step_path),
            self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            config=self.config,
            global_step=self.global_step,
            tokens_seen=self.tokens_seen,
            best_metric=self.best_metric if metric is None else metric,
            data_state=self.data_state_dict(),
        )
        rotate_checkpoints(str(output_dir), save_total_limit=self.config.checkpointing.save_total_limit)

    def _save_legacy_config_if_needed(self) -> None:
        config_path = self.config.finetune.output_config if self.is_finetune_mode else self.config.training.ckpt_config
        if config_path:
            self.config.to_yaml(config_path)


def _build_grad_scaler(enabled: bool):
    """Construct a GradScaler across torch versions (new ``torch.amp`` API first)."""
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - older torch fallback
        return torch.cuda.amp.GradScaler(enabled=enabled)


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and torch (incl. CUDA) RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def perplexity(loss: float) -> float:
    """Convert a cross-entropy loss to perplexity, capped to avoid overflow."""
    return math.exp(loss) if loss < 100 else float("inf")


def model_size_summary(model) -> str:
    """One-line parameter summary; fp32 weight memory since weights are held in fp32."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return (
        f"{total / 1e6:.1f}M params ({total:,} total | {trainable:,} trainable), ~{total * 4 / 1e6:.0f} MB fp32 weights"
    )
