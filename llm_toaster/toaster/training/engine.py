"""TrainingEngine coordinates model, data, optimization, and checkpoints."""

from __future__ import annotations

import logging
import math
import random
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..config import ConfigHandler
from ..data.adapters import JsonlSFTDataLoader
from ..data.errors import DataExhausted
from ..data.protocol import PretrainBatchInfo
from ..data.sources import build_pretrain_train_source, build_validation_source
from ..models.registry import build_model
from ..peft.lora import inject_lora, lora_state_dict
from ..tokenizers import build_tokenizer
from .checkpointing import git_commit, rotate_checkpoints
from .checkpointing import load_checkpoint as load_training_checkpoint
from .checkpointing import save_checkpoint as save_training_checkpoint
from .metrics import (
    CsvMetricsWriter,
    JsonlMetricsWriter,
    architecture_summary,
    compute_mfu,
    format_architecture_summary,
    format_metrics_line,
    human_count,
)
from .optim import build_optimizer, build_scheduler

logger = logging.getLogger(__name__)


class ResumeCheckpointNotFoundError(FileNotFoundError):
    """Resume was requested (e.g. ``trainer.py -ct``) but the checkpoint file is absent.

    Carries an actionable message; the CLI (``trainer.py``) catches it to exit cleanly
    rather than dumping a traceback.
    """


class TrainingEngine:
    """Small but extensible engine for pretraining and supervised finetuning."""

    def __init__(self, config: ConfigHandler | str):
        self.config = ConfigHandler.from_yaml(config) if isinstance(config, str) else config
        self.device = self.config.training.device
        self.global_step = self.config.training.training_step
        self.tokens_seen = 0
        self.unique_tokens_seen = 0
        self.repeated_tokens_seen = 0
        self.data_pass = 0
        self.current_shard_id = None
        self.data_wait_s = 0.0
        self.best_metric = None
        self.current_shard = self.config.training.current_shard
        self.coordinator = None
        self.max_steps = self.config.training.max_iter
        self._data_exhausted = False
        # Cumulative training seconds across resumes (elapsed time, not wall-clock timestamps).
        self.wall_clock_s = float(self.config.training.training_duration)
        self._run_start = None
        self._original_signal_handlers = {}
        self._stop_requested = False
        self._stop_signum = None
        self.last_val_loss = None
        self.resumed = False
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
            # build_pretrain_train_source may start a background prefetch producer; the engine owns
            # its lifecycle and stops it in train()'s finally block.
            self.train_loader, self.coordinator = build_pretrain_train_source(self.config, self.tokenizer)
            self.val_loader = build_validation_source(self.config)
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
        """Run one optimizer step over ``n_batches`` micro-batches.

        Token accounting is committed only after a *full* step succeeds, so a ``DataExhausted``
        mid gradient-accumulation discards the partial step cleanly (zero grads, no optimizer/step
        advance, no token counting) and propagates to the loop, which stops at this boundary.
        """
        self.model.train()
        if self.scaler is None:
            self.setup_scaler()
        self.optimizer.zero_grad(set_to_none=True)
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        total_loss = 0.0
        step_tokens = step_unique = step_repeated = 0
        last_info = None

        for _ in range(self.config.training.n_batches):
            try:
                x, y, info = self.train_loader.next_batch()
            except DataExhausted:
                # Discard the incomplete step; do not advance the optimizer or global step.
                self.optimizer.zero_grad(set_to_none=True)
                logger.warning(
                    "data exhausted mid gradient-accumulation at step %d; discarding the incomplete step",
                    self.global_step,
                )
                raise
            x = torch.as_tensor(x, dtype=torch.long, device=self.device)
            y = torch.as_tensor(y, dtype=torch.long, device=self.device)
            with self._autocast_context():
                logits = self.model(x)
                loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / self.config.training.n_batches
            self.scaler.scale(loss).backward()
            total_loss += float(loss.detach().cpu())
            tokens = int(x.numel())
            step_tokens += tokens
            if isinstance(info, PretrainBatchInfo) and info.repeated:
                step_repeated += tokens
            else:
                step_unique += tokens
            last_info = info

        trainable_parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        if trainable_parameters:
            self.scaler.unscale_(self.optimizer)
            self.last_grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0))
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.global_step += 1
        # Commit accounting now that the full step succeeded.
        self.tokens_seen += step_tokens
        self.unique_tokens_seen += step_unique
        self.repeated_tokens_seen += step_repeated
        if isinstance(last_info, PretrainBatchInfo):
            self.data_pass = last_info.pass_index
            self.current_shard_id = last_info.shard_id
        return total_loss

    @torch.no_grad()
    def eval_step(self) -> float | None:
        """Average loss over the fixed validation set.

        By default the validation cursor is reset before each eval so successive evaluations see
        the *same* examples (comparable val curves). Uses the same autocast as training. The train
        cursor is never touched. A fixed val set smaller than ``eval_steps`` simply stops early.
        """
        if self.val_loader is None:
            return None
        validation = self.config.data.validation
        if validation.reset_each_eval and not validation.sequential:
            try:
                self.val_loader.reset()
            except Exception:  # pragma: no cover - reset is best effort for non-resettable loaders
                pass
        self.model.eval()
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        losses = []
        for _ in range(self.config.evaluation.eval_steps):
            try:
                x, y, _ = self.val_loader.next_batch()
            except DataExhausted:
                break  # fixed validation smaller than eval_steps -> evaluate on what exists
            x = torch.as_tensor(x, dtype=torch.long, device=self.device)
            y = torch.as_tensor(y, dtype=torch.long, device=self.device)
            with self._autocast_context():
                logits = self.model(x)
                losses.append(float(criterion(logits.view(-1, logits.size(-1)), y.view(-1)).cpu()))
        return sum(losses) / max(1, len(losses)) if losses else None

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
            wall_clock_s=self._elapsed_seconds_total(),
            tokenizer_info=self._tokenizer_info(),
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
        self.wall_clock_s = float(checkpoint.get("wall_clock_s", self.wall_clock_s))
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
        self._compute_step_budget()

        self._save_legacy_config_if_needed()  # write a resolved-config snapshot up front
        metrics = JsonlMetricsWriter(self.config.logging.metrics_file)
        csv_metrics = CsvMetricsWriter(self._metrics_csv_path())
        summary = architecture_summary(self.model, self.config)
        for line in format_architecture_summary(summary):
            logger.info(line)
        for line in self._data_summary_lines():
            logger.info(line)
        metrics.write(
            {
                "type": "architecture",
                **summary,
                **self._data_metrics(),
                "max_steps": self.max_steps,
                "max_tokens": self.config.training.max_tokens,
                "tokens_per_step": self._tokens_per_step(),
                "git_commit": git_commit(),
                "config_path": self._resolved_config_path(),
                "resumed": self.resumed,
            }
        )

        if "cuda" in str(self.device):
            torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        self._run_start = start
        last_log_time, last_log_tokens, last_log_step = start, self.tokens_seen, self.global_step
        log_every = max(1, self.config.logging.log_every_steps)
        self._install_signal_handlers()
        try:
            while self.global_step < self.max_steps:
                if self._token_budget_reached():
                    logger.info(
                        "token budget reached (%d tokens, ~%d/step); stopping at step %d",
                        self.config.training.max_tokens,
                        self._tokens_per_step(),
                        self.global_step,
                    )
                    self._data_exhausted = True
                    break
                try:
                    loss = self.train_step()
                except DataExhausted as exc:
                    logger.warning(
                        "unique data exhausted after %d pass(es); ending training at step %d",
                        exc.pass_index,
                        self.global_step,
                    )
                    self._data_exhausted = True
                    break
                self._update_consumer_cursor()
                metric = self._maybe_evaluate()  # before logging so the latest val_loss is recorded
                if self.global_step % log_every == 0:
                    now = time.perf_counter()
                    interval = now - last_log_time
                    steps_since = max(1, self.global_step - last_log_step)
                    tokens_per_sec = (self.tokens_seen - last_log_tokens) / interval if interval > 0 else 0.0
                    elapsed_total = self._elapsed_seconds_total()
                    record = {
                        "step": self.global_step,
                        "max_iter": self.max_steps,
                        "loss": loss,
                        "lr": self.scheduler.get_last_lr()[0],
                        "grad_norm": self.last_grad_norm,
                        "tokens_per_sec": tokens_per_sec,
                        "tokens_seen": self.tokens_seen,
                        "unique_tokens_seen": self.unique_tokens_seen,
                        "repeated_tokens_seen": self.repeated_tokens_seen,
                        "val_loss": self.last_val_loss,
                        "val_perplexity": (perplexity(self.last_val_loss) if self.last_val_loss is not None else None),
                        "iter_time_ms": (interval / steps_since) * 1000.0,
                        "elapsed_s": elapsed_total,
                        "eta_s": self._eta_seconds(elapsed_total),
                        "mfu": compute_mfu(
                            summary["flops_per_token"], tokens_per_sec, self.config.logging.device_peak_flops
                        ),
                        "peak_mem_bytes": self._peak_mem_bytes(),
                        "peak_mem_reserved_bytes": self._peak_mem_reserved_bytes(),
                        "resumed": self.resumed,
                        **self._data_metrics(),
                    }
                    logger.info(format_metrics_line(record))
                    metrics.write({"type": "step", **record})
                    csv_metrics.write(record)
                    last_log_time, last_log_tokens, last_log_step = now, self.tokens_seen, self.global_step
                self._maybe_save(metric)
                if self._stop_requested:  # interrupt requested -> stop at this clean step boundary
                    break
        except KeyboardInterrupt:
            # Forced interrupt (a second Ctrl-C, or a context where no handler was installed).
            self._stop_requested = True
            logger.warning("training interrupted; saving emergency checkpoint")
        finally:
            if self._stop_requested:
                try:
                    self.save_checkpoint(self._emergency_checkpoint_path())
                except Exception:  # pragma: no cover - emergency save is best effort
                    logger.exception("emergency checkpoint failed")
            elif self._data_exhausted:
                # Clean stop at a step boundary (exhaustion or token budget): save a final checkpoint.
                try:
                    self.save_checkpoint()
                except Exception:  # pragma: no cover - best effort
                    logger.exception("final checkpoint failed")
            self.wall_clock_s = self._elapsed_seconds_total()
            self._run_start = None
            self._restore_signal_handlers()
            self._close_data_sources()
            metrics.close()
            csv_metrics.close()
        return self

    def _compute_step_budget(self) -> None:
        """Effective max optimizer steps = min(max_iter, floor(max_tokens / tokens_per_step))."""
        max_iter = self.config.training.max_iter
        max_tokens = self.config.training.max_tokens
        if max_tokens is None:
            self.max_steps = max_iter
            return
        steps_from_tokens = max_tokens // max(1, self._tokens_per_step())
        self.max_steps = min(max_iter, int(steps_from_tokens))

    def _tokens_per_step(self) -> int:
        return self.config.training.batch_size * self.config.model.seq_len * self.config.training.n_batches

    def _token_budget_reached(self) -> bool:
        """True when starting another full step would exceed ``training.max_tokens``."""
        max_tokens = self.config.training.max_tokens
        if max_tokens is None:
            return False
        return self.tokens_seen + self._tokens_per_step() > max_tokens

    def _update_consumer_cursor(self) -> None:
        """Tell the prefetch producer how many train shards we've consumed (bounds the queue)."""
        if self.coordinator is None:
            return
        consumed = getattr(self.train_loader, "consumed_shard_count", None)
        if callable(consumed):
            try:
                self.coordinator.update_consumer(consumed())
            except Exception:  # pragma: no cover - advisory bookkeeping
                pass

    def _close_data_sources(self) -> None:
        for loader in (self.train_loader, self.val_loader):
            closer = getattr(loader, "close", None)
            if callable(closer):
                try:
                    closer()
                except Exception:  # pragma: no cover - best effort
                    pass
        if self.coordinator is not None:
            try:
                self.coordinator.stop()
            except Exception:  # pragma: no cover - best effort
                logger.exception("prefetch coordinator stop failed")

    def _data_metrics(self) -> dict:
        """Low-cardinality data fields for metrics records (no token buffers / large values)."""
        stats = {}
        getter = getattr(self.train_loader, "stats", None)
        if callable(getter):
            try:
                stats = getter()
            except Exception:  # pragma: no cover
                stats = {}
        producer = self._producer_metrics()
        return {
            "data_mode": self.config.data.materialization.mode,
            "dataset_id": stats.get("dataset_id"),
            "dataset_fingerprint": stats.get("dataset_fingerprint"),
            "manifest_generation": stats.get("manifest_generation"),
            "current_shard_id": stats.get("current_shard_id", self.current_shard_id),
            "data_pass": self.data_pass,
            "unique_tokens_seen": self.unique_tokens_seen,
            "repeated_tokens_seen": self.repeated_tokens_seen,
            "data_wait_s": stats.get("data_wait_s", self.data_wait_s),
            "source_records_consumed": stats.get("source_records_consumed"),
            **producer,
        }

    def _producer_metrics(self) -> dict:
        if self.coordinator is None:
            return {}
        status = self.coordinator.status()
        return {
            "producer_status": status.get("status"),
            "producer_queue_depth": status.get("queue_depth"),
        }

    def _data_summary_lines(self) -> list[str]:
        """Concise startup data summary (mode, dataset, budget, exhaustion)."""
        data = self.config.data
        tps = self._tokens_per_step()
        budget = self.config.training.max_tokens or (self.max_steps * tps)
        stats = self._data_metrics()
        dataset = stats.get("dataset_id") or ("legacy-dir" if self._is_legacy_data() else "(pending)")
        lines = [
            f"data: mode={data.materialization.mode} | dataset={dataset} | exhaustion={data.sampling.exhaustion}",
        ]
        if stats.get("manifest_generation") is not None:
            lines.append(f"manifest: generation={stats['manifest_generation']}")
        lines.append(f"budget: {human_count(budget)} tokens | {tps:,} tokens/step | up to {self.max_steps:,} steps")
        if data.materialization.mode == "prefetch":
            lines.append(f"exhaustion: wait | prefetch queue target={data.materialization.prefetch_shards} shards")
        return lines

    def _is_legacy_data(self) -> bool:
        return getattr(self.config, "_data_is_legacy", False) and not self.is_finetune_mode

    def _eta_seconds(self, elapsed: float) -> float:
        if self.global_step <= 0:
            return 0.0
        remaining = max(0, self.max_steps - self.global_step)
        return remaining * (elapsed / self.global_step)

    def _peak_mem_bytes(self) -> int:
        return int(torch.cuda.max_memory_allocated()) if "cuda" in str(self.device) else 0

    def _peak_mem_reserved_bytes(self) -> int:
        return int(torch.cuda.max_memory_reserved()) if "cuda" in str(self.device) else 0

    def _metrics_csv_path(self) -> str | None:
        """CSV metrics path: explicit ``logging.metrics_csv`` ('' disables), else derived from the
        JSONL path so it lands next to it (works for per-run sweep directories too)."""
        explicit = self.config.logging.metrics_csv
        if explicit is not None:
            return explicit or None
        jsonl = self.config.logging.metrics_file
        return str(Path(jsonl).with_suffix(".csv")) if jsonl else None

    def _resolved_config_path(self) -> str:
        return self.config.finetune.output_config if self.is_finetune_mode else self.config.training.ckpt_config

    def _elapsed_seconds_total(self) -> float:
        """Cumulative training seconds across resumes (perf_counter deltas, never timestamps)."""
        if self._run_start is None:
            return self.wall_clock_s
        return self.wall_clock_s + (time.perf_counter() - self._run_start)

    def _tokenizer_info(self) -> dict:
        """Special-token ids + vocab size, persisted in the checkpoint for resume validation."""
        if self.tokenizer is None:
            return {}
        return {
            "eos_token_id": getattr(self.tokenizer, "eos_token_id", None),
            "bos_token_id": getattr(self.tokenizer, "bos_token_id", None),
            "pad_token_id": getattr(self.tokenizer, "pad_token_id", None),
            "vocab_size": getattr(self.tokenizer, "vocab_size", None),
        }

    def _emergency_checkpoint_path(self) -> str:
        return str(Path(self.config.checkpointing.output_dir) / "emergency.pt")

    def _install_signal_handlers(self) -> None:
        """Catch SIGINT/SIGTERM so an interrupted long run saves an emergency checkpoint first.

        signal handlers can only be set from the main thread; in any other context (some test
        runners, worker threads) this is skipped and training proceeds without the safety net.
        """
        self._original_signal_handlers = {}
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._original_signal_handlers[sig] = signal.signal(sig, self._handle_interrupt)
            except (ValueError, OSError, RuntimeError):
                self._original_signal_handlers.pop(sig, None)

    def _restore_signal_handlers(self) -> None:
        for sig, handler in self._original_signal_handlers.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError, RuntimeError):
                pass
        self._original_signal_handlers = {}

    def _handle_interrupt(self, signum, _frame):
        """Request a graceful stop. Runs in async-signal context, so it only sets a flag -- the
        loop then saves a *consistent* emergency checkpoint at the next step boundary rather than
        risking a save mid-optimizer-step. Restoring the handlers means a second interrupt hits the
        default handler and force-quits immediately."""
        self._stop_requested = True
        self._stop_signum = signum
        logger.warning(
            "received signal %s; saving an emergency checkpoint at the next step boundary "
            "(interrupt again to force-quit)",
            signum,
        )
        self._restore_signal_handlers()

    @property
    def is_finetune_mode(self) -> bool:
        return self.config.training.mode in {"finetune", "sft"} or self.config.finetune.enabled

    @property
    def default_checkpoint_path(self) -> str:
        if self.is_finetune_mode:
            return self.config.finetune.output_ckpt
        return self.config.training.ckpt

    def data_state_dict(self) -> dict:
        """Resume state: engine token/pass counters + the source's own cursor/identity state."""
        state = {
            "current_shard": self.current_shard,
            "unique_tokens_seen": self.unique_tokens_seen,
            "repeated_tokens_seen": self.repeated_tokens_seen,
            "data_pass": self.data_pass,
        }
        if hasattr(self.train_loader, "state_dict"):
            state["train_loader"] = self.train_loader.state_dict()
        return state

    def load_data_state_dict(self, state: dict) -> None:
        self.current_shard = int(state.get("current_shard", self.current_shard))
        self.unique_tokens_seen = int(state.get("unique_tokens_seen", self.unique_tokens_seen))
        self.repeated_tokens_seen = int(state.get("repeated_tokens_seen", self.repeated_tokens_seen))
        self.data_pass = int(state.get("data_pass", self.data_pass))
        if self.train_loader is not None and hasattr(self.train_loader, "load_state_dict"):
            # The source validates dataset identity/revision/tokenizer/transform/shard checksum
            # and the committed manifest prefix here, raising ResumeIncompatibleError on a mismatch.
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
        if not checkpoint:
            return
        if Path(checkpoint).exists():
            self.load_checkpoint(checkpoint)
            self.resumed = True
            return
        self._handle_missing_resume_checkpoint(checkpoint)

    @staticmethod
    def _stdin_is_interactive() -> bool:
        """True only when stdin can answer a prompt. False in Colab ``!python`` cells, CI,
        pytest capture, or nohup -- where ``input()`` raises instead of blocking for an answer."""
        try:
            return bool(sys.stdin) and sys.stdin.isatty()
        except (ValueError, OSError):
            return False

    def _handle_missing_resume_checkpoint(self, checkpoint: str) -> None:
        """Resume was requested but the checkpoint isn't there.

        Common on the very first run, or on a fresh Colab VM whose earlier session never
        reached the first save. Always warn. On an interactive terminal, offer to start fresh;
        otherwise raise so the CLI exits with an actionable message instead of silently
        restarting a long run the user thought they were resuming.
        """
        logger.warning(
            "Resume requested but no checkpoint exists at %s. This is expected on the first run, "
            "or on a fresh Colab VM whose earlier session never reached the first save.",
            checkpoint,
        )
        if self._stdin_is_interactive():
            reply = input(f"Start a fresh run (no resume) and create {checkpoint} at the first save? [y/N] ")
            if reply.strip().lower() in {"y", "yes"}:
                logger.warning("Starting a fresh run; %s will be written at the first save.", checkpoint)
                return
        raise ResumeCheckpointNotFoundError(
            f"No checkpoint to resume from at {checkpoint}. To start fresh, re-run without -ct "
            f"(or clear checkpointing.resume_from_checkpoint); the checkpoint is created at the first "
            f"save. To resume instead, point the resume path at a checkpoint that already exists."
        )

    def _maybe_evaluate(self) -> float | None:
        if self.val_loader is None:
            return None
        if self.global_step % max(1, self.config.evaluation.eval_every_steps) != 0:
            return None
        metric = self.eval_step()
        if metric is not None:
            self.last_val_loss = metric
            logger.info("step %s validation loss %.4f (perplexity %.2f)", self.global_step, metric, perplexity(metric))
            if self.best_metric is None or metric < self.best_metric:
                self.best_metric = metric
                if self.config.checkpointing.save_best:
                    self.save_checkpoint(str(Path(self.config.checkpointing.output_dir) / "best.pt"), metric=metric)
        return metric

    def _maybe_save(self, metric: float | None) -> None:
        should_save = self.global_step % self.config.checkpointing.save_every_steps == 0
        is_last_step = self.global_step >= self.max_steps
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
            wall_clock_s=self._elapsed_seconds_total(),
            tokenizer_info=self._tokenizer_info(),
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
