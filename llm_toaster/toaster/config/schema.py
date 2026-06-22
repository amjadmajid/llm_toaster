"""Configuration objects for the modular LLM Toaster engine.

The schema intentionally uses dataclasses instead of a heavier validation
framework so the project stays easy to read.  Validation is centralized in
``ConfigHandler.validate`` and legacy fields from the original repo are mirrored
into the newer sections for backward compatibility.
"""

from __future__ import annotations

import warnings
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Optional

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - fallback for very small envs
    yaml = None


class ConfigError(ValueError):
    """Raised for malformed config files, with file + offending key context."""


@dataclass
class ModelConfig:
    architecture: str = "decoder_transformer"
    vocab_size: Optional[int] = None
    n_embd: int = 768
    n_head: int = 8
    n_blocks: int = 16
    seq_len: int = 1024
    dropout_rate: float = 0.2
    norm: str = "layernorm"  # layernorm | rmsnorm
    ffn: str = "gelu"  # gelu | geglu | swiglu | moe placeholder
    ffn_mult: int = 4  # FFN hidden expansion ratio (hidden = ffn_mult * n_embd)
    position: str = "learned"  # learned | rope | none
    num_key_value_heads: Optional[int] = None
    tie_embeddings: bool = True


@dataclass
class TokenizerConfig:
    type: str = "tiktoken"  # tiktoken | hf | sentencepiece
    name: str = "gpt2"
    path: Optional[str] = None


@dataclass
class DataSourceConfig:
    """Where raw documents come from (orthogonal to how they are materialized)."""

    type: str = "huggingface"  # huggingface | local
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    config_name: Optional[str] = "sample-10BT"
    split: str = "train"
    revision: Optional[str] = None  # branch/tag/sha; resolved to an immutable commit when preparing
    text_field: str = "text"


@dataclass
class DataTransformConfig:
    """Deterministic tokenization + packing applied to produce shards."""

    add_eot: bool = True
    packing: str = "contiguous"  # contiguous (only supported mode)
    shard_tokens: int = int(1e7)
    dtype: str = "uint16"  # uint16 | uint32 | int32


@dataclass
class MaterializationConfig:
    """How shards are produced/consumed: prepared (upfront), prefetch (background), direct (stream)."""

    mode: str = "prepared"  # prepared | prefetch | direct
    store_dir: Optional[str] = None  # persistent shard store (manifest + shards/) when no manifest_path
    cache_dir: Optional[str] = None  # optional local read-through cache (does not change manifest identity)
    prefetch_shards: int = 3  # bound on sealed-but-unconsumed train shards (prefetch)
    min_ready_shards: int = 1  # train shards required before training starts (prefetch)
    wait_timeout_s: float = 300.0  # max wait for a new shard / producer readiness
    tokenizer_workers: int = 1  # one worker by default for correctness + resume safety
    retain_consumed: bool = True  # keep consumed shards (needed for exact resume)
    external_producer: bool = False  # set when a producer outside this process feeds a wait-mode loader


@dataclass
class SamplingConfig:
    """How batches are sampled from the materialized data and what happens at exhaustion."""

    seed: int = 1337
    exhaustion: str = "stop"  # stop | repeat | wait
    shuffle: str = "none"  # none | shard


@dataclass
class DataValidationConfig:
    """Fixed, reproducible validation data (never grows during training)."""

    manifest_path: Optional[str] = None  # use a separate validation manifest if set
    tokens: Optional[int] = None  # size validation by token budget, or...
    shards: Optional[int] = None  # ...by shard count (exactly one when generating validation)
    reset_each_eval: bool = True  # reset the val cursor before each eval so evals are comparable
    sequential: bool = False  # opt in to walking val sequentially instead of resetting


@dataclass
class DataConfig:
    # Canonical manifest-backed pretraining data (new, orthogonal sub-sections).
    manifest_path: Optional[str] = None
    source: DataSourceConfig = field(default_factory=DataSourceConfig)
    transform: DataTransformConfig = field(default_factory=DataTransformConfig)
    materialization: MaterializationConfig = field(default_factory=MaterializationConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    validation: DataValidationConfig = field(default_factory=DataValidationConfig)
    # SFT / generic data fields.
    train_path: Optional[str] = None
    eval_path: Optional[str] = None
    format: str = "auto"
    train_on_prompt: bool = False
    # --- Legacy flat fields (deprecated; translated by apply_backward_compatibility) ---
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    split_ratio: float = 0.98
    tokenized_data: str = "dataspace/fineweb"
    remote_name: str = "sample-10BT"
    shard_size: int = int(1e7)


@dataclass
class OptimizerConfig:
    name: str = "adamw"  # adamw | fused_adamw
    lr: float = 0.0006
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    fused: bool = False

    @property
    def betas(self) -> tuple[float, float]:
        return (self.beta1, self.beta2)


@dataclass
class SchedulerConfig:
    name: str = "constant"  # constant | cosine
    warmup_steps: int = 0
    max_steps: Optional[int] = None
    min_lr_ratio: float = 0.1


@dataclass
class AttentionConfig:
    backend: str = "sdpa"  # eager | sdpa | sdpa_auto | sdpa_flash | sdpa_mem_efficient | sdpa_math
    sdpa_kernel: str = "auto"
    use_gqa: bool = False
    sliding_window: Optional[int] = None
    dropout: float = 0.0


@dataclass
class PEFTConfig:
    enabled: bool = False
    method: str = "lora"
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])


@dataclass
class CheckpointingConfig:
    output_dir: str = "checkpoints"
    save_every_steps: int = 500
    save_best: bool = True
    save_total_limit: int = 3
    resume_from_checkpoint: Optional[str] = None


@dataclass
class LoggingConfig:
    log_file: str = "logs/log.txt"
    log_every_steps: int = 20
    metrics_file: str = "logs/metrics.jsonl"  # machine-readable per-step metrics; "" disables
    # CSV copy of the per-step metrics. None -> derive from metrics_file (.jsonl -> .csv); "" disables.
    metrics_csv: Optional[str] = None
    device_peak_flops: Optional[float] = None  # device peak FLOP/s for MFU; None disables MFU


@dataclass
class DistributedConfig:
    backend: str = "none"  # none | accelerate | ddp
    mixed_precision: str = "bf16"  # no | fp16 | bf16
    gradient_checkpointing: bool = False
    compile: bool = False


@dataclass
class EvaluationConfig:
    eval_every_steps: int = 200
    eval_steps: int = 10


@dataclass
class FineTuneConfig:
    enabled: bool = False
    dataset_path: str = "dataspace/instruction_data.jsonl"
    base_ckpt: str = "checkpoints/base_ckpt"
    output_ckpt: str = "checkpoints/instruct_ckpt"
    output_config: str = "checkpoints/instruct_config.yaml"
    prompt_template: str = "Instruction:\n{instruction}\n\nResponse:\n"
    response_template: str = "{response}"
    train_on_prompt: bool = False
    seed: int = 1337
    shuffle: bool = True


@dataclass
class TrainingConfig:
    mode: str = "pretrain"
    n_batches: int = 16
    batch_size: int = 6
    n_embd: int = 768
    n_head: int = 8
    n_blocks: int = 16
    seq_len: int = 1024
    lr: float = 0.0006
    dropout_rate: float = 0.2
    log_inter: int = 20
    eval_inter: int = 200
    eval_iter: int = 10
    max_iter: int = 100_000
    max_tokens: Optional[int] = None  # optional token budget; stop at min(max_iter, max_tokens)
    dtype: str = "long"
    tokenizer_type: str = "gpt2"
    tokenizer_dir: str = "checkpoints/tokenizer_dir"
    ckpt: str = "checkpoints/ckpt"
    ckpt_config: str = "checkpoints/ckpt_config.yaml"
    current_shard: int = 0
    training_step: int = 0
    training_duration: float = 0.0
    log_file: str = "logs/log.txt"
    data_dir: str = "dataspace/fineweb"
    num_workers: int = 1
    max_loss: float = float("inf")
    seed: int = 1337
    device: Optional[str] = None
    vocab_size: Optional[int] = None

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = _default_device()


@dataclass
class InferenceConfig:
    pretrained_model: str = "babyGPT_base.llm"
    pretrained_model_config: str = "babyGPT_base.yaml"
    generate_max_length: int = 50


@dataclass
class ConfigHandler:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    peft: PEFTConfig = field(default_factory=PEFTConfig)
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    finetune: FineTuneConfig = field(default_factory=FineTuneConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # Runtime flag (not a serialized field): set by apply_backward_compatibility when an old-style
    # data config was loaded, so the engine selects the deprecated LegacyShardDirSource path.
    _data_is_legacy = False

    @staticmethod
    def from_yaml(filepath: str) -> "ConfigHandler":
        raw = _load_yaml(filepath)
        if not isinstance(raw, dict):
            raise ConfigError(f"{filepath}: top-level YAML must be a mapping of sections, got {type(raw).__name__}")
        unknown = sorted(set(raw) - set(_SECTION_TYPES))
        if unknown:
            raise ConfigError(
                f"{filepath}: unknown config section(s) {unknown}. Valid sections: {sorted(_SECTION_TYPES)}"
            )

        config = ConfigHandler()
        for section_name, section_cls in _SECTION_TYPES.items():
            if section_name not in raw:
                continue
            section = _build_dataclass(section_cls, raw[section_name], f"{filepath}: section '{section_name}'")
            setattr(config, section_name, section)

        config.apply_backward_compatibility(raw)
        try:
            config.validate()
        except (ValueError, NotImplementedError) as exc:
            # Preserve the error type but add the file for context.
            raise type(exc)(f"{filepath}: {exc}") from exc
        return config

    load = from_yaml

    def apply_backward_compatibility(self, raw: dict) -> None:
        """Mirror original ``training.*`` fields into newer config sections."""
        for key in ["n_embd", "n_head", "n_blocks", "seq_len", "dropout_rate", "vocab_size"]:
            value = getattr(self.training, key, None)
            if value is not None:
                setattr(self.model, key, value)

        if "optimizer" not in raw or "lr" not in raw.get("optimizer", {}):
            self.optimizer.lr = self.training.lr
        if "logging" not in raw:
            self.logging.log_file = self.training.log_file
            self.logging.log_every_steps = self.training.log_inter
        if "evaluation" not in raw:
            self.evaluation.eval_every_steps = self.training.eval_inter
            self.evaluation.eval_steps = self.training.eval_iter
        if "data" not in raw or "train_on_prompt" not in raw.get("data", {}):
            self.data.train_on_prompt = self.finetune.train_on_prompt
        if self.training.tokenizer_type and "tokenizer" not in raw:
            self.tokenizer.type = "tiktoken" if self.training.tokenizer_type == "gpt2" else self.training.tokenizer_type
        self._apply_legacy_data_compatibility(raw)

    def _apply_legacy_data_compatibility(self, raw: dict) -> None:
        """Translate the old flat data layout (``training.data_dir`` / ``data.{dataset_name,...}``)
        into the new ``data.{source,transform,materialization}`` structure, once, with one warning.

        Legacy pretraining runs read a *directory* of shards with no manifest; the engine keeps that
        path working (via ``LegacyShardDirSource``) during the deprecation period.
        """
        data_raw = raw.get("data", {}) or {}
        new_data_keys = {"manifest_path", "source", "transform", "materialization", "sampling", "validation"}
        legacy_signals = bool(data_raw) or ("data_dir" in raw.get("training", {}))
        self._data_is_legacy = legacy_signals and not (new_data_keys & set(data_raw))
        if not self._data_is_legacy:
            return
        # Mirror legacy values so new-style tools/engine see a consistent view.
        self.data.source.dataset_name = self.data.dataset_name
        self.data.source.config_name = self.data.remote_name
        self.data.transform.shard_tokens = self.data.shard_size
        self.data.materialization.store_dir = self.training.data_dir or self.data.tokenized_data
        warnings.warn(
            "Legacy data configuration detected (training.data_dir / data.{dataset_name,remote_name,"
            "split_ratio,shard_size,tokenized_data}). These still work but are deprecated; migrate to the "
            "manifest-backed layout: run `python scripts/data.py migrate-legacy --data-dir <dir> "
            "--manifest <dir>/manifest.json`, then set data.manifest_path. See docs/data-pipeline.md.",
            DeprecationWarning,
            stacklevel=3,
        )

    def validate(self) -> None:
        _require(
            self.training.mode in {"pretrain", "finetune", "sft"}, "training.mode must be pretrain, finetune, or sft"
        )
        _require(self.training.batch_size > 0, "training.batch_size must be positive")
        _require(self.training.seq_len > 0, "training.seq_len must be positive")
        _require(self.training.n_batches > 0, "training.n_batches must be positive")
        _require(
            self.training.max_iter >= self.training.training_step, "training.max_iter must be >= training.training_step"
        )
        _require(self.model.n_embd % self.model.n_head == 0, "model.n_embd must be divisible by model.n_head")
        if self.model.num_key_value_heads is not None:
            _require(
                self.model.n_head % self.model.num_key_value_heads == 0,
                "model.n_head must be divisible by model.num_key_value_heads",
            )
        _require(self.model.norm in {"layernorm", "rmsnorm"}, "model.norm must be layernorm or rmsnorm")
        _require(self.model.ffn in {"gelu", "geglu", "swiglu", "moe"}, "model.ffn must be gelu, geglu, swiglu, or moe")
        _require(self.model.ffn_mult > 0, "model.ffn_mult must be positive")
        _require(self.model.position in {"learned", "rope", "none"}, "model.position must be learned, rope, or none")
        _require(self.optimizer.name in {"adamw", "fused_adamw"}, "optimizer.name must be adamw or fused_adamw")
        _require(self.scheduler.name in {"constant", "cosine"}, "scheduler.name must be constant or cosine")
        _require(
            self.attention.backend
            in {
                "eager",
                "sdpa",
                "sdpa_auto",
                "sdpa_flash",
                "sdpa_mem_efficient",
                "sdpa_math",
                "flash_attn_2",
                "xformers",
            },
            "unknown attention.backend",
        )
        _require(
            self.distributed.backend in {"none", "accelerate", "ddp"},
            "distributed.backend must be none, accelerate, or ddp",
        )
        _require(
            self.distributed.mixed_precision in {"no", "fp16", "bf16"},
            "distributed.mixed_precision must be no, fp16, or bf16",
        )
        _require(self.peft.method == "lora", "only peft.method=lora is currently supported")
        _require(self.peft.r > 0, "peft.r must be positive")
        _require(self.checkpointing.save_every_steps > 0, "checkpointing.save_every_steps must be positive")
        _require(self.checkpointing.save_total_limit > 0, "checkpointing.save_total_limit must be positive")
        self._validate_data()
        self._reject_unimplemented()

    def _validate_data(self) -> None:
        """Validate the orthogonal data sub-sections and their cross-section combinations."""
        materialization = self.data.materialization
        sampling = self.data.sampling
        transform = self.data.transform
        validation = self.data.validation
        source = self.data.source

        _require(
            materialization.mode in {"prepared", "prefetch", "direct"},
            "data.materialization.mode must be prepared, prefetch, or direct",
        )
        _require(
            sampling.exhaustion in {"stop", "repeat", "wait"},
            "data.sampling.exhaustion must be stop, repeat, or wait",
        )
        _require(sampling.shuffle in {"none", "shard"}, "data.sampling.shuffle must be none or shard")
        _require(transform.packing == "contiguous", "data.transform.packing only supports 'contiguous'")
        _require(
            transform.dtype in {"uint16", "uint32", "int32"},
            "data.transform.dtype must be uint16, uint32, or int32",
        )
        _require(materialization.prefetch_shards > 0, "data.materialization.prefetch_shards must be positive")
        _require(materialization.min_ready_shards > 0, "data.materialization.min_ready_shards must be positive")
        _require(materialization.tokenizer_workers >= 1, "data.materialization.tokenizer_workers must be >= 1")

        # dtype must hold the vocabulary.
        vocab = self.model.vocab_size or self.training.vocab_size
        if transform.dtype == "uint16" and vocab is not None:
            _require(vocab < 65536, f"data.transform.dtype=uint16 cannot hold vocab_size={vocab} (>= 65536)")

        # exactly one of validation.tokens / validation.shards when generating validation.
        if validation.manifest_path is None and (validation.tokens is not None or validation.shards is not None):
            _require(
                (validation.tokens is None) != (validation.shards is None),
                "set exactly one of data.validation.tokens or data.validation.shards (not both)",
            )

        # max_tokens must fit at least one full optimizer step.
        if self.training.max_tokens is not None:
            tokens_per_step = self.training.batch_size * self.training.seq_len * self.training.n_batches
            _require(
                self.training.max_tokens >= tokens_per_step,
                f"training.max_tokens={self.training.max_tokens} is smaller than one optimizer step "
                f"({tokens_per_step} tokens). Raise max_tokens or lower batch_size/seq_len/n_batches.",
            )

        # The combination rules below only constrain genuinely new-style pretraining configs.
        if self.training.mode in {"finetune", "sft"} or getattr(self, "_data_is_legacy", False):
            return
        if materialization.mode == "prefetch":
            _require(
                source.type == "huggingface" and bool(source.dataset_name),
                "data.materialization.mode=prefetch requires a materializable data.source "
                "(type=huggingface with a dataset_name)",
            )
        if materialization.mode == "direct":
            _require(
                source.type == "huggingface" and bool(source.dataset_name),
                "data.materialization.mode=direct requires a Hugging Face data.source",
            )
            _require(
                validation.manifest_path is not None,
                "data.materialization.mode=direct requires fixed validation: set data.validation.manifest_path",
            )
            _require(
                self.training.num_workers == 0,
                "direct mode requires training.num_workers=0 (single-process, no buffered prefetch)",
            )
            _require(
                self.distributed.backend == "none",
                "direct mode does not support a distributed backend; set distributed.backend=none",
            )
            _require(sampling.shuffle == "none", "direct mode does not support buffered shuffle; set shuffle=none")
            _require(sampling.exhaustion != "wait", "direct mode cannot use exhaustion=wait (no shard producer)")
        if sampling.exhaustion == "wait":
            _require(
                materialization.mode == "prefetch" or materialization.external_producer,
                "data.sampling.exhaustion=wait requires materialization.mode=prefetch or "
                "materialization.external_producer=true (a producer must be appending shards)",
            )

    def _reject_unimplemented(self) -> None:
        """Reject options the engine validates but does not yet implement.

        These are spelled correctly (so they pass the enum checks above) but have
        no working code path. Failing here at config load is far clearer than a
        surprise crash or, worse, silently running a single-process job for a
        config that asked for distributed training.
        """
        if self.attention.sliding_window is not None:
            raise NotImplementedError("attention.sliding_window is reserved for future work and is not implemented yet")
        if self.distributed.backend != "none":
            raise NotImplementedError(
                f"distributed.backend={self.distributed.backend!r} is not implemented; "
                "training is single-process only. Use distributed.backend='none'."
            )
        if self.model.ffn == "moe":
            raise NotImplementedError("model.ffn='moe' is reserved but not implemented yet")
        if self.attention.backend in {"flash_attn_2", "xformers"}:
            raise NotImplementedError(f"attention.backend={self.attention.backend!r} integration is not wired yet")
        if self.tokenizer.type in {"sp", "sentencepiece"}:
            raise NotImplementedError("tokenizer.type='sentencepiece' is a stub and not implemented yet")

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self, filepath: str) -> None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as handle:
            if yaml is None:
                import json

                json.dump(self.to_dict(), handle, indent=2)
            else:
                yaml.safe_dump(self.to_dict(), handle, sort_keys=False)


_SECTION_TYPES = {
    "training": TrainingConfig,
    "model": ModelConfig,
    "tokenizer": TokenizerConfig,
    "data": DataConfig,
    "optimizer": OptimizerConfig,
    "scheduler": SchedulerConfig,
    "attention": AttentionConfig,
    "peft": PEFTConfig,
    "checkpointing": CheckpointingConfig,
    "logging": LoggingConfig,
    "distributed": DistributedConfig,
    "evaluation": EvaluationConfig,
    "finetune": FineTuneConfig,
    "inference": InferenceConfig,
}


def _nested_dataclass_type(field_obj):
    """Return the nested dataclass type for a field built via ``field(default_factory=SomeDataclass)``."""
    factory = field_obj.default_factory
    if factory is not MISSING and isinstance(factory, type) and is_dataclass(factory):
        return factory
    return None


def _build_dataclass(cls, data, key_path: str):
    """Recursively construct ``cls`` from ``data``, rejecting unknown keys at every nesting level.

    Error messages carry the full key path (e.g. ``cfg.yaml: section 'data'.materialization``) so a
    typo deep in a nested section is easy to locate. Nested dataclasses are detected by their
    ``field(default_factory=...)`` being a dataclass type.
    """
    if not isinstance(data, dict):
        raise ConfigError(f"{key_path} must be a mapping, got {type(data).__name__}")
    valid = {f.name for f in fields(cls)}
    bad = sorted(set(data) - valid)
    if bad:
        raise ConfigError(f"{key_path}: unknown key(s) {bad}. Valid keys: {sorted(valid)}")
    kwargs = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        nested_cls = _nested_dataclass_type(f)
        if nested_cls is not None:
            kwargs[f.name] = _build_dataclass(nested_cls, data[f.name], f"{key_path}.{f.name}")
        else:
            kwargs[f.name] = data[f.name]
    return cls(**kwargs)


def _load_yaml(filepath: str) -> dict:
    if yaml is None:
        import json

        with open(filepath, "r", encoding="utf-8") as handle:
            return json.load(handle)
    with open(filepath, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _default_device() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)
