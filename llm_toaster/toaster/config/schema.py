"""Configuration objects for the modular LLM Toaster engine.

The schema intentionally uses dataclasses instead of a heavier validation
framework so the project stays easy to read.  Validation is centralized in
``ConfigHandler.validate`` and legacy fields from the original repo are mirrored
into the newer sections for backward compatibility.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
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
    ffn: str = "gelu"  # gelu | swiglu | geglu | moe placeholder
    position: str = "learned"  # learned | rope placeholder
    num_key_value_heads: Optional[int] = None
    tie_embeddings: bool = True


@dataclass
class TokenizerConfig:
    type: str = "tiktoken"  # tiktoken | hf | sentencepiece
    name: str = "gpt2"
    path: Optional[str] = None


@dataclass
class DataConfig:
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    split_ratio: float = 0.98
    tokenized_data: str = "dataspace/fineweb"
    remote_name: str = "sample-10BT"
    shard_size: int = int(1e7)
    train_path: Optional[str] = None
    eval_path: Optional[str] = None
    format: str = "auto"
    train_on_prompt: bool = False


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
            section_raw = raw[section_name]
            if not isinstance(section_raw, dict):
                raise ConfigError(
                    f"{filepath}: section '{section_name}' must be a mapping, got {type(section_raw).__name__}"
                )
            valid = {f.name for f in fields(section_cls)}
            bad = sorted(set(section_raw) - valid)
            if bad:
                raise ConfigError(
                    f"{filepath}: unknown key(s) {bad} in section '{section_name}'. Valid keys: {sorted(valid)}"
                )
            setattr(config, section_name, section_cls(**section_raw))

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
        _require(self.model.position in {"learned", "rope"}, "model.position must be learned or rope")
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
        self._reject_unimplemented()

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
        if self.model.position == "rope":
            raise NotImplementedError("model.position='rope' is reserved but not implemented yet")
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
