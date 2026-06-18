import logging
from dataclasses import asdict, dataclass, field
from typing import Optional

try:
    import yaml
except ModuleNotFoundError:
    yaml = None

config_logger = logging.getLogger(__name__)


def _coerce_scalar(value: str):
    value = value.split(" #", 1)[0].strip()
    if not value:
        return ""
    if (value.startswith("\"") and value.endswith("\"")) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1].encode("utf-8").decode("unicode_escape")
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value.replace("_", ""))
        return int(value.replace("_", ""))
    except ValueError:
        return value


def _load_yaml(filepath: str):
    if yaml is not None:
        with open(filepath, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    # Minimal fallback parser for this repository's simple two-level config YAML.
    result = {}
    section = None
    with open(filepath, "r", encoding="utf-8") as f:
        for raw in f:
            if not raw.strip() or raw.lstrip().startswith("#"):
                continue
            if not raw.startswith(" ") and raw.rstrip().endswith(":"):
                section = raw.strip()[:-1]
                result[section] = {}
                continue
            if section and ":" in raw:
                key, value = raw.strip().split(":", 1)
                result[section][key] = _coerce_scalar(value)
    return result


@dataclass
class DataConfig:
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    split_ratio: float = 0.98
    tokenized_data: str = "dataspace/fineweb"
    remote_name: str = "sample-10BT"
    shard_size: int = int(1e7)


@dataclass
class InferenceConfig:
    pretrained_model: str = "babyGPT_base.llm"
    pretrained_model_config: str = "babyGPT_base.yaml"
    generate_max_length: int = 50


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
    max_loss: float = field(default_factory=lambda: float("inf"))
    device: Optional[str] = None
    vocab_size: Optional[int] = None

    def __post_init__(self):
        if self.device is None:
            try:
                import torch
                self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
            except Exception:
                self.device = "cpu"

    def validate(self):
        if self.mode not in {"pretrain", "finetune"}:
            raise ValueError("training.mode must be 'pretrain' or 'finetune'")
        assert self.batch_size > 0, "Batch size must be positive"
        assert self.seq_len > 0, "Sequence length must be positive"
        assert self.n_batches > 0, "n_batches must be positive"


@dataclass
class ConfigHandler:
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    finetune: FineTuneConfig = field(default_factory=FineTuneConfig)

    @staticmethod
    def from_yaml(filepath: str) -> "ConfigHandler":
        config_logger.info("Loading configuration from %s", filepath)
        yaml_dict = _load_yaml(filepath)
        config = ConfigHandler()
        config.training = TrainingConfig(**yaml_dict.get("training", {}))
        config.data = DataConfig(**yaml_dict.get("data", {}))
        config.inference = InferenceConfig(**yaml_dict.get("inference", {}))
        config.finetune = FineTuneConfig(**yaml_dict.get("finetune", {}))
        config.training.validate()
        return config

    # Backwards-compatible alias used by older scripts in this repository.
    load = from_yaml

    def to_yaml(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            if yaml is None:
                import json
                json.dump(asdict(self), f, indent=2)
            else:
                yaml.safe_dump(asdict(self), f, sort_keys=False)
