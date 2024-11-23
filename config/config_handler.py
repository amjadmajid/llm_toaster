import yaml
import torch
from dataclasses import dataclass, field, asdict
from typing import Optional
import logging

config_logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    split_ratio: float = 0.98
    tokenized_data: str = "../../dataspace/fineweb"
    remote_name: str = "sample-10BT"
    shard_size: int = int(1e7)

@dataclass
class InferenceConfig:
    pretrained_model: str = "default_model"
    pretrained_model_config: str  ="default_model_config.yaml"
    generate_max_length: int = 50 

@dataclass
class TrainingConfig:
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
    log_file: str = "log.txt"
    data_dir: str = "dataspace/tokenized_data"
    num_workers: int = 1
    max_loss: float = field(default_factory=lambda: float('inf'))
    device: Optional[str] = None
    vocab_size: Optional[int] = None #50304

    def __post_init__(self):
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    def validate(self):
        assert self.batch_size > 0, "Batch size must be positive"
        

@dataclass
class ConfigHandler:
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    inference: InferenceConfig = InferenceConfig()

    @staticmethod
    def from_yaml(filepath: str) -> 'ConfigHandler':
        config_logger.info(f"Loading configuration from {filepath}")
        try:
            with open(filepath, 'r') as f:
                yaml_dict = yaml.safe_load(f)
            config = ConfigHandler()
            config.training = TrainingConfig(**yaml_dict.get('training', {}))
            config.data = DataConfig(**yaml_dict.get('data', {}))
            config.inference = InferenceConfig(**yaml_dict.get('inference', {}))
            return config
        except Exception as e:
            config_logger.error(f"Error loading config: {e}")
            raise

    def to_yaml(self, filepath: str):
        try:
            with open(filepath, 'w') as f:
                yaml.dump(asdict(self), f, indent=4)
        except Exception as e:
            config_logger.error(f"Error saving config: {e}")