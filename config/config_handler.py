import yaml
import torch
from dataclasses import dataclass, asdict
import logging

config_logger = logging.getLogger(__name__)

@dataclass
class DataConfig:
    dataset_name: str = "HuggingFaceFW/fineweb-edu"
    split_ratio: float = .98
    output_dir: str =  "../../dataspace/fineweb" 
    remote_name: str = "sample-10BT"
    shard_size:int = int(1e7)

@dataclass
class InferenceConfig:
    babyGPT_name: str = "BabyGPT_152M.llm"
    babyGPT_config: str = "babyGPT_config.yaml"


@dataclass
class ConfigHandler:
    n_batches: int
    batch_size: int
    n_embd: int
    n_head: int
    seq_len: int
    lr: float
    dropout_rate: float
    log_inter: int
    eval_inter: int
    eval_iter: int
    max_iter: int
    n_blocks: int
    dtype: str
    ckpt_dir: str
    ckpt_model: str
    ckpt_config: str
    tokenizer_type: str
    vocab_size: int
    current_shard: int
    training_step: int
    training_duration: float
    log_file : str
    max_loss: float = float('inf')
    device: str = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    @staticmethod
    def from_dict(obj: dict) -> 'ConfigHandler':
        return ConfigHandler(**obj)

    def to_dict(self) -> dict: 
        return asdict(self)
    
    def save(self, filepath: str):
        try:
            with open(filepath, 'w') as f: 
                yaml.dump(self.to_dict(), f, indent=4)
        except Exception as e:
            config_logger.error(f"Error saving config: {e}")

    @staticmethod
    def load(filepath: str) -> 'ConfigHandler': 
        config_logger.info(filepath)
        try:
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
            return ConfigHandler.from_dict(config_dict)
        except Exception as e:
            config_logger.error(f"Error loading config: {e}")
            raise
