# this script is used to extract the inference model from the trained model and save it to a file
# This is done to remove the optimizer and scaler from the model and save only the model's state_dict
# This is done to reduce the model's size and make it easier to load the model for inference

import torch
import logging
from model import TransformerModel
from config import ConfigHandler
from utils import load_checkpoint_
from pathlib import Path
from config import InferenceConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":

    inference_config = InferenceConfig()
    # Load configurations
    try:
        # TODO: this is not a good approach. Enable the user to load the desired checkpoint
        config = ConfigHandler.load("config/default_config.yaml")
        config = ConfigHandler.load(Path(config.ckpt_dir)/ Path(config.ckpt_config))
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        exit(1)

    logger.info(f"the selected device is automatically selected according to this device")
    config.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    logger.info(f"The selected device is {config.device}")

    # Initialize the model
    model = TransformerModel(
        n_head=config.n_head,
        vocab_size=config.vocab_size,
        n_embd=config.n_embd,
        seq_len=config.seq_len,
        device=config.device,
        dropout_rate=config.dropout_rate,
        n_blocks=config.n_blocks, 
        decoder=True
    ).to(config.device)

    model_pth = Path(config.ckpt_dir) / Path(config.ckpt_model)
    load_checkpoint_(model, "_", "_",  model_pth, config.device, inference=True)

    # Save only the model's state_dict
    babyGPT_path = Path('model/babyGPT/') / Path(inference_config.babyGPT_name)
    torch.save(model.state_dict(), babyGPT_path)