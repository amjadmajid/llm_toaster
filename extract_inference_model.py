# this script is used to extract the inference model from the trained model and save it to a file
# This is done to remove the optimizer and scaler from the model and save only the model's state_dict
# This is done to reduce the model's size and make it easier to load the model for inference

# TODO
# copy over the config file
# add an argument to enable the user to name the model


import torch
import logging
from model import TransformerModel
from config import ConfigHandler
from utils import load_checkpoint_
from pathlib import Path
from config import ConfigHandler
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Run inference on a pre-trained Transformer model.")
    parser.add_argument('--config', type=str, help='Path to the configuration YAML file')
    args = parser.parse_args()

    # Load configurations
    try:
        config = ConfigHandler.from_yaml(args.config)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        exit(1)

    logger.info(f"the selected device is automatically selected according to this device")
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    logger.info(f"The selected device is {device}")

    # Initialize the model
    model = TransformerModel(
        n_head=config.training.n_head,
        vocab_size=config.training.vocab_size,
        n_embd=config.training.n_embd,
        seq_len=config.training.seq_len,
        device=device,
        dropout_rate=config.training.dropout_rate,
        n_blocks=config.training.n_blocks, 
        decoder=True
    ).to(device)

    load_checkpoint_(model, "_", "_",  config.training.ckpt, device, inference=True)

    # Save only the model's state_dict
    pretrained_model = Path('model/pretrained_models/') / Path(config.inference.pretrained_model)
    torch.save(model.state_dict(), pretrained_model)