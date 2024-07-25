import torch
import torch.nn as nn
import logging
import time

from utils import evaluate_model, save_model, count_parameters, load_model_weights_
from dataspace import DataLoaderLite
from config import ConfigHandler
from llm_model import TransformerModel
from pathlib import Path
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


## Load configurations
try:
    config = ConfigHandler.load("config/config.yaml")
    assert config.batch_size % config.micro_batch_size == 0, "batch_size has to be divisible by micro_batch_size"
except Exception as e:
    logger.error(f"Error loading configuration: {e}")
    exit(1)

logger.info(f"The selected device is {config.device}")

# initialize data loaders
training_data = DataLoaderLite(config.micro_batch_size, config.seq_len, 0, 1, 'train')
val_data      = DataLoaderLite(config.micro_batch_size, config.seq_len, 0, 1, 'val')

torch.set_float32_matmul_precision('high')

# initialize the model
model = TransformerModel(
    n_head=config.n_head,
    vocab_size=config.vocab_size,
    n_embd=config.n_embd, 
    seq_len=config.seq_len, 
    device=config.device,
    dropout_rate=config.embd_pdrop, 
    n_blocks=config.n_blocks 
).to(config.device)

# model = torch.compile(model)

# initialize optimizer and loss
optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
criterion = nn.CrossEntropyLoss()

logger.info(f"The model has {count_parameters(model)} trainable parameters")

def train(continue_training, config): 
    max_loss = float('inf')
    train_losses = 0

    if continue_training:
        # load checkpointed configuration and model weights
        config = ConfigHandler.load(config.ckpt_dir+"/"+config.ckpt_config)
        load_model_weights_(model, config.ckpt_dir+"/"+config.ckpt_model, config.device)
        logger.info("Loaded model's weights")

    start_time = time.time()

    for iteration in range(1, config.max_iter+1):
        
        optimizer.zero_grad()
        grad_accumulations = config.batch_size // config.micro_batch_size
        accum_loss = 0

        for _ in range(grad_accumulations):

            X, Y = training_data.next_batch()
            X = torch.tensor(X, dtype=torch.long).to(config.device)
            Y = torch.tensor(Y, dtype=torch.long).to(config.device)
            # X, Y = training_data.get_rand_batch(config.batch_size, config.seq_len)
            # logger.info(f"Training batch shapes - X: {X.shape}, Y: {Y.shape}")
            with torch.autocast(device_type=config.device, dtype=torch.bfloat16):
                logits = model(X)
                # logger.info(f"Logits shape: {logits.shape}")

                loss = criterion(logits.view(-1, logits.size(-1)), Y.view(-1))
                loss = loss / grad_accumulations
            accum_loss += loss.item()
            loss.backward()
            # logger.info(f"Loss: {loss.item()}")

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        train_losses += accum_loss

        if iteration  % config.eval_inter == 0:

            # processing speed logging
            end_time = time.time() 
            dt =  (end_time - start_time ) / config.eval_inter
            start_time = end_time
            processed_tokens = config.batch_size * config.seq_len
            tokens_per_sec = processed_tokens // dt 

            # loss logging
            train_loss = train_losses / config.eval_inter
            train_losses = 0
            # TODO: put with torch.no_grad() here. Outside the function
            val_loss = evaluate_model(model,val_data, criterion, config)
            logger.info(f"Iteration {iteration} | Train Loss {train_loss:.5f} | Validation Loss {val_loss:.5f} | dt {dt * 1000:.3f} ms | {tokens_per_sec} tokens/sec")

            # checkpointing
            if val_loss < max_loss: 
                max_loss = val_loss
                ckpt_model_path =  Path(config.ckpt_dir) / Path(config.ckpt_model)
                ckpt_config_path =  Path(config.ckpt_dir) / Path(config.ckpt_config)
                save_model(model, ckpt_model_path )
                config.save(ckpt_config_path)
                logger.info(f"{iteration} - Model saved to {ckpt_model_path}; loss: {val_loss:.4f}")


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="BabyGPT script for LLM training")
    parser.add_argument('-ct', '--continue-training', action='store_true', help='Flag to continue training from a saved model state')
    args = parser.parse_args()

    train(args.continue_training, config)