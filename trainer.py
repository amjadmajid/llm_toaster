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

def train(model, optimizer, criterion ,continue_training, config): 

    if continue_training:
        # load checkpointed configuration and model weights
        config = ConfigHandler.load(Path(config.ckpt_dir) / Path(config.ckpt_config))
        load_model_weights_(model, Path(config.ckpt_dir) / Path(config.ckpt_model), config.device) # update model weights inplace 
        logger.info("Loaded model's weights")

    train_losses = 0
    model = torch.compile(model)
    start_time = time.time()

    for iteration in range(1, config.max_iter+1):
        
        optimizer.zero_grad()
        grad_accumulations = config.batch_size // config.micro_batch_size
        accum_loss = 0

        for _ in range(grad_accumulations):

            X, Y = training_data.next_batch()
            X = torch.tensor(X, dtype=torch.long).to(config.device)
            Y = torch.tensor(Y, dtype=torch.long).to(config.device)
            with torch.autocast(device_type=config.device, dtype=torch.bfloat16):
                logits = model(X)

                loss = criterion(logits.view(-1, logits.size(-1)), Y.view(-1))
                loss = loss / grad_accumulations
            accum_loss += loss.item()
            loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        train_losses += accum_loss

        if iteration % config.log_inter == 0:
            # processing speed logging
            end_time = time.time() 
            dt =  (end_time - start_time ) / config.log_inter
            start_time = end_time
            processed_tokens = config.batch_size * config.seq_len
            tokens_per_sec = processed_tokens // dt 
            # loss logging
            train_loss = train_losses / config.log_inter
            train_losses = 0

            logger.info(f"Iteration {iteration} | Train Loss {train_loss:.5f} |  dt {dt * 1000:.3f} ms | {tokens_per_sec} tokens/sec")

        if iteration  % config.eval_inter == 0:
            # TODO: put with torch.no_grad() here. Outside the function
            val_loss = evaluate_model(model,val_data, criterion, config)
            logger.info(f"Iteration {iteration} | Validation Loss {val_loss:.5f} ")
        
        # checkpointing
        if accum_loss < config.max_loss: 
            config.max_loss = accum_loss
            ckpt_model_path =  Path(config.ckpt_dir) / Path(config.ckpt_model)
            ckpt_config_path =  Path(config.ckpt_dir) / Path(config.ckpt_config)
            save_model(model, ckpt_model_path )
            config.save(ckpt_config_path)
            logger.info(f"{iteration} - Model saved to {ckpt_model_path}; loss: {accum_loss:.4f}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description="BabyGPT script for LLM training")
    parser.add_argument('-ct', '--continue-training', action='store_true', help='Flag to continue training from a saved model state')
    args = parser.parse_args()

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
        dropout_rate=config.dropout_rate, 
        n_blocks=config.n_blocks,
        decoder=True 
    ).to(config.device)

    # initialize optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.CrossEntropyLoss()

    logger.info(f"The model has {count_parameters(model)} trainable parameters")

    train(model, optimizer, criterion, args.continue_training, config)