import torch
import torch.nn as nn
import logging
import time

from utils import evaluate_model, save_model, count_parameters, load_model_weights_, log_training_info, setup_logging
from dataspace import DataLoaderLite
from config import ConfigHandler
from model import TransformerModel
from pathlib import Path
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# load configurations
try:
    config = ConfigHandler.load("config/config.yaml")
except Exception as e:
    print(f"Error loading configuration: {e}")

logger.info(f"The selected device is {config.device}")

def train_model(model, optimizer, criterion, continue_training, config): 

    if continue_training:
        # load checkpointed configuration and model weights
        config = ConfigHandler.load(Path(config.ckpt_dir) / Path(config.ckpt_config))
        # update model weights inplace 
        load_model_weights_(model, Path(config.ckpt_dir) / Path(config.ckpt_model), config.device)  
        logger.info("Loaded model's weights")

    ckpt_model_path = Path(config.ckpt_dir) / Path(config.ckpt_model)
    ckpt_config_path = Path(config.ckpt_dir) / Path(config.ckpt_config)

    # setup the training progress logger
    training_logger = setup_logging("log.txt")

    # initialize data loaders
    training_data = DataLoaderLite(config.batch_size, config.seq_len, config.current_shard,  0, 1, 'train')
    val_data = DataLoaderLite(config.batch_size, config.seq_len,  0, 0, 1, 'val')

    total_loss = 0
    model = torch.compile(model)

    start_interval_timing = time.time()
    start_ckpt_timing = time.time()

    log_iteration = config.training_step + config.log_inter 

    assert config.training_step < config.max_iter, "config.training_step must be smaller than config.max_iter"


    for iteration in range(config.training_step, config.max_iter + 1):
        
        optimizer.zero_grad()
        batch_loss = 0

        for _ in range(config.n_batches):

            X, Y, current_shard = training_data.next_batch()
            X = torch.tensor(X, dtype=torch.long).to(config.device)
            Y = torch.tensor(Y, dtype=torch.long).to(config.device)
            with torch.autocast(device_type=config.device, dtype=torch.bfloat16):
                logits = model(X)

                loss = criterion(logits.view(-1, logits.size(-1)), Y.view(-1)) 
                loss /=config.n_batches
            batch_loss += loss.item()
            loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += batch_loss

        if iteration >= log_iteration:
            log_iteration += config.log_inter
            current_time = time.time()
            iteration_duration = (current_time - start_interval_timing) / config.log_inter
            training_duration = current_time - start_ckpt_timing + config.training_duration
            log_training_info(training_logger, iteration, \
                                config, total_loss, iteration_duration, training_duration)
            total_loss = 0
            start_interval_timing = current_time

        # if iteration % config.eval_inter == 0:
        #     val_loss = evaluate_model(model, val_data, criterion, config)
        #     logger.info(f"Iteration {iteration} | Validation Loss {val_loss:.5f} ")
        
        # checkpointing
        if batch_loss < config.max_loss: 
            config.max_loss = batch_loss
            config.training_duration = training_duration
            start_ckpt_timing = time.time()
            config.current_shard = current_shard
            config.training_step = iteration  # keep track of training progress across training sessions
            save_model(model, ckpt_model_path)
            config.save(ckpt_config_path)
            logger.info(f"{iteration} - Model saved to {ckpt_model_path}; loss: {batch_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BabyGPT script for LLM training")
    parser.add_argument('-ct', '--continue-training', action='store_true', 
                        help='Flag to continue training from a saved model state')
    args = parser.parse_args()

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

    train_model(model, optimizer, criterion, args.continue_training, config)
