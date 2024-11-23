import torch
import torch.nn as nn
import logging
import time

from utils import save_model, count_parameters, load_checkpoint_, training_logs, write_logs
from dataspace import DataLoaderLite
from config import ConfigHandler
from tokenizer_lib import init_tokenizer
from model import TransformerModel
from pathlib import Path
import argparse
import signal


# Global flag for interrupt
interrupted = False

def signal_handler(sig, frame):
    """
    Signal handler for SIGINT (Ctrl-C).
    Sets the interrupted flag to True.
    """
    global interrupted
    print('\nReceived Ctrl-C. Will save checkpoint and exit after this iteration.')
    interrupted = True

# Register the signal handler
signal.signal(signal.SIGINT, signal_handler)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TERMINAL_LOG")

# load configurations
try:
    config = ConfigHandler.from_yaml("config/default_config.yaml")
except Exception as e:
    print(f"Error loading configuration: {e}")

logger.info(f"The selected device is {config.training.device}")

def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)

def train_model(model, optimizer, criterion, continue_training, config): 

    scaler = torch.cuda.amp.GradScaler()

    if continue_training:
        # load checkpointed configuration and model weights
        config = ConfigHandler.from_yaml(config.training.ckpt_config)
        # update model weights inplace 
        load_checkpoint_(model,optimizer, scaler,  config.training.ckpt, config.training.device)  
        logger.info("Model, optimizer, and scaler weights are loaded")
    else:
        write_logs(config.training.log_file, "", append_txt=False)

    assert config.training.training_step < config.training.max_iter, "config.training_step must be smaller than config.max_iter"

    last_log_iteration = 0
    total_loss = 0
    log_str = ""

    log_iteration = config.training.training_step + config.training.log_inter 
    

    if hasattr(torch, 'compile'):
        model = torch.compile(model)
    else:
        print("torch.compile is not available. Proceeding without compilation.")

    # initialize data loaders
    training_data = DataLoaderLite(config.training.batch_size, config.training.seq_len, config.training.current_shard,  0, 1, 'train')
    # val_data = DataLoaderLite(config.batch_size, config.seq_len,  0, 0, 1, 'val')

    start_interval_timing = time.time()
    start_ckpt_timing = time.time()

    try: 
        for iteration in range(config.training.training_step, config.training.max_iter ):
            
            optimizer.zero_grad()
            batch_loss = 0

            for _ in range(config.training.n_batches):

                X, Y, current_shard = training_data.next_batch()
                X = torch.as_tensor(X, dtype=torch.long).to(config.training.device)
                Y = torch.as_tensor(Y, dtype=torch.long).to(config.training.device)
                with torch.autocast(device_type=config.training.device, dtype=torch.bfloat16):
                    logits = model(X)

                    loss = criterion(logits.view(-1, logits.size(-1)), Y.view(-1)) 
                    loss /=config.training.n_batches
                batch_loss += loss.item()

                scaler.scale(loss).backward()
                
            # gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            # the reason for not using a modulo here is that when training resumed `iteration` can any number
            if iteration >= log_iteration or batch_loss < config.training.max_loss:
                log_iteration += config.training.log_inter

                current_time = time.time()
                
                #Calculate iteration duration
                iteration_duration = (current_time - start_interval_timing) / max(1, (iteration - last_log_iteration))
                # Update the total training duration
                training_duration = current_time - start_ckpt_timing + config.training.training_duration
                # Log training progress to terminal
                _log_str = training_logs(iteration, \
                                    config.training, batch_loss, iteration_duration, training_duration)
                
                logger.info(_log_str[:-1]) # remove the newline because the logger adds one
                log_str += _log_str
                
                last_log_iteration = iteration
                start_interval_timing = current_time

            # if iteration % config.eval_inter == 0:
            #     val_loss = evaluate_model(model, val_data, criterion, config)
            #     logger.info(f"Iteration {iteration} | Validation Loss {val_loss:.5f} ")
            
                # Checkpointing
                if batch_loss < config.training.max_loss or interrupted: 
                    # Temporarily ignore SIGINT during checkpointing
                    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

                    try: 
                        config.training.max_loss = batch_loss

                        # Update the training duration and reset checkpoint timing
                        config.training.training_duration = current_time - start_ckpt_timing + config.training.training_duration
                        start_ckpt_timing = current_time

                        # Update config checkpoint with current progress
                        config.training.current_shard = current_shard
                        config.training.training_step = iteration  

                        # Save the model and configuration
                        save_model(model, optimizer, scaler, config.training.ckpt)
                        config.to_yaml(config.training.ckpt_config)
                        #log to terminal
                        logger.info(f"{iteration} - Model saved to {config.training.ckpt}; loss: {batch_loss:.4f}")
                        #log traning progress to a file

                        write_logs(config.training.log_file, log_str)
                        log_str =""
                    finally:
                        # Restore the original SIGINT handler
                        signal.signal(signal.SIGINT, original_sigint_handler)
                    if interrupted:
                        logger.info("Checkpoint saved. Exiting training loop.")
                        break
    
    except KeyboardInterrupt:
        # Catch any KeyboardInterrupt exceptions that may not have been handled
        logger.info("Training interrupted by user. Exiting without saving.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BabyGPT script for LLM training")
    parser.add_argument('-ct', '--continue-training', action='store_true', 
                        help='Flag to continue training from a saved model state')
    args = parser.parse_args()

    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')
    else:
        print("PyTorch version does not support set_float32_matmul_precision.")

    #TODO
    config.training.vocab_size = 50304 # len(tokenizer)

    # initialize the model
    model = TransformerModel(
        n_head=config.training.n_head,
        vocab_size=config.training.vocab_size,
        n_embd=config.training.n_embd, 
        seq_len=config.training.seq_len, 
        device=config.training.device,
        dropout_rate=config.training.dropout_rate, 
        n_blocks=config.training.n_blocks,
        decoder=True 
    ).to(config.training.device)

    model.apply(init_weights)

    # Clean CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Cleared CUDA cache to free up memory.")

    # initialize optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr)
    criterion = nn.CrossEntropyLoss()

    logger.info(f"The model has {count_parameters(model)} trainable parameters")

    train_model(model, optimizer, criterion, args.continue_training, config)
