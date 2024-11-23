# train.py

import torch
import torch.nn as nn
import logging
import time
import argparse
import signal
from pathlib import Path

from utils import save_model, count_parameters, load_checkpoint_, training_logs, write_logs
from dataspace import get_dataloader
from tokenizer_lib import init_tokenizer
from config import ConfigHandler
from model import TransformerModel
from transformers import GPT2Tokenizer

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

def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)



def train_model(model, optimizer, criterion, config, tokenizer):

    last_log_iteration = 0
    total_loss = 0
    log_str = ""

    log_iteration = config.training.training_step + config.training.log_inter

    if hasattr(torch, 'compile'):
        model = torch.compile(model)
    else:
        print("torch.compile is not available. Proceeding without compilation.")

    # Initialize tokenizer and data loaders
    train_loader = get_dataloader(
        data_dir=config.training.data_dir,
        split_name='train',
        tokenizer=tokenizer,
        seq_length=config.training.seq_len,
        batch_size=config.training.batch_size,
        num_workers=config.training.num_workers
    )

    start_interval_timing = time.time()
    start_ckpt_timing = time.time()

    iteration = config.training.training_step

    try:
        while iteration < config.training.max_iter:
            for batch_idx, batch in enumerate(train_loader):
                optimizer.zero_grad()
                accumulated_loss = 0.0

                for _ in range(config.training.n_batches):
                    input_ids = batch['input_ids'].to(config.training.device)
                    attention_mask = batch['attention_mask'].to(config.training.device)
                    labels = batch['labels'].to(config.training.device)

                    with torch.autocast(device_type=config.training.device, dtype=torch.bfloat16):
                        logits = model(input_ids, attention_mask)

                        # Shift logits and labels for next-token prediction
                        logits = logits[:, :-1, :].contiguous()
                        labels = labels[:, 1:].contiguous()

                        # Flatten the tensors
                        logits = logits.view(-1, logits.size(-1))
                        labels = labels.view(-1)

                        # Calculate loss
                        loss = criterion(logits, labels) 

                    scaler.scale(loss).backward()
                    accumulated_loss += loss.item()

                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

                batch_loss = accumulated_loss / config.training.n_batches

                # Logging and checkpointing
                # TODO: This condition is buggy. It increases by log_inter even when the batch_loss is less than max_loss
                if iteration >= log_iteration or batch_loss < config.training.max_loss:
                    log_iteration += config.training.log_inter

                    current_time = time.time()

                    # Calculate iteration duration
                    iteration_duration = (current_time - start_interval_timing) / max(1, (iteration - last_log_iteration))
                    # Update the total training duration
                    training_duration = current_time - start_ckpt_timing + config.training.training_duration
                    # Log training progress to terminal
                    _log_str = training_logs(iteration, config, batch_loss, iteration_duration, training_duration)

                    logger.info(_log_str[:-1])  # Remove the newline because the logger adds one
                    log_str += _log_str

                    last_log_iteration = iteration
                    start_interval_timing = current_time

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
                            config.training.training_step = iteration

                            # Save the model and configuration
                            save_model(model, optimizer, scaler, config.training.ckpt, config.training.tokenizer_dir, tokenizer=tokenizer)
                            config.to_yaml(config.training.ckpt_config)
                            # Log to terminal
                            logger.info(f"{iteration} - Model saved to {config.training.ckpt}; loss: {batch_loss:.4f}")
                            # Log training progress to a file
                            write_logs(config.training.log_file, log_str)
                            log_str = ""
                        finally:
                            # Restore the original SIGINT handler
                            signal.signal(signal.SIGINT, original_sigint_handler)
                        if interrupted:
                            logger.info("Checkpoint saved. Exiting training loop.")
                            return  # Exit the function

                iteration += 1
                if iteration >= config.training.max_iter or interrupted:
                    break  # Exit the inner loop

            if interrupted:
                break  # Exit the outer loop

    except KeyboardInterrupt:
        # Catch any KeyboardInterrupt exceptions that may not have been handled
        logger.info("Training interrupted by user. Exiting without saving.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BabyGPT script for LLM training")
    parser.add_argument('-ct', '--continue-training', action='store_true',
                        help='Flag to continue training from a saved model state')
    parser.add_argument('--config', type=str , help='Path to the configuration YAML file')
    args = parser.parse_args()

    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')
    else:
        logger.info("PyTorch version does not support set_float32_matmul_precision.")

    # Load configurations
    try:
        config = ConfigHandler.from_yaml(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        exit(1)
    
    # This can be violated when the configuration is loaded from a checkpoint
    assert config.training.training_step < config.training.max_iter, "config.training.training_step must be smaller than config.training.max_iter"

    # Ensure device is set
    if not config.training.device:
        config.training.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    logger.info(f"The selected device is {config.training.device}")

    # Initialize or load the tokenizer
    if args.continue_training:
        tokenizer = GPT2Tokenizer.from_pretrained(config.training.tokenizer_dir)
        logger.info(f"Tokenizer loaded from {config.training.tokenizer_dir}")
    else:
        # Initialize the tokenizer
        tokenizer = init_tokenizer()  

    config.training.vocab_size = len(tokenizer)

    # Initialize the model
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

    # Initialize optimizer, loss, and scaler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    scaler = torch.cuda.amp.GradScaler()

    if args.continue_training:
        # Update model weights in-place
        load_checkpoint_(model, optimizer, scaler, config.training.ckpt, config.training.device)
        logger.info("Model, optimizer, and scaler weights are loaded")
    else:
        write_logs(config.training.log_file, "", append_txt=False)
    
    logger.info(f"The model has {count_parameters(model)} trainable parameters")

    train_model(model, optimizer, criterion, config, tokenizer)
