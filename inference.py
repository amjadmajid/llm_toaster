import torch
import logging
import argparse
from tokenizer_lib import init_tokenizer
from model import TransformerModel
from config import ConfigHandler
from pathlib import Path
import sys
from utils import load_checkpoint_
from transformers import GPT2Tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inference(config, model, tokenizer, prompt):
    model.eval()

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(config.training.device)
    
    if input_ids.size(1) > config.training.seq_len:
        input_ids = input_ids[:, -config.training.seq_len:]
        print(f"Input truncated to the last {config.training.seq_len} tokens.")

    with torch.no_grad():
        output_ids = model.generate_text(input_ids, max_length=config.inference.generate_max_length, tokenizer=tokenizer)
    
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(generated_text)

def delete_last_line():
    sys.stdout.write('\x1b[1A')  # Move cursor up one line
    sys.stdout.write('\x1b[2K')  # Delete the entire line
    sys.stdout.flush()

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Run inference on a pre-trained Transformer model.")
    parser.add_argument('-p', '--prompt', type=str, required=True, help='Prompt for the model')
    parser.add_argument('--config', type=str, help='Path to the configuration YAML file')
    args = parser.parse_args()

    initial_prompt = args.prompt

    # Load configurations
    try:
        config = ConfigHandler.from_yaml(args.config)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        exit(1)

    # Ensure device is set
    if not config.training.device:
        config.training.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    logger.info(f"The selected device is {config.training.device}")

    # Initialize the tokenizer
    try:
        tokenizer = GPT2Tokenizer.from_pretrained(config.training.tokenizer_dir)
        logger.info(f"Tokenizer loaded from {config.training.tokenizer_dir}")
        config.training.vocab_size = len(tokenizer)
    except:
        logger.error(f"Tokenizer directory {config.training.tokenizer_dir} does not exist.")
    
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

    # Load model state
    model_path = config.training.ckpt
    try:
        load_checkpoint_(model, optimizer=None, scaler=None, path=model_path, device=config.training.device, inference=True)
        logger.info(f"Model loaded from {model_path}")
    except Exception as e:
        logger.error(f"Error loading model state dict: {e}")
        exit(1)

    # Compile model if possible
    if hasattr(torch, 'compile') and 'cuda' in config.training.device:
        model = torch.compile(model)
    else:
        print("torch.compile is not available or not using CUDA. Proceeding without compilation.")

    # Set default generation length if not specified
    if not hasattr(config.inference, 'generate_max_length'):
        config.inference.generate_max_length = 50  # Adjust as needed

    # Run inference with the initial prompt
    prompt = initial_prompt
    while True:
        inference(config, model, tokenizer, prompt)
        print("\n-------------------------------------------")
        print("Enter the next prompt (or 'exit' to quit):")
        print("-------------------------------------------")
        prompt = input("")
        delete_last_line()
        if prompt.lower() == 'exit':
            break
