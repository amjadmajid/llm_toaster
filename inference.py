import torch
import numpy as np
import logging
import argparse
from tokenizer_lib import gpt2_decode, gpt2_encode, init_gpt2_tokenizer
from llm_model import TransformerModel
from config import ConfigHandler
from utils import load_model_weights_
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def inference(config, model, prompt):
    model.eval()
    init_gpt2_tokenizer()
    input_ids = gpt2_encode(prompt, dtype=np.int32)
    input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0).to(config.device)
    
    if input_ids.size(1) > config.seq_len:
        input_ids = input_ids[:, -config.seq_len:]
        print(f"Input truncated to the last {config.seq_len} tokens.")
    
    with torch.no_grad():
        output = model.generate_text(input_ids, max_length=35)
    generated_text = gpt2_decode(output[0].cpu().tolist())
    print(generated_text)

if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Run inference on a pre-trained Transformer model.")
    parser.add_argument('-p', '--prompt', type=str, required=True, help='Prompt for the model')
    args = parser.parse_args()

    initial_prompt = args.prompt

    # Load configurations
    try:
        # TODO: this is not a good approach. Enable the user to load the desired checkpoint
        config = ConfigHandler.load("config/config.yaml")
        config = ConfigHandler.load(Path(config.ckpt_dir)/ Path(config.ckpt_config))
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        exit(1)

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
    load_model_weights_(model, model_pth, config.device)

    model = torch.compile(model)

    # Run inference with the initial prompt
    prompt = initial_prompt
    while True:
        inference(config, model, prompt)
        prompt = input("\nEnter the next prompt (or 'exit' to quit): ")
        if prompt.lower() == 'exit':
            break