import torch
import numpy as np
import logging
import argparse
from tokenizer_lib import gpt2_decode, gpt2_encode, init_gpt2_tokenizer
from model import TransformerModel
from config import ConfigHandler, InferenceConfig
from pathlib import Path
import sys

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

def delete_last_line():
    sys.stdout.write('\x1b[1A')  # Move cursor up one line
    sys.stdout.write('\x1b[2K')  # Delete the entire line
    sys.stdout.flush()


if __name__ == "__main__":
    # Argument parser
    parser = argparse.ArgumentParser(description="Run inference on a pre-trained Transformer model.")
    parser.add_argument('-p', '--prompt', type=str, required=True, help='Prompt for the model')
    args = parser.parse_args()

    initial_prompt = args.prompt

    inference_config = InferenceConfig()
    babyGPT_config_path = Path('model/babyGPT/') / Path(inference_config.babyGPT_config)
    babyGPT_path = Path('model/babyGPT/') / Path(inference_config.babyGPT_name)
    # Load configurations
    try:
        # TODO: this is not a good approach. Enable the user to load the desired checkpoint
        config = ConfigHandler.load(babyGPT_config_path)
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

    model.load_state_dict(torch.load(babyGPT_path, map_location=config.device))

    if hasattr(torch, 'compile') and 'cuda' in config.device:
        model = torch.compile(model)
    else:
        print("torch.compile is not available. Proceeding without compilation.")

    if hasattr(torch, 'compile') and 'cuda' in config.device:
        model = torch.compile(model)
    else:
        print("torch.compile is not available. Proceeding without compilation.")


    # Run inference with the initial prompt
    prompt = initial_prompt
    while True:
        inference(config, model, prompt)
        print("\n-------------------------------------------")
        print("Enter the next prompt (or 'exit' to quit):")
        print("-------------------------------------------")
        prompt = input("")
        delete_last_line()
        if prompt.lower() == 'exit':
            break