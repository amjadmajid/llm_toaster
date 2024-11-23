# download_and_tokenize.py

import os
from datasets import load_dataset
import numpy as np
from tokenizer_lib import init_tokenizer, tokenize_sample
from tqdm import tqdm
import multiprocessing as mp
import argparse

from config import ConfigHandler

def process_sample(sample):
    """
    Function to tokenize a single sample.

    Args:
        sample (dict): The sample to tokenize.

    Returns:
        tokens (list): List of token IDs.
    """
    return tokenize_sample(sample, tokenizer_global)

def process_and_save_dataset(dataset_name, split_name, output_dir, tokenizer, sequences_per_shard=10000):
    """
    Downloads, tokenizes, and saves the dataset into shards.

    Args:
        dataset_name (str): Name of the dataset to download.
        split_name (str): Data split to use ('train' or 'validation').
        output_dir (str): Directory to save the tokenized data.
        tokenizer (GPT2Tokenizer): The tokenizer to use.
        sequences_per_shard (int): Number of sequences per shard file.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset
    dataset = load_dataset(dataset_name, split=split_name)

    sequences = []
    shard_index = 0

    # Make tokenizer a global variable so it can be accessed by worker processes
    global tokenizer_global
    tokenizer_global = tokenizer

    with mp.Pool(mp.cpu_count()) as pool:
        results = pool.imap(process_sample, dataset)

        for tokens in tqdm(results, total=len(dataset), desc=f"Tokenizing {split_name} data"):
            sequences.append(tokens)

            if len(sequences) >= sequences_per_shard:
                np.save(os.path.join(output_dir, f'shard_{shard_index:06d}_{split_name}.npy'), np.array(sequences, dtype=object), allow_pickle=True)
                shard_index += 1
                sequences = []

        # Save any remaining sequences
        if len(sequences) > 0:
            np.save(os.path.join(output_dir, f'shard_{shard_index:06d}_{split_name}.npy'), np.array(sequences, dtype=object), allow_pickle=True)

if __name__ == '__main__':
    # Argument parser
    parser = argparse.ArgumentParser(description="Download and tokenize data.")
    parser.add_argument('--config', type=str, default='config/default_config.yaml', help='Path to the configuration YAML file')
    args = parser.parse_args()

    # Load configurations
    try:
        config = ConfigHandler.from_yaml(args.config)
    except Exception as e:
        print(f"Error loading configuration: {e}")
        exit(1)

    tokenizer = init_tokenizer()
    dataset_name = config.data.dataset_name 
    output_dir = config.data.tokenized_data

    # Process train and validation splits
    process_and_save_dataset(dataset_name, 'train', output_dir, tokenizer)
    # process_and_save_dataset(dataset_name, 'validation', output_dir, tokenizer)
