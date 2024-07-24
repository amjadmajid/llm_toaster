import os
import sys
import numpy as np
from datasets import load_dataset
from tqdm import tqdm
import multiprocessing as mp
from dataclasses import dataclass
import logging
from pathlib import Path
from config import DataConfig

# Add the parent directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tokenizer_lib import init_gpt2_tokenizer, gpt2_encode_hf


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _reset_names(root_dir, shard, split_name):
    """This helper function resets shards names."""
    new_shard = shard.split('_')
    new_shard = f'{new_shard[0]}_{new_shard[1]}_{split_name}.npy'
    os.rename(Path(root_dir) / shard, Path(root_dir) / new_shard)

def val_shards(data_root, split_ratio):
    logger.info("Creating the validation shards...")
    shards = sorted(os.listdir(data_root))
    shards = [_reset_names(data_root, s, "train") if "val" in s else s  
              for s in shards if s.endswith('.npy')]
    
    n_shard = len(shards)

    val_shards = max(1, int(n_shard * (1 - split_ratio)))
    # Rename the shards files from "shard_{shard_index:06d}_train.npy" to "shard_{shard_index:06d}_val.npy"
    for i in range(val_shards):
        old_name = Path(data_root) / f"shard_{i:06d}_train.npy"
        new_name = Path(data_root) / f"shard_{i:06d}_val.npy"
        os.rename(old_name, new_name)
    logging.info(f"Renamed {val_shards} shards to validation set.")


def create_data_cache_dir(output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

def download_and_tokenize(dataset_name, remote_name, split_ratio, output_dir, shard_size=int(1e8)):
    """
    Download and tokenize the dataset, saving it into shards.
    
    Args:
    - dataset_name (str): Name of the dataset to download.
    - remote_name (str): Remote configuration name of the dataset.
    - split (str): Dataset split to use.
    - output_dir (str): Directory to save the tokenized dataset.
    - shard_size (int): Number of tokens per shard.
    """
    create_data_cache_dir(output_dir)
    
    split="train"

    logging.info(f"Loading dataset: {dataset_name} with config: {remote_name} and split: {split}")
    try:
        dataset = load_dataset(dataset_name, name=remote_name, split=split)
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return

    nprocs = max(1, os.cpu_count() // 2)
    logging.info(f"Number of processors: {nprocs}")

    try:
        with mp.Pool(nprocs, initializer=init_gpt2_tokenizer) as pool:
            shard_index = 0
            all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
            token_count = 0
            progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")

            for tokens in pool.imap(gpt2_encode_hf, dataset, chunksize=16):
                n_tokens = len(tokens)
                if token_count + n_tokens < shard_size:
                    all_tokens_np[token_count:token_count + n_tokens] = tokens
                    token_count += n_tokens
                    progress_bar.update(n_tokens)
                else:
                    remainder = shard_size - token_count
                    all_tokens_np[token_count:token_count + remainder] = tokens[:remainder]
                    write_datafile(Path(output_dir) / f"shard_{shard_index:06d}_{split}.npy", all_tokens_np)

                    shard_index += 1
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                    token_count = n_tokens - remainder
                    all_tokens_np[:token_count] = tokens[remainder:]

            if token_count != 0:
                write_datafile(Path(output_dir) / f"shard_{shard_index:06d}_{split}.npy", all_tokens_np[:token_count])
                
            progress_bar.close()

        val_shards(output_dir, split_ratio)
    except Exception as e:
        logging.error(f"Error during tokenization: {e}")
        progress_bar.close()
        return

    logging.info("Tokenization complete and data saved.")

if __name__ == "__main__":
    data_config = DataConfig()

    download_and_tokenize(
        dataset_name=data_config.dataset_name,
        remote_name=data_config.remote_name,
        split_ratio=data_config.split_ratio,
        output_dir=data_config.output_dir,
        shard_size=data_config.shard_size
    )
