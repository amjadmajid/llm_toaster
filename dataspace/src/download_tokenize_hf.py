import argparse
import logging
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from config import DataConfig
from tokenizer_lib import gpt2_encode_hf, init_gpt2_tokenizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _reset_names(root_dir, shard, split_name):
    """
    Reset shard names to include the split name.

    Args:
    - root_dir (str): Root directory containing the shards.
    - shard (str): Name of the shard file.
    - split_name (str): New split name to include in the shard file name.
    """
    new_shard = shard.split("_")
    new_shard = f"{new_shard[0]}_{new_shard[1]}_{split_name}.npy"
    os.rename(Path(root_dir) / shard, Path(root_dir) / new_shard)


def val_shards(data_root, split_ratio):
    """
    Create validation shards by renaming a portion of training shards.

    Args:
    - data_root (str): Directory containing the data shards.
    - split_ratio (float): Ratio of data to use for training; remainder is for validation.
    """
    logger.info("Creating the validation shards...")
    shards = sorted(os.listdir(data_root))
    shards = [_reset_names(data_root, s, "train") if "val" in s else s for s in shards if s.endswith(".npy")]

    n_shard = len(shards)
    val_shards = max(1, int(n_shard * (1 - split_ratio)))

    for i in range(val_shards):
        old_name = Path(data_root) / f"shard_{i:06d}_train.npy"
        new_name = Path(data_root) / f"shard_{i:06d}_val.npy"
        os.rename(old_name, new_name)
    logger.info(f"Renamed {val_shards} shards to validation set.")


def create_data_cache_dir(output_dir):
    """
    Create a directory to cache the tokenized data.

    Args:
    - output_dir (str): Path to the output directory.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def write_datafile(filename, tokens_np):
    """
    Save tokenized data to a .npy file.

    Args:
    - filename (str): Path to the output file.
    - tokens_np (numpy.ndarray): Array of tokenized data.
    """
    np.save(filename, tokens_np)


def download_and_tokenize(
    dataset_name,
    remote_name,
    split_ratio,
    output_dir,
    shard_size=int(1e8),
    stream=False,
    max_shards=None,
):
    """
    Download and tokenize the dataset, saving it into shards.

    Args:
    - dataset_name (str): Name of the dataset to download.
    - remote_name (str): Remote configuration name of the dataset.
    - split_ratio (float): Ratio of data to use for training; remainder is for validation.
    - output_dir (str): Directory to save the tokenized dataset.
    - shard_size (int): Number of tokens per shard.
    - stream (bool): If True, stream the dataset (datasets streaming=True) instead of downloading it
      in full first. Combined with max_shards this pulls only as much data as it writes (ideal for
      Colab / limited disk).
    - max_shards (int | None): If set, stop after writing this many shards; None processes the whole split.
    """
    create_data_cache_dir(output_dir)
    split = "train"

    mode = "streaming" if stream else "full download"
    logger.info(f"Loading dataset: {dataset_name} | config: {remote_name} | split: {split} | mode: {mode}")
    if max_shards is not None:
        logger.info(f"Will stop after {max_shards} shard(s) of {shard_size} tokens each.")
    try:
        dataset = load_dataset(dataset_name, name=remote_name, split=split, streaming=stream)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return False

    nprocs = 1 + os.cpu_count() // 2
    logger.info(f"Number of processors: {nprocs}")

    try:
        with mp.Pool(nprocs, initializer=init_gpt2_tokenizer) as pool:
            shard_index = 0
            all_tokens_np = np.empty((shard_size,), dtype=np.uint16)
            token_count = 0
            progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")

            for tokens in pool.imap(gpt2_encode_hf, dataset, chunksize=16):
                n_tokens = len(tokens)
                if token_count + n_tokens < shard_size:
                    all_tokens_np[token_count : token_count + n_tokens] = tokens
                    token_count += n_tokens
                    progress_bar.update(n_tokens)
                else:
                    remainder = shard_size - token_count
                    all_tokens_np[token_count : token_count + remainder] = tokens[:remainder]
                    write_datafile(Path(output_dir) / f"shard_{shard_index:06d}_{split}.npy", all_tokens_np)

                    shard_index += 1
                    if max_shards is not None and shard_index >= max_shards:
                        break
                    progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
                    token_count = n_tokens - remainder
                    all_tokens_np[:token_count] = tokens[remainder:]

            # When we stop early at max_shards, skip the trailing partial shard (it would exceed the
            # cap); still write it on a natural end-of-stream so no tokens are dropped.
            if token_count != 0 and (max_shards is None or shard_index < max_shards):
                write_datafile(Path(output_dir) / f"shard_{shard_index:06d}_{split}.npy", all_tokens_np[:token_count])

            progress_bar.close()

        val_shards(output_dir, split_ratio)
    except Exception as e:
        logger.error(f"Error during tokenization: {e}")
        progress_bar.close()
        return False

    logger.info("Tokenization complete and data saved.")
    return True


def _resolve_output_dir(tokenized_data: str) -> Path:
    """Resolve a relative output dir against the repo root, not the CWD.

    Running this script from inside dataspace/src used to write shards to
    dataspace/src/dataspace/fineweb; anchoring to the repo root keeps them in
    the configured dataspace/fineweb regardless of the working directory.
    """
    output_dir = Path(tokenized_data)
    if output_dir.is_absolute():
        return output_dir
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and tokenize a HF dataset into .npy token shards.")
    parser.add_argument(
        "--stream",
        action="store_true",
        help=(
            "Stream the dataset (datasets streaming=True) instead of downloading it in full. "
            "Combine with --max-shards to pull only a small slice (ideal for Colab / limited disk)."
        ),
    )
    parser.add_argument(
        "--max-shards",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Stop after writing N shards (shard_size tokens each); default processes the whole split. "
            "Use N>=2 so a train shard remains after the validation split is carved off."
        ),
    )
    args = parser.parse_args()
    if args.max_shards is not None and args.max_shards < 1:
        parser.error("--max-shards must be >= 1")

    data_config = DataConfig()

    ok = download_and_tokenize(
        dataset_name=data_config.dataset_name,
        remote_name=data_config.remote_name,
        split_ratio=data_config.split_ratio,
        output_dir=str(_resolve_output_dir(data_config.tokenized_data)),
        shard_size=data_config.shard_size,
        stream=args.stream,
        max_shards=args.max_shards,
    )

    sys.stdout.flush()
    sys.stderr.flush()
    if args.stream:
        # HF Hub's streaming clients (Xet/httpx) keep background threads alive. Once the shards are
        # safely on disk, those threads can abort during normal interpreter shutdown
        # (PyGILState_Release / "Bad file descriptor"). Exit promptly, skipping finalizers, so a
        # successful streaming run doesn't end in a noisy core dump.
        os._exit(0 if ok else 1)
    sys.exit(0 if ok else 1)

    # # change the number of shards for validation set if desire without downloading
    # val_shards(data_config.tokenized_data, .95)
