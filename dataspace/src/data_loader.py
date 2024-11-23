# data_loader.py

import os
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
from tokenizer_lib import init_tokenizer
import logging
import argparse
from config import ConfigHandler
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QAIterableDataset(IterableDataset):
    """
    An IterableDataset that loads one shard at a time and yields data samples for a Q&A dataset.
    """

    def __init__(self, data_dir, split_name, tokenizer, seq_length):
        """
        Initializes the QAIterableDataset.

        Args:
            data_dir (str): Directory containing the tokenized data shards.
            split_name (str): Data split to use ('train' or 'validation').
            tokenizer (GPT2Tokenizer): The tokenizer used.
            seq_length (int): Maximum sequence length for padding/truncation.
        """
        super(QAIterableDataset).__init__()
        self.tokenizer = tokenizer
        self.seq_length = seq_length

        # Collect all shard files for the specified split
        logger.info(os.curdir)
        self.shard_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(f'_{split_name}.npy')]
        self.shard_files = sorted(self.shard_files)
        if not self.shard_files:
            raise FileNotFoundError(f"No shard files found for split '{split_name}' in directory '{data_dir}'")

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            # Single-process data loading
            shard_indices = range(len(self.shard_files))
        else:
            # In a worker process; split workload among workers
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            shard_indices = range(worker_id, len(self.shard_files), num_workers)

        for shard_idx in shard_indices:
            shard_file = self.shard_files[shard_idx]
            logger.info(f"Worker {worker_info.id if worker_info else 0} loading shard: {shard_file}")
            shard_data = np.load(shard_file, allow_pickle=True)

            # Shuffle sequences within the shard
            indices = np.random.permutation(len(shard_data))
            for idx in indices:
                tokens = shard_data[idx]

                # Truncate sequence to maximum length
                tokens = tokens[:self.seq_length]

                # Create attention mask before padding
                attention_mask = [1] * len(tokens)

                # Pad sequence if shorter than seq_length
                padding_length = self.seq_length - len(tokens)
                if padding_length > 0:
                    tokens.extend([self.tokenizer.pad_token_id] * padding_length)
                    attention_mask.extend([0] * padding_length)

                # Convert to torch tensors
                input_ids = torch.tensor(tokens, dtype=torch.long)
                attention_mask = torch.tensor(attention_mask, dtype=torch.long)
                labels = input_ids.clone()

                yield {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': labels
                }

def get_dataloader(data_dir, split_name, tokenizer, seq_length, batch_size, num_workers=0):
    """
    Creates a DataLoader for the QAIterableDataset.

    Args:
        data_dir (str): Directory containing the tokenized data shards.
        split_name (str): Data split to use ('train' or 'validation').
        tokenizer (GPT2Tokenizer): The tokenizer used.
        seq_length (int): Maximum sequence length for padding/truncation.
        batch_size (int): Batch size for the DataLoader.
        num_workers (int): Number of subprocesses to use for data loading.

    Returns:
        DataLoader: A PyTorch DataLoader instance for the dataset.
    """
    dataset = QAIterableDataset(data_dir, split_name, tokenizer, seq_length)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True
    )
    return dataloader

if __name__ == '__main__':

    ## Initial Test

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
    data_dir = config.data.tokenized_data 
    output_dir = config.data.tokenized_data
    seq_length = config.training.seq_len
    batch_size = config.training.batch_size


    # Create the DataLoader
    train_loader = get_dataloader(data_dir, 'train', tokenizer, seq_length, batch_size, num_workers=2)

    # Iterate over the DataLoader
    for batch in train_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        print(f"Input IDs shape: {input_ids.shape}")
        print(f"Attention Mask shape: {attention_mask.shape}")
        print(f"Labels shape: {labels.shape}")
        break  # Exit after one batch for testing
