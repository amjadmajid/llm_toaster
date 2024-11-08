import os
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_tokens(filepath):
    """Load tokenized data from a .npy file."""
    try:
        return np.load(filepath).astype(np.int32)
    except Exception as e:
        logger.error(f"Error loading tokens from {filepath}: {e}")
        raise

class DataLoaderLite:
    def __init__(self, B, T, current_shard=0, process_rank=0, num_processes=1, split="train", data_root="dataspace/fineweb", master_process=True):
        """
        Initialize the DataLoaderLite.

        Parameters:
        - B: Batch size.
        - T: Sequence length.
        - process_rank: Rank of the current process in a multi-process setup.
        - num_processes: Total number of processes.
        - split: Data split to use ('train' or 'val').
        - data_root: Root directory of the data.
        - master_process: Boolean indicating if the current process is the master process.
        """
        if not isinstance(B, int) or B <= 0:
            raise ValueError("Batch size B must be a positive integer")
        if not isinstance(T, int) or T <= 0:
            raise ValueError("Sequence length T must be a positive integer")

        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        assert split in {'train', 'val'}, "split must be 'train' or 'val'"

        # Get the shard filenames
        try:
            shards = os.listdir(data_root)
        except FileNotFoundError:
            raise FileNotFoundError(f"The directory {data_root} does not exist")
        
        # Filter for the appropriate split and .npy files
        shards = [s for s in shards if split in s and s.endswith('.npy')]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"No shards found for split {split}"

        if master_process:
            logger.info(f"Found {len(shards)} shards for split {split}")
        
        self.current_shard = current_shard
        self.reset()

    def reset(self):
        """Initialize or reset the state to start from the first shard."""
        logger.info(f"{self.current_shard=}")
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        """Get the next batch of data."""
        B, T = self.B, self.T
        buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        
        if len(buf) < B * T + 1:
            self.advance_shard()
            buf = self.tokens[self.current_position:self.current_position + B * T + 1]
        
        x = buf[:-1].reshape(B, T)  # inputs
        y = buf[1:].reshape(B, T)   # targets
        
        # Advance the position in the tensor
        self.current_position += B * T * self.num_processes
        
        # If loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance_shard()
        
        return x, y, self.current_shard

    def advance_shard(self):
        """Advance to the next shard and reset the position."""
        self.current_shard = (self.current_shard + 1) % len(self.shards)
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
        logger.info(f"Advanced to shard {self.current_shard}")

    def shuffle_shards(self):
        """Shuffle the order of shards."""
        np.random.shuffle(self.shards)
        logger.info("Shuffled shards")

# Example usage of the improved DataLoaderLite
if __name__ == "__main__":
    data_loader = DataLoaderLite(B=32, T=128, split="train", data_root="dataspace/fineweb", master_process=True)
    data_loader.shuffle_shards()
    for _ in range(10):
        x, y, _ = data_loader.next_batch()
        print(x.shape, y.shape)
