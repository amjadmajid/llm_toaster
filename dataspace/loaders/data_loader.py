import os
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_tokens(filepath):
    """Load tokenized data from a .npy file."""
    return np.load(filepath).astype(np.int32)

class DataLoaderLite:
    def __init__(self, B, T, process_rank=0, num_processes=1, split="train", data_root="dataspace/fineweb", master_process=True):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = split
        assert split in {'train', 'val'}

        # get the shard filenames
        shards = os.listdir(data_root) # list all files in the data directory
        # filter out the files that (1) don't have the split (e.g., train or val) in their names and (2) are not .npy files
        shards = [s for s in shards if split in s and s.endswith('.npy')]
        shards = sorted(shards)
        # prepend the data root to the shard filenames
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"No shards found for split {split}"
        # in a multi-process setup, only the master process logs information to avoid duplications
        if master_process:
            logger.info(f"Found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        """Initialize or reset the state to start from the first shard."""
        self.current_shard = 0
        if self.split == 'train': # TODO this is a hack that MUST BE REMOVED
            self.current_shard = 13

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
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance_shard()
        return x, y

    def advance_shard(self):
        """Advance to the next shard and reset the position."""
        self.current_shard = (self.current_shard + 1) % len(self.shards)
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank
        logger.info(f"current shard: {self.current_shard}")
