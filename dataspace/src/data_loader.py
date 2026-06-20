import logging
import os

import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_tokens(filepath):
    """Load tokenized data from a binary .npy shard or a text token shard.

    Text shards are useful for tiny repository fixtures because GitHub renders
    and diffs them normally. They may contain integer token IDs separated by
    whitespace or commas.
    """
    try:
        if filepath.endswith(".npy"):
            return np.load(filepath).astype(np.int32)
        with open(filepath, "r", encoding="utf-8") as handle:
            raw = handle.read().replace(",", " ").split()
        return np.asarray([int(token) for token in raw], dtype=np.int32)
    except Exception as e:
        logger.error(f"Error loading tokens from {filepath}: {e}")
        raise


class DataLoaderLite:
    def __init__(
        self,
        B,
        T,
        current_shard=0,
        process_rank=0,
        num_processes=1,
        split="train",
        data_root="dataspace/fineweb",
        master_process=True,
    ):
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
        assert split in {"train", "val"}, "split must be 'train' or 'val'"

        # Get the shard filenames
        try:
            shards = os.listdir(data_root)
        except FileNotFoundError:
            raise FileNotFoundError(f"The directory {data_root} does not exist")

        # Filter for the appropriate split and supported token shard files
        supported_exts = (".npy", ".txt", ".tokens")
        shards = [s for s in shards if split in s and s.endswith(supported_exts)]
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
        buf = self.tokens[self.current_position : self.current_position + B * T + 1]

        if len(buf) < B * T + 1:
            self.advance_shard()
            buf = self.tokens[self.current_position : self.current_position + B * T + 1]

        x = buf[:-1].reshape(B, T)  # inputs
        y = buf[1:].reshape(B, T)  # targets

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


class InstructionDataLoader:
    """Small JSONL supervised fine-tuning loader for instruction/response pairs.

    Each JSONL row may contain either {"instruction": ..., "response": ...},
    {"prompt": ..., "completion": ...}, or a single {"text": ...} field. Prompt
    tokens are masked with -100 by default so loss is only applied to responses.
    """

    def __init__(
        self,
        B,
        T,
        dataset_path,
        encode_fn,
        prompt_template,
        response_template,
        train_on_prompt=False,
        seed=1337,
        shuffle=True,
    ):
        import json
        import random

        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Instruction dataset does not exist: {dataset_path}")
        self.B = B
        self.T = T
        self.encode_fn = encode_fn
        self.prompt_template = prompt_template
        self.response_template = response_template
        self.train_on_prompt = train_on_prompt
        self.rng = random.Random(seed)
        self.shuffle = shuffle
        self.examples = []
        with open(dataset_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                self.examples.append(self._encode_row(row, line_no))
        if not self.examples:
            raise ValueError(f"No training examples found in {dataset_path}")
        self.index = 0
        if self.shuffle:
            self.rng.shuffle(self.examples)

    def _encode_row(self, row, line_no):
        if "text" in row:
            ids = self._encode(row["text"], add_eot=True)
            return ids[: self.T + 1], None
        instruction = row.get("instruction", row.get("prompt"))
        response = row.get("response", row.get("completion"))
        if instruction is None or response is None:
            raise ValueError(f"Line {line_no} must contain text, instruction/response, or prompt/completion")
        prompt = self.prompt_template.format(**row)
        completion = self.response_template.format(**row)
        prompt_ids = self._encode(prompt, add_eot=True)
        response_ids = self._encode(completion, add_eot=False)
        ids = (prompt_ids + response_ids)[: self.T + 1]
        labels = ids[1:] + [-100]
        if not self.train_on_prompt:
            prompt_label_tokens = max(0, min(len(prompt_ids) - 1, len(labels)))
            labels[:prompt_label_tokens] = [-100] * prompt_label_tokens
        return ids, labels

    def _encode(self, text, add_eot):
        try:
            return self.encode_fn(text, add_eot=add_eot).astype(np.int64).tolist()
        except TypeError:
            return self.encode_fn(text).astype(np.int64).tolist()

    def next_batch(self):
        x = np.zeros((self.B, self.T), dtype=np.int64)
        y = np.full((self.B, self.T), -100, dtype=np.int64)
        for b in range(self.B):
            ids, labels = self.examples[self.index]
            self.index += 1
            if self.index >= len(self.examples):
                self.index = 0
                if self.shuffle:
                    self.rng.shuffle(self.examples)
            if len(ids) < 2:
                continue
            source = ids[:-1][: self.T]
            target = labels[: self.T] if labels is not None else ids[1:][: self.T]
            x[b, : len(source)] = source
            y[b, : len(target)] = target
        return x, y, 0
