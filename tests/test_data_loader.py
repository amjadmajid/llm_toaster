"""DataLoaderLite shard handling (torch-free; needs numpy only)."""

import os
import tempfile
import unittest

from dataspace.src.data_loader import DataLoaderLite


def _write_shard(directory, name, tokens):
    with open(os.path.join(directory, name), "w", encoding="utf-8") as handle:
        handle.write(" ".join(str(t) for t in tokens))


class DataLoaderLiteTests(unittest.TestCase):
    def test_short_shard_raises_clear_error(self):
        with tempfile.TemporaryDirectory() as td:
            _write_shard(td, "train_000.txt", [1, 2, 3, 4, 5])  # < B*T+1
            loader = DataLoaderLite(B=2, T=8, split="train", data_root=td)
            with self.assertRaises(ValueError) as ctx:
                loader.next_batch()
            self.assertIn("fewer than", str(ctx.exception))

    def test_normal_shard_yields_batches(self):
        with tempfile.TemporaryDirectory() as td:
            _write_shard(td, "train_000.txt", [i % 50 for i in range(200)])
            loader = DataLoaderLite(B=2, T=8, split="train", data_root=td)
            x, y, shard = loader.next_batch()
            self.assertEqual(x.shape, (2, 8))
            self.assertEqual(y.shape, (2, 8))
            self.assertEqual(shard, 0)

    def test_state_dict_roundtrip_restores_read_position(self):
        with tempfile.TemporaryDirectory() as td:
            _write_shard(td, "train_000.txt", list(range(400)))
            _write_shard(td, "train_001.txt", list(range(400, 800)))
            loader = DataLoaderLite(B=2, T=8, split="train", data_root=td)
            for _ in range(5):
                loader.next_batch()
            saved = loader.state_dict()
            expected_x, _, _ = loader.next_batch()  # the batch that should come next

            resumed = DataLoaderLite(B=2, T=8, split="train", data_root=td)  # fresh = shard 0, pos 0
            self.assertEqual(resumed.current_position, 0)
            resumed.load_state_dict(saved)
            self.assertEqual(resumed.current_shard, saved["current_shard"])
            self.assertEqual(resumed.current_position, saved["current_position"])
            got_x, _, _ = resumed.next_batch()
            self.assertTrue((got_x == expected_x).all())  # resumes the exact same data


if __name__ == "__main__":
    unittest.main()
