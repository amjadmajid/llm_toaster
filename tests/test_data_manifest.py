"""Manifest + packing: schema round-trip, deterministic hashing, append-only, validation.

Torch-free; needs only numpy. No network.
"""

import json
import tempfile
import unittest
from pathlib import Path

from llm_toaster.toaster.data.errors import ManifestError, ManifestVersionError
from llm_toaster.toaster.data.manifest import (
    MANIFEST_FORMAT_VERSION,
    Manifest,
    ShardEntry,
    SourceSpec,
    TokenizerSpec,
    TransformSpec,
    load_manifest,
    resolve_shard_path,
)
from llm_toaster.toaster.data.packing import (
    TokenPacker,
    canonical_json,
    check_dtype_fits_vocab,
    hash_canonical,
    make_batch,
    tokenizer_fingerprint,
)


class FakeTokenizer:
    eos_token_id = 7
    bos_token_id = None
    vocab_size = 256

    def __init__(self, name="fake"):
        self.name = name

    def encode(self, text, add_special_tokens=False):
        return [(ord(c) % 200) + 1 for c in text]


def _manifest(**overrides):
    kwargs = dict(
        dataset_id="fineweb-edu-gpt2-v1",
        source=SourceSpec(dataset_name="HuggingFaceFW/fineweb-edu", resolved_revision="abc123"),
        tokenizer=TokenizerSpec(type="tiktoken", name="gpt2", vocab_size=256, fingerprint="sha256:deadbeef"),
        transform=TransformSpec(shard_tokens=100, dtype="uint16"),
    )
    kwargs.update(overrides)
    return Manifest(**kwargs)


def _entry(index, split="train", tokens=100):
    sid = f"{split}-{index:06d}"
    return ShardEntry(
        id=sid,
        index=index,
        split=split,
        path=f"shards/{sid}.npy",
        tokens=tokens,
        dtype="uint16",
        bytes=tokens * 2 + 128,
        sha256="0" * 64,
    )


class PackingTests(unittest.TestCase):
    def test_canonical_json_is_stable_and_sorted(self):
        self.assertEqual(canonical_json({"b": 1, "a": 2}), '{"a":2,"b":1}')
        self.assertEqual(hash_canonical({"a": 1, "b": 2}), hash_canonical({"b": 2, "a": 1}))

    def test_tokenizer_fingerprint_distinguishes_behavior(self):
        a = tokenizer_fingerprint(FakeTokenizer())

        class Other(FakeTokenizer):
            def encode(self, text, add_special_tokens=False):
                return [1 for _ in text]  # different behavior

        self.assertTrue(a.startswith("sha256:"))
        self.assertEqual(a, tokenizer_fingerprint(FakeTokenizer()))  # deterministic
        self.assertNotEqual(a, tokenizer_fingerprint(Other()))

    def test_dtype_vocab_guard(self):
        check_dtype_fits_vocab("uint16", 50257)  # ok
        with self.assertRaises(ValueError):
            check_dtype_fits_vocab("uint16", 70000)

    def test_packer_prepends_eot(self):
        packer = TokenPacker(FakeTokenizer(), add_eot=True, dtype_name="uint16")
        arr = packer.encode_document("hi")
        self.assertEqual(int(arr[0]), 7)  # eot prepended
        self.assertEqual(len(arr), 3)

    def test_make_batch_single_shift(self):
        import numpy as np

        tokens = np.arange(10, 30)
        x, y = make_batch(tokens, 0, batch_size=1, seq_len=3)
        self.assertEqual(x.tolist(), [[10, 11, 12]])
        self.assertEqual(y.tolist(), [[11, 12, 13]])
        self.assertEqual(y[0][:-1].tolist(), x[0][1:].tolist())

    def test_make_batch_short_window_raises(self):
        import numpy as np

        with self.assertRaises(ValueError):
            make_batch(np.arange(3), 0, batch_size=2, seq_len=8)


class ManifestRoundTripTests(unittest.TestCase):
    def test_roundtrip_through_dict(self):
        m = _manifest()
        m.append_shard(_entry(0))
        m.append_shard(_entry(1))
        again = Manifest.from_dict(json.loads(json.dumps(m.to_dict())))
        self.assertEqual(again.to_dict(), m.to_dict())

    def test_deterministic_dataset_fingerprint(self):
        self.assertEqual(_manifest().dataset_fingerprint, _manifest().dataset_fingerprint)
        changed = _manifest(source=SourceSpec(dataset_name="other", resolved_revision="zzz"))
        self.assertNotEqual(_manifest().dataset_fingerprint, changed.dataset_fingerprint)

    def test_relative_path_resolution(self):
        with tempfile.TemporaryDirectory() as td:
            manifest_path = Path(td) / "manifest.json"
            entry = _entry(0)
            resolved = resolve_shard_path(manifest_path, entry)
            self.assertEqual(resolved, (Path(td) / "shards" / "train-000000.npy"))

    def test_save_and_load_atomic(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "manifest.json"
            m = _manifest()
            m.append_shard(_entry(0))
            m.save_atomic(path)
            loaded = load_manifest(path)
            self.assertEqual(loaded.dataset_id, "fineweb-edu-gpt2-v1")
            self.assertEqual(len(loaded.split("train").shards), 1)


class ManifestAppendOnlyTests(unittest.TestCase):
    def test_generation_increments_on_append(self):
        m = _manifest()
        self.assertEqual(m.generation, 0)
        m.append_shard(_entry(0))
        self.assertEqual(m.generation, 1)
        m.append_shard(_entry(1))
        self.assertEqual(m.generation, 2)
        m.mark_complete("train")
        self.assertEqual(m.generation, 3)

    def test_split_token_counter_tracks_shards(self):
        m = _manifest()
        m.append_shard(_entry(0, tokens=100))
        m.append_shard(_entry(1, tokens=50))
        self.assertEqual(m.split("train").tokens, 150)

    def test_duplicate_id_rejected(self):
        m = _manifest()
        m.append_shard(_entry(0))
        dup = _entry(1)
        dup.id = "train-000000"
        with self.assertRaises(ManifestError):
            m.append_shard(dup)

    def test_duplicate_path_rejected(self):
        m = _manifest()
        m.append_shard(_entry(0))
        dup = _entry(1)
        dup.path = "shards/train-000000.npy"
        with self.assertRaises(ManifestError):
            m.append_shard(dup)

    def test_non_contiguous_index_rejected(self):
        m = _manifest()
        m.append_shard(_entry(0))
        with self.assertRaises(ManifestError):
            m.append_shard(_entry(5))

    def test_append_to_complete_split_rejected(self):
        m = _manifest()
        m.append_shard(_entry(0))
        m.mark_complete("train")
        with self.assertRaises(ManifestError):
            m.append_shard(_entry(1))

    def test_prefix_fingerprint_stable_across_later_appends(self):
        m = _manifest()
        m.append_shard(_entry(0))
        before = m.prefix_fingerprint("train", "train-000000")
        m.append_shard(_entry(1))  # appending more must not change the earlier prefix
        after = m.prefix_fingerprint("train", "train-000000")
        self.assertEqual(before, after)
        self.assertNotEqual(after, m.prefix_fingerprint("train", "train-000001"))


class ManifestValidationTests(unittest.TestCase):
    def test_unsupported_format_version_rejected(self):
        data = _manifest().to_dict()
        data["format_version"] = MANIFEST_FORMAT_VERSION + 1
        with self.assertRaises(ManifestVersionError):
            Manifest.from_dict(data)

    def test_missing_field_names_key_path(self):
        data = _manifest().to_dict()
        del data["source"]["type"]
        with self.assertRaises(ManifestError) as ctx:
            Manifest.from_dict(data)
        self.assertIn("source.type", str(ctx.exception))

    def test_unsupported_dtype_in_shard_named(self):
        m = _manifest()
        m.append_shard(_entry(0))
        data = m.to_dict()
        data["splits"]["train"]["shards"][0]["dtype"] = "float64"
        with self.assertRaises(ManifestError) as ctx:
            Manifest.from_dict(data)
        self.assertIn("dtype", str(ctx.exception))

    def test_token_counter_mismatch_rejected(self):
        m = _manifest()
        m.append_shard(_entry(0, tokens=100))
        data = m.to_dict()
        data["splits"]["train"]["tokens"] = 999
        with self.assertRaises(ManifestError) as ctx:
            Manifest.from_dict(data)
        self.assertIn("tokens", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
