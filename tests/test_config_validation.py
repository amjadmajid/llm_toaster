"""Config validation and legacy backward-compatibility coverage.

These tests intentionally avoid importing torch so they run in minimal
environments; ConfigHandler only touches torch lazily for device detection.
"""

import tempfile
import unittest
from pathlib import Path

from llm_toaster.toaster.config import ConfigError, ConfigHandler


class ConfigValidationTests(unittest.TestCase):
    def test_default_config_is_valid(self):
        ConfigHandler().validate()  # should not raise

    def test_n_embd_must_be_divisible_by_n_head(self):
        config = ConfigHandler()
        config.model.n_head = 7  # 768 % 7 != 0
        with self.assertRaises(ValueError):
            config.validate()

    def test_invalid_enums_are_rejected(self):
        for section, field, value in [
            ("training", "mode", "nonsense"),
            ("model", "norm", "groupnorm"),
            ("model", "ffn", "relu"),
            ("attention", "backend", "made_up"),
            ("optimizer", "name", "sgd"),
            ("scheduler", "name", "step"),
            ("distributed", "mixed_precision", "fp8"),
        ]:
            config = ConfigHandler()
            setattr(getattr(config, section), field, value)
            with self.assertRaises(ValueError, msg=f"{section}.{field}={value}"):
                config.validate()

    def test_sliding_window_is_not_implemented(self):
        config = ConfigHandler()
        config.attention.sliding_window = 256
        with self.assertRaises(NotImplementedError):
            config.validate()

    def test_unimplemented_options_are_rejected_at_validate(self):
        # Spelled correctly (pass the enum checks) but unimplemented -> must fail early.
        cases = [
            ("distributed", "backend", "ddp"),
            ("distributed", "backend", "accelerate"),
            ("model", "ffn", "moe"),
            ("attention", "backend", "flash_attn_2"),
            ("attention", "backend", "xformers"),
            ("tokenizer", "type", "sentencepiece"),
        ]
        for section, field, value in cases:
            config = ConfigHandler()
            setattr(getattr(config, section), field, value)
            with self.assertRaises(NotImplementedError, msg=f"{section}.{field}={value}"):
                config.validate()

    def test_fp16_is_accepted_now_that_scaler_exists(self):
        config = ConfigHandler()
        config.distributed.mixed_precision = "fp16"
        config.validate()  # implemented in Phase 1; must not raise


class BackwardCompatibilityTests(unittest.TestCase):
    def test_legacy_training_fields_mirror_into_new_sections(self):
        config = ConfigHandler.from_yaml("config/default_config.yaml")
        # Model dims live under training.* in legacy YAML and must mirror to model.*
        self.assertEqual(config.model.n_embd, config.training.n_embd)
        self.assertEqual(config.model.n_head, config.training.n_head)
        # lr / logging / evaluation / tokenizer mirrors
        self.assertEqual(config.optimizer.lr, config.training.lr)
        self.assertEqual(config.logging.log_every_steps, config.training.log_inter)
        self.assertEqual(config.evaluation.eval_every_steps, config.training.eval_inter)
        self.assertEqual(config.tokenizer.type, "tiktoken")

    def test_roundtrip_to_yaml_reloads(self):
        config = ConfigHandler.from_yaml("config/default_config.yaml")
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cfg.yaml"
            config.to_yaml(str(path))
            reloaded = ConfigHandler.from_yaml(str(path))
            self.assertEqual(reloaded.model.n_embd, config.model.n_embd)
            self.assertEqual(reloaded.training.mode, config.training.mode)


class DefaultsRoundTripTests(unittest.TestCase):
    """Issue #3 confirmation: sections use field(default_factory=...), so defaults are
    independent per instance and survive a full YAML round trip."""

    def test_sections_are_not_shared_between_instances(self):
        # The classic mutable-default bug would make two ConfigHandler() share one
        # TrainingConfig/list; field(default_factory=...) must give each its own.
        a, b = ConfigHandler(), ConfigHandler()
        self.assertIsNot(a.training, b.training)
        self.assertIsNot(a.peft.target_modules, b.peft.target_modules)
        a.model.n_embd = 123
        a.peft.target_modules.append("__mutated__")
        self.assertNotEqual(b.model.n_embd, 123)
        self.assertNotIn("__mutated__", b.peft.target_modules)

    def test_bare_defaults_survive_yaml_round_trip(self):
        config = ConfigHandler()
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "defaults.yaml"
            config.to_yaml(str(path))
            reloaded = ConfigHandler.from_yaml(str(path))
        # Every default value must come back unchanged (no dropped/mutated fields).
        self.assertEqual(reloaded.to_dict(), ConfigHandler().to_dict())


class NestedDataConfigTests(unittest.TestCase):
    """Stage: orthogonal nested data config (source/transform/materialization/sampling/validation)."""

    def _write(self, td, text):
        path = Path(td) / "cfg.yaml"
        path.write_text(text, encoding="utf-8")
        return str(path)

    def test_nested_sections_parse(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write(
                td,
                "data:\n"
                "  manifest_path: /tmp/m.json\n"
                "  source:\n    dataset_name: foo/bar\n    config_name: x\n"
                "  transform:\n    shard_tokens: 123\n    dtype: uint32\n"
                "  materialization:\n    mode: prepared\n    store_dir: /tmp/store\n"
                "  sampling:\n    exhaustion: repeat\n    shuffle: shard\n"
                "  validation:\n    shards: 2\n",
            )
            cfg = ConfigHandler.from_yaml(path)
            self.assertEqual(cfg.data.source.dataset_name, "foo/bar")
            self.assertEqual(cfg.data.transform.shard_tokens, 123)
            self.assertEqual(cfg.data.transform.dtype, "uint32")
            self.assertEqual(cfg.data.sampling.exhaustion, "repeat")
            self.assertEqual(cfg.data.validation.shards, 2)
            self.assertFalse(cfg._data_is_legacy)

    def test_unknown_nested_key_names_full_path(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write(td, "data:\n  materialization:\n    mode: prepared\n    bogus_knob: 1\n")
            with self.assertRaises(ConfigError) as ctx:
                ConfigHandler.from_yaml(path)
            message = str(ctx.exception)
            self.assertIn("materialization", message)
            self.assertIn("bogus_knob", message)

    def test_non_mapping_nested_section_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write(td, "data:\n  source: 5\n")
            with self.assertRaises(ConfigError):
                ConfigHandler.from_yaml(path)

    def test_nested_round_trip(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write(td, "data:\n  materialization:\n    mode: prefetch\n  source:\n    dataset_name: a/b\n")
            cfg = ConfigHandler.from_yaml(path)
            out = Path(td) / "out.yaml"
            cfg.to_yaml(str(out))
            reloaded = ConfigHandler.from_yaml(str(out))
            self.assertEqual(reloaded.data.materialization.mode, "prefetch")
            self.assertEqual(reloaded.data.source.dataset_name, "a/b")


class LegacyDataDeprecationTests(unittest.TestCase):
    def test_legacy_data_config_warns_and_translates(self):
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cfg = ConfigHandler.from_yaml("config/smoke_test_config.yaml")
        self.assertTrue(any(issubclass(w.category, DeprecationWarning) for w in caught))
        self.assertTrue(cfg._data_is_legacy)
        # Legacy fields are mirrored into the new structure.
        self.assertEqual(cfg.data.source.dataset_name, cfg.data.dataset_name)
        self.assertEqual(cfg.data.materialization.store_dir, cfg.training.data_dir)

    def test_new_style_config_does_not_warn(self):
        import warnings

        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "cfg.yaml"
            path.write_text("data:\n  materialization:\n    mode: prepared\n", encoding="utf-8")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                cfg = ConfigHandler.from_yaml(str(path))
            self.assertFalse(any(issubclass(w.category, DeprecationWarning) for w in caught))
            self.assertFalse(cfg._data_is_legacy)


class DataValidationRulesTests(unittest.TestCase):
    def test_prefetch_requires_materializable_source(self):
        cfg = ConfigHandler()
        cfg.data.materialization.mode = "prefetch"
        cfg.data.source.dataset_name = ""
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_direct_requires_fixed_validation_and_single_process(self):
        cfg = ConfigHandler()
        cfg.data.materialization.mode = "direct"
        with self.assertRaises(ValueError):  # no validation.manifest_path
            cfg.validate()
        cfg.data.validation.manifest_path = "/tmp/val.json"
        cfg.training.num_workers = 2
        with self.assertRaises(ValueError):  # workers must be 0
            cfg.validate()
        cfg.training.num_workers = 0
        cfg.validate()  # now valid

    def test_wait_requires_prefetch_or_external_producer(self):
        cfg = ConfigHandler()
        cfg.data.sampling.exhaustion = "wait"
        cfg.data.materialization.mode = "prepared"
        with self.assertRaises(ValueError):
            cfg.validate()
        cfg.data.materialization.external_producer = True
        cfg.validate()  # external producer declared -> allowed

    def test_uint16_requires_small_vocab(self):
        cfg = ConfigHandler()
        cfg.model.vocab_size = 70000
        cfg.data.transform.dtype = "uint16"
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_validation_tokens_xor_shards(self):
        cfg = ConfigHandler()
        cfg.data.validation.tokens = 1000
        cfg.data.validation.shards = 2
        with self.assertRaises(ValueError):
            cfg.validate()

    def test_max_tokens_must_fit_one_step(self):
        cfg = ConfigHandler()
        cfg.training.batch_size = 4
        cfg.training.seq_len = 8
        cfg.training.n_batches = 2  # one step = 64 tokens
        cfg.training.max_tokens = 10
        with self.assertRaises(ValueError):
            cfg.validate()
        cfg.training.max_tokens = 640
        cfg.validate()


class ConfigLoadingErrorTests(unittest.TestCase):
    def _write(self, td, text):
        path = Path(td) / "cfg.yaml"
        path.write_text(text, encoding="utf-8")
        return str(path)

    def test_unknown_section_names_the_section(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write(td, "nonsense:\n  x: 1\n")
            with self.assertRaises(ConfigError) as ctx:
                ConfigHandler.from_yaml(path)
            self.assertIn("nonsense", str(ctx.exception))

    def test_unknown_key_names_key_and_section(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write(td, "training:\n  batch_size: 2\n  bogus_key: 1\n")
            with self.assertRaises(ConfigError) as ctx:
                ConfigHandler.from_yaml(path)
            message = str(ctx.exception)
            self.assertIn("bogus_key", message)
            self.assertIn("training", message)

    def test_non_mapping_section_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write(td, "training: 5\n")
            with self.assertRaises(ConfigError):
                ConfigHandler.from_yaml(path)

    def test_validation_error_includes_filepath(self):
        with tempfile.TemporaryDirectory() as td:
            path = self._write(td, "training:\n  n_embd: 10\n  n_head: 3\n")
            with self.assertRaises(ValueError) as ctx:
                ConfigHandler.from_yaml(path)
            self.assertIn(path, str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
