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
            ("model", "position", "rope"),
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
