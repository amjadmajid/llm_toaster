import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - dependency guard for minimal CI images
    np = None

from config import ConfigHandler

if np is not None:
    from dataspace.src.data_loader import InstructionDataLoader
else:
    InstructionDataLoader = None


def fake_encode(text, dtype=None, add_eot=True):
    if np is None:
        raise unittest.SkipTest("numpy is required for fake token arrays")
    values = [999] if add_eot else []
    values.extend((ord(ch) % 200) + 1 for ch in text)
    return np.array(values, dtype=dtype or np.uint16)


class ConfigAndDataTests(unittest.TestCase):
    def test_smoke_config_loads_finetune_fields(self):
        config = ConfigHandler.from_yaml("config/smoke_test_config.yaml")
        self.assertEqual(config.training.batch_size, 2)
        self.assertEqual(config.finetune.dataset_path, "tests/fixtures/instruction_data.jsonl")
        self.assertFalse(config.finetune.shuffle)

    @unittest.skipIf(np is None, "numpy is required for InstructionDataLoader tests")
    def test_instruction_loader_masks_prompt_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.jsonl"
            path.write_text('{"instruction":"A","response":"B"}\n', encoding="utf-8")
            loader = InstructionDataLoader(
                1,
                32,
                str(path),
                fake_encode,
                "Q:{instruction}\nA:",
                "{response}",
                train_on_prompt=False,
                shuffle=False,
            )
            x, y, _ = loader.next_batch()
            self.assertEqual(x.shape, (1, 32))
            self.assertEqual(y.shape, (1, 32))
            self.assertIn(-100, y[0].tolist())
            self.assertTrue(any(token != -100 for token in y[0].tolist()))

    @unittest.skipIf(np is None, "numpy is required for InstructionDataLoader tests")
    def test_instruction_loader_supports_text_schema(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "data.jsonl"
            path.write_text('{"text":"plain sample"}\n', encoding="utf-8")
            loader = InstructionDataLoader(1, 16, str(path), fake_encode, "", "", shuffle=False)
            x, y, _ = loader.next_batch()
            self.assertEqual(x.shape, (1, 16))
            self.assertTrue(any(token != -100 for token in y[0].tolist()))


if __name__ == "__main__":
    unittest.main()
