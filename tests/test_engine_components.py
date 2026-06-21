import json
import os
import signal
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from llm_toaster.toaster.config import ConfigHandler
from llm_toaster.toaster.data.adapters import DataAdapterRegistry, JsonlSFTDataLoader
from llm_toaster.toaster.generation import generate
from llm_toaster.toaster.models.registry import build_model
from llm_toaster.toaster.peft.lora import inject_lora
from llm_toaster.toaster.training.checkpointing import load_checkpoint, save_checkpoint
from llm_toaster.toaster.training.engine import TrainingEngine
from llm_toaster.toaster.training.optim import build_optimizer, build_scheduler


class FakeTokenizer:
    eos_token_id = 2
    pad_token_id = 0
    bos_token_id = 1
    vocab_size = 128

    def encode(self, s, add_special_tokens=False):
        return ([1] if add_special_tokens else []) + [3 + (ord(c) % 100) for c in s]

    def decode(self, ids):
        return "".join(chr((i - 3) % 100) for i in ids if i > 2)

    def apply_chat_template(self, msgs):
        return "".join(f"{m['role']}: {m['content']}\n" for m in msgs) + "assistant: "


class FakeLoader:
    """Yields random in-vocabulary batches so train_step needs no real data."""

    def __init__(self, B, T, vocab):
        self.B, self.T, self.vocab = B, T, vocab

    def next_batch(self):
        x = np.random.randint(0, self.vocab, size=(self.B, self.T))
        y = np.random.randint(0, self.vocab, size=(self.B, self.T))
        return x, y, 0


def _tiny_engine(td, vocab=128):
    cfg = ConfigHandler.from_yaml("config/smoke_test_config.yaml")
    cfg.training.device = "cpu"
    cfg.model.vocab_size = vocab
    cfg.model.n_embd = 16
    cfg.model.n_head = 4
    cfg.model.n_blocks = 1
    cfg.model.seq_len = 8
    cfg.logging.log_file = os.path.join(td, "log.txt")
    cfg.checkpointing.output_dir = td
    cfg.training.ckpt = os.path.join(td, "base_ckpt")
    cfg.training.ckpt_config = os.path.join(td, "base_config.yaml")
    engine = TrainingEngine(cfg)
    engine.tokenizer = FakeTokenizer()
    engine.model = build_model(cfg)
    engine.train_loader = FakeLoader(cfg.training.batch_size, cfg.model.seq_len, vocab)
    engine.optimizer = build_optimizer(engine.model, cfg)
    engine.scheduler = build_scheduler(engine.optimizer, cfg)
    engine.setup_scaler()
    return engine, cfg


class EngineComponents(unittest.TestCase):
    def test_all_data_formats_and_masking(self):
        rows = [
            {"text": "abc"},
            {"prompt": "p", "completion": "c"},
            {"instruction": "i", "response": "r"},
            {"instruction": "i", "input": "in", "output": "out"},
            {"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]},
            {"conversations": [{"from": "human", "value": "u"}, {"from": "gpt", "value": "a"}]},
            {"prompt": "p", "chosen": "yes", "rejected": "no"},
        ]
        for row in rows:
            prompt, response = DataAdapterRegistry.format_row(row, "auto", FakeTokenizer())
            self.assertIsInstance(prompt, str)
            self.assertIsInstance(response, str)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sft.jsonl"
            p.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
            loader = JsonlSFTDataLoader(2, 32, str(p), FakeTokenizer(), shuffle=False)
            x, y, _ = loader.next_batch()
            self.assertEqual(x.shape, (2, 32))
            self.assertIn(-100, y.tolist()[0])

    def test_model_attention_shapes_checkpoint_and_lora(self):
        cfg = ConfigHandler.from_yaml("config/smoke_test_config.yaml")
        cfg.model.vocab_size = 128
        cfg.model.n_embd = 16
        cfg.model.n_head = 4
        cfg.model.n_blocks = 1
        cfg.model.seq_len = 8
        for backend in ["eager", "sdpa"]:
            cfg.attention.backend = backend
            model = build_model(cfg)
            out = model(torch.ones(2, 8, dtype=torch.long))
            self.assertEqual(out.shape, (2, 8, 128))
        trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
        inject_lora(model, cfg.peft)
        trainable_after = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.assertLess(trainable_after, trainable_before)
        with tempfile.TemporaryDirectory() as td:
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
            path = os.path.join(td, "ckpt.pt")
            save_checkpoint(path, model, opt, config=cfg, global_step=3, tokens_seen=10)
            ck = load_checkpoint(path, model, opt, device="cpu", strict=False)
            self.assertEqual(ck["global_step"], 3)


class EngineTrainingTests(unittest.TestCase):
    def test_scaler_disabled_off_cuda(self):
        with tempfile.TemporaryDirectory() as td:
            engine, _ = _tiny_engine(td)
            self.assertFalse(engine.scaler.is_enabled())

    def test_train_step_advances_and_is_finite(self):
        with tempfile.TemporaryDirectory() as td:
            engine, cfg = _tiny_engine(td)
            loss = engine.train_step()
            self.assertEqual(engine.global_step, 1)
            self.assertTrue(np.isfinite(loss))
            # n_batches batches of B*T tokens were consumed.
            self.assertEqual(engine.tokens_seen, cfg.training.batch_size * cfg.model.seq_len * cfg.training.n_batches)

    def test_checkpoint_resume_restores_progress(self):
        with tempfile.TemporaryDirectory() as td:
            engine, cfg = _tiny_engine(td)
            engine.train_step()
            engine.train_step()
            path = os.path.join(td, "ckpt.pt")
            engine.save_checkpoint(path)
            resumed, _ = _tiny_engine(td)
            self.assertEqual(resumed.global_step, 0)
            resumed.load_checkpoint(path)
            self.assertEqual(resumed.global_step, engine.global_step)
            self.assertEqual(resumed.tokens_seen, engine.tokens_seen)


class EmergencyCheckpointTests(unittest.TestCase):
    def test_interrupt_handler_requests_stop_without_saving_or_raising(self):
        # Issue #9: the signal handler must NOT save/raise in async-signal context -- it only flags
        # the stop, so the save happens at a clean step boundary (not mid-optimizer-step).
        with tempfile.TemporaryDirectory() as td:
            engine, _ = _tiny_engine(td)
            self.assertFalse(engine._stop_requested)
            engine._handle_interrupt(signal.SIGINT, None)  # must not raise
            self.assertTrue(engine._stop_requested)
            self.assertEqual(engine._stop_signum, signal.SIGINT)

    def test_emergency_checkpoint_saved_at_boundary_then_stops(self):
        # Requesting a stop after the first step must save a consistent emergency checkpoint and
        # stop the loop early (well before max_iter).
        from llm_toaster.toaster.training.engine import TrainingEngine

        with tempfile.TemporaryDirectory() as td:
            cfg = ConfigHandler.from_yaml("config/smoke_test_config.yaml")
            cfg.training.device = "cpu"
            cfg.distributed.mixed_precision = "no"
            cfg.training.max_iter = 1000  # large; we interrupt right after the first step
            cfg.evaluation.eval_every_steps = 0
            cfg.logging.log_file = os.path.join(td, "train.log")
            cfg.logging.metrics_file = os.path.join(td, "metrics.jsonl")
            cfg.checkpointing.output_dir = td
            cfg.training.ckpt = os.path.join(td, "base_ckpt")
            cfg.training.ckpt_config = os.path.join(td, "base_config.yaml")

            engine = TrainingEngine(cfg)
            original_step = TrainingEngine.train_step

            def step_then_request_stop(self):
                loss = original_step(self)
                self._handle_interrupt(signal.SIGINT, None)  # simulate Ctrl-C after the step
                return loss

            engine.train_step = step_then_request_stop.__get__(engine, TrainingEngine)
            engine.train()

            self.assertEqual(engine.global_step, 1)  # stopped at the boundary, not max_iter
            emergency = engine._emergency_checkpoint_path()
            self.assertTrue(os.path.exists(emergency))
            checkpoint = load_checkpoint(emergency, engine.model, device="cpu", strict=False)
            self.assertEqual(checkpoint["global_step"], 1)
            self.assertIn("wall_clock_s", checkpoint)


class LabelShiftLossTests(unittest.TestCase):
    """Issue #5: the single-shift contract end to end.

    The dataloader returns already-shifted (x, y); ``train_step`` compares logits to y
    directly with NO second shift (no ``logits[:, :-1]`` / ``y[:, 1:]``). This runs the real
    DataLoaderLite -> model -> loss path and confirms a finite loss with aligned lengths.
    """

    def test_loss_computes_on_dataloader_batch_without_double_shift(self):
        from dataspace.src.data_loader import DataLoaderLite

        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "train_000.txt"), "w", encoding="utf-8") as handle:
                handle.write(" ".join(str(i % 50) for i in range(256)))
            loader = DataLoaderLite(B=2, T=8, split="train", data_root=td)
            x, y, _ = loader.next_batch()

            cfg = ConfigHandler.from_yaml("config/smoke_test_config.yaml")
            cfg.model.vocab_size = 64
            cfg.model.n_embd = 16
            cfg.model.n_head = 4
            cfg.model.n_blocks = 1
            cfg.model.seq_len = 8
            model = build_model(cfg)

            logits = model(torch.as_tensor(x, dtype=torch.long))
            # Exactly mirrors engine.train_step: loss on (logits, y) with no re-slicing.
            loss = torch.nn.CrossEntropyLoss(ignore_index=-100)(
                logits.view(-1, logits.size(-1)), torch.as_tensor(y, dtype=torch.long).view(-1)
            )
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(float(loss.detach()), 0.0)
            # logits and targets align position-for-position (same T) -> no off-by-one.
            self.assertEqual(logits.shape[1], y.shape[1])


class GenerationTests(unittest.TestCase):
    def test_generate_text_supports_top_k_and_top_p(self):
        cfg = ConfigHandler.from_yaml("config/smoke_test_config.yaml")
        cfg.model.vocab_size = 128
        cfg.model.n_embd = 16
        cfg.model.n_head = 4
        cfg.model.n_blocks = 1
        cfg.model.seq_len = 8
        model = build_model(cfg)
        out = model.generate_text(torch.ones(1, 3, dtype=torch.long), 5, temperature=0.8, top_k=10, top_p=0.9)
        self.assertEqual(out.shape, (1, 8))
        self.assertLess(int(out.max()), 128)

    def test_shared_generate_helper_returns_text(self):
        cfg = ConfigHandler.from_yaml("config/smoke_test_config.yaml")
        cfg.model.vocab_size = 128
        cfg.model.n_embd = 16
        cfg.model.n_head = 4
        cfg.model.n_blocks = 1
        cfg.model.seq_len = 8
        text = generate(build_model(cfg), FakeTokenizer(), "hello", "cpu", max_new_tokens=4, top_k=5, top_p=0.9)
        self.assertIsInstance(text, str)


if __name__ == "__main__":
    unittest.main()
