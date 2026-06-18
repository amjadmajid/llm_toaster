#!/usr/bin/env python
"""Benchmark eager and SDPA attention backends on a tiny synthetic batch."""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llm_toaster.toaster.config import ConfigHandler
from llm_toaster.toaster.models.registry import build_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark attention backends.")
    parser.add_argument("--config", default="config/smoke_test_config.yaml")
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    for backend in ["eager", "sdpa"]:
        config = ConfigHandler.from_yaml(args.config)
        config.attention.backend = backend
        config.model.vocab_size = config.model.vocab_size or 50304
        model = build_model(config).to(config.training.device)
        x = torch.randint(0, config.model.vocab_size, (config.training.batch_size, config.training.seq_len), device=config.training.device)
        start = time.perf_counter()
        for _ in range(args.iters):
            _ = model(x)
        if "cuda" in str(config.training.device):
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        print(f"{backend}: {elapsed / args.iters:.6f}s/iter")


if __name__ == "__main__":
    main()
