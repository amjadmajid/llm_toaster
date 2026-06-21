"""On-device inference benchmark: decode tok/s, time-to-first-token, peak memory, and energy.

Run this *on the target device* (Jetson Nano/NX/AGX or the laptop GPU). Energy is sampled from
Jetson INA rails via ``tegrastats`` or, on a desktop GPU, ``nvidia-smi`` power polling; integrating
power over the timed generation gives joules and joules/token.

Decode uses the model's KV-cache (``generate_cached``): the prompt is prefilled once and each new
token reuses cached keys/values, so the reported decode tok/s reflects a realistic deployment.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
import threading
import time
from pathlib import Path

import torch

from ..config import ConfigHandler
from ..models.registry import build_model
from ..tokenizers import build_tokenizer
from ..training.checkpointing import load_state_dict_any

logger = logging.getLogger(__name__)


def integrate_energy(samples: list[tuple[float, float]]) -> float:
    """Trapezoidal integral of power (watts) over time (seconds) -> joules."""
    if len(samples) < 2:
        return 0.0
    joules = 0.0
    for (t0, w0), (t1, w1) in zip(samples, samples[1:]):
        joules += 0.5 * (w0 + w1) * (t1 - t0)
    return joules


class NullSampler:
    """No energy measurement (e.g. CPU dry-run)."""

    def start(self) -> None:
        pass

    def stop(self) -> list[tuple[float, float]]:
        return []


class _SubprocessSampler:
    """Base: poll a subprocess line stream and parse (elapsed_s, watts) until stopped."""

    def __init__(self, command: list[str], interval_s: float = 0.1):
        self.command = command
        self.interval_s = interval_s
        self._samples: list[tuple[float, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0

    def _parse_watts(self, line: str) -> float | None:  # pragma: no cover - overridden
        raise NotImplementedError

    def _loop(self) -> None:  # pragma: no cover - requires device tooling
        process = subprocess.Popen(self.command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        try:
            for line in process.stdout:
                if self._stop.is_set():
                    break
                watts = self._parse_watts(line)
                if watts is not None:
                    self._samples.append((time.perf_counter() - self._t0, watts))
        finally:
            process.terminate()

    def start(self) -> None:  # pragma: no cover - requires device tooling
        self._samples = []
        self._stop.clear()
        self._t0 = time.perf_counter()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> list[tuple[float, float]]:  # pragma: no cover - requires device tooling
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        return self._samples


class TegrastatsSampler(_SubprocessSampler):
    """Parse Jetson ``tegrastats`` power rails (e.g. 'VDD_IN 4032mW/4032mW')."""

    def __init__(self, interval_ms: int = 100, rail: str = "auto"):
        super().__init__(["tegrastats", "--interval", str(interval_ms)], interval_ms / 1000.0)
        self.rail = rail

    def _parse_watts(self, line: str) -> float | None:
        # Prefer a named total rail; fall back to the first "<NAME> <mW>mW" reading.
        for rail in [self.rail] if self.rail != "auto" else ["VDD_IN", "POM_5V_IN", "VDD_SYS_SOC"]:
            match = re.search(rf"{rail}\s+(\d+)mW", line)
            if match:
                return int(match.group(1)) / 1000.0
        match = re.search(r"([A-Z0-9_]+)\s+(\d+)mW", line)
        return int(match.group(2)) / 1000.0 if match else None


class NvidiaSmiSampler(_SubprocessSampler):
    """Poll ``nvidia-smi`` instantaneous power draw (watts) on a desktop GPU."""

    def __init__(self, interval_s: float = 0.1):
        command = [
            "nvidia-smi",
            "--query-gpu=power.draw",
            "--format=csv,noheader,nounits",
            "-l",
            str(max(1, int(interval_s))),
        ]
        super().__init__(command, interval_s)

    def _parse_watts(self, line: str) -> float | None:
        line = line.strip()
        try:
            return float(line)
        except ValueError:
            return None


def make_sampler(kind: str):
    if kind == "tegrastats":
        return TegrastatsSampler()
    if kind == "nvidia-smi":
        return NvidiaSmiSampler()
    if kind == "auto":  # pragma: no cover - device dependent
        if Path("/usr/bin/tegrastats").exists():
            return TegrastatsSampler()
        if torch.cuda.is_available():
            return NvidiaSmiSampler()
    return NullSampler()


def _peak_mem_bytes(device: str) -> int:
    if "cuda" in str(device):
        return int(torch.cuda.max_memory_allocated())
    import resource  # Linux: ru_maxrss is KB

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def _sync(device: str) -> None:
    if "cuda" in str(device):
        torch.cuda.synchronize()


@torch.no_grad()
def benchmark_generation(
    model,
    tokenizer,
    device: str,
    *,
    prompt: str = "The",
    max_new_tokens: int = 64,
    warmup_tokens: int = 8,
    sampler=None,
    temperature: float = 1.0,
    top_k: int | None = 50,
    use_cache: bool = True,
) -> dict:
    """Measure TTFT, decode throughput, peak memory, and energy for one model.

    ``use_cache`` selects KV-cached decode (the realistic deployment path) vs the full-forward
    reference; measuring both is a fair comparison and itself a reportable result.
    """
    sampler = sampler or NullSampler()
    model.eval()
    decode = model.generate_cached if use_cache else model.generate_text
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    prompt_len = ids.shape[1]

    decode(ids, warmup_tokens, temperature=temperature, top_k=top_k)  # warm caches/JIT
    _sync(device)

    t0 = time.perf_counter()
    decode(ids, 1, temperature=temperature, top_k=top_k)
    _sync(device)
    ttft = time.perf_counter() - t0

    if "cuda" in str(device):
        torch.cuda.reset_peak_memory_stats()
    sampler.start()
    t0 = time.perf_counter()
    out = decode(ids, max_new_tokens, temperature=temperature, top_k=top_k)
    _sync(device)
    total = time.perf_counter() - t0
    samples = sampler.stop()

    generated = int(out.shape[1] - prompt_len)
    energy = integrate_energy(samples)
    return {
        "device": str(device),
        "use_cache": use_cache,
        "max_new_tokens": max_new_tokens,
        "generated_tokens": generated,
        "ttft_s": ttft,
        "total_s": total,
        "decode_tokens_per_sec": generated / total if total > 0 else 0.0,
        "peak_mem_bytes": _peak_mem_bytes(device),
        "energy_joules": energy,
        "energy_per_token": (energy / generated) if (energy and generated) else None,
    }


def run_bench(args) -> dict:  # pragma: no cover - exercised on real devices
    config = ConfigHandler.from_yaml(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config.training.device = device
    tokenizer = build_tokenizer(config)
    if config.model.vocab_size is None:
        config.model.vocab_size = tokenizer.vocab_size
    model = build_model(config).to(device)
    if args.checkpoint:
        model.load_state_dict(load_state_dict_any(args.checkpoint, device), strict=False)
    if args.precision == "fp16" and device == "cuda":
        model = model.half()

    result = benchmark_generation(
        model,
        tokenizer,
        device,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        warmup_tokens=args.warmup,
        sampler=make_sampler(args.energy),
        top_k=args.top_k,
        use_cache=not args.no_cache,
    )
    result["device_name"] = args.device_name
    result["precision"] = args.precision
    out_path = Path(args.out or f"results/{args.device_name}_{args.precision}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result) + "\n")
    logger.info("device=%s %s", args.device_name, result)
    return result


def main() -> None:  # pragma: no cover - CLI on real devices
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="On-device inference benchmark (run on the target).")
    parser.add_argument("--config", required=True, help="Model config YAML.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint or .llm (random init if omitted).")
    parser.add_argument("--device-name", required=True, help="Label, e.g. jetson-nano, jetson-agx, laptop.")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    parser.add_argument("--prompt", default="The")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--energy", choices=["auto", "tegrastats", "nvidia-smi", "none"], default="auto")
    parser.add_argument("--no-cache", action="store_true", help="Use the full-forward decode instead of the KV-cache.")
    parser.add_argument("--out", default=None, help="Output JSONL (default results/<device>_<precision>.jsonl).")
    run_bench(parser.parse_args())


if __name__ == "__main__":
    main()
