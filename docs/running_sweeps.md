# Running a controlled architecture sweep

This guide walks through a full architecture-comparison run: train a matched-parameter sweep on a
single GPU, aggregate the results into a Pareto table, and measure the winners on-device. The
methodology is **one variable at a time at matched parameters** — so a result reflects the
architecture choice, not model size.

Tooling used: `scripts/sweep.py`, `scripts/aggregate.py`, `scripts/bench_device.py`,
`scripts/describe_arch.py`, and the sweep specs in `config/sweeps/`.

## 0. Setup

The sweep tooling currently lives on the `feat/architecture-profiling` branch.

```bash
git branch --show-current     # expect: feat/architecture-profiling (or main once merged)
pip install -e ".[viz]"       # matplotlib, for the Pareto plots (optional)
```

## 1. Sanity-check the pipeline (CPU, ~10 s)

```bash
python scripts/sweep.py --spec config/sweeps/smoke.yaml
python scripts/aggregate.py --dir runs/smoke_sweep
```

A 2-row table confirms the train → metrics → aggregate loop works end to end.

## 2. Free the GPU

Stop any other training first — a sweep needs the full GPU, and concurrent runs will compete for
VRAM (and likely OOM).

## 3. Run the sweep

```bash
tmux new -s sweep        # survives terminal/SSH disconnects
systemd-inhibit --what=sleep:idle:handle-lid-switch --mode=block --why="arch sweep" \
  python scripts/sweep.py --spec config/sweeps/pilot_gqa.yaml
```

`config/sweeps/pilot_gqa.yaml` runs the headline **KV-cache axis** — `num_key_value_heads ∈ {8,4,2,1}`
(MHA → GQA → MQA) × 2 seeds = **8 jobs**, each solved to **matched ~60M params** (the solver adjusts
`n_embd`), on a LLaMA-style baseline (RoPE / RMSNorm / SwiGLU).

Each job writes to `runs/pilot_gqa/<axis>=<value>__seed<n>/`:

| file | what |
| --- | --- |
| `metrics.jsonl` | self-describing `architecture` row + one `step` row per logged step |
| `config.yaml` | the exact resolved config (use it for on-device benchmarking) |
| `ckpt` | trained checkpoint |
| `run.json` / `done` | run metadata / completion marker |

The sweep is **resumable**: re-running skips jobs that have a `done` marker. Add `--force` to redo
them.

## 4. Aggregate → table + Pareto plots

```bash
python scripts/aggregate.py --dir runs/pilot_gqa --csv runs/pilot_gqa.csv --plot
```

Prints a Markdown table and writes a CSV with: params, FLOPs/token, **KV-cache bytes/token**, final
loss, perplexity, tok/s, peak memory, and MFU. With `--plot` it also writes
`runs/pilot_gqa/pareto_*.png` (loss vs params / FLOPs / KV-cache).

## 5. Measure on-device (run ON each Jetson / the laptop)

Copy a run's `config.yaml` + `ckpt` to the device, then:

```bash
python scripts/bench_device.py \
  --config runs/pilot_gqa/num_key_value_heads=2__seed0/config.yaml \
  --checkpoint runs/pilot_gqa/num_key_value_heads=2__seed0/ckpt \
  --device-name jetson-nx --precision fp16
```

Writes `results/<device>_<precision>.jsonl` with time-to-first-token, decode tok/s, peak RAM, and
**energy/token** (via `tegrastats` on Jetson, `nvidia-smi` on a desktop GPU). Repeat per device ×
variant. Use `--no-cache` to also measure the full-forward decode path (the cached-vs-uncached gap
is itself a result).

## Tuning knobs (edit the spec's `overrides:` / `axes:`)

- **Token budget / runtime.** `tokens/step = batch_size × seq_len × n_batches`. The pilot's
  `16 × 512 × 4 = 32,768`, so `max_iter: 3000` ≈ **98M tokens/run** (a quick first signal). For a
  publishable result raise it — e.g. `max_iter: 15000` ≈ ~0.5B tokens/run. Runtime ≈ `tokens ÷ tok/s`.
- **OOM.** Lower `training.batch_size` and/or `training.n_batches`.
- **MFU.** Set `logging.device_peak_flops` to your GPU's peak FLOP/s.
- **Other axes.** Change `axes:` to sweep a different knob, holding the rest at the baseline:
  - `model.norm: [rmsnorm, layernorm]`
  - `model.ffn: [swiglu, gelu, geglu]`
  - `model.position: [rope, learned, none]`
  - depth-vs-width: set `target_params` and sweep `model.n_blocks` with `vary: n_embd` so total
    params stay matched.
- **Scales.** Re-run with a larger `target_params` (e.g. 150M, 350M) to check whether rankings hold
  across sizes.

## Reproducibility notes

- Every run is seeded (`training.seed`), uses the same data shards + tokenizer, and records its exact
  config and a self-describing metrics file — so the whole study is reproducible from the specs.
- Keep `seeds: [0, 1, 2]` (≥3) for the final study and report mean ± CI per configuration.
- Generate an architecture card for any run with
  `python scripts/describe_arch.py --config runs/pilot_gqa/<run>/config.yaml`.
