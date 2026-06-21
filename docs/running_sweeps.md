# Running a controlled architecture sweep

This is the **study guide** for LLM Toaster's architecture experiments — it covers both the *why*
(the research question, why each dimension is varied, and the hypothesis behind it) and the *how*
(designing, running, aggregating, and benchmarking a sweep). If you only want the commands, jump to
[§6 Execution](#6-execution-step-by-step); but the value of the framework is in the methodology, so
read §1–§4 first.

> See [`docs/architecture.md`](architecture.md) for what each architectural term means and its
> implementation. This document is about how to *study* those choices.

**Contents**
1. [The research question & main goal](#1-the-research-question--main-goal)
2. [Core principle: one axis at a time, matched parameter budget](#2-core-principle-one-axis-at-a-time-matched-parameter-budget)
3. [What we measure and why](#3-what-we-measure-and-why)
4. [The axes: what we change, why, and the hypothesis](#4-the-axes-what-we-change-why-and-the-hypothesis)
5. [Experiment stages: sanity → pilot → main](#5-experiment-stages-sanity--pilot--main)
6. [Execution, step by step](#6-execution-step-by-step)
7. [Designing your own sweep spec](#7-designing-your-own-sweep-spec)
8. [Reading the results & drawing conclusions](#8-reading-the-results--drawing-conclusions)
9. [Reproducibility & honest caveats](#9-reproducibility--honest-caveats)
10. [Tuning knobs](#10-tuning-knobs)

---

## 1. The research question & main goal

> **Goal:** *Systematically compare architecture choices for small decoder-only language models under
> matched parameter budgets, and map the resulting quality ↔ memory ↔ latency ↔ energy Pareto
> trade-offs on real hardware.*

Small LMs are increasingly deployed on constrained hardware (laptops, Jetson-class edge devices,
phones). At that scale the bottleneck is rarely raw accuracy — it is the **inference footprint**:
how much memory the KV-cache eats at long context, how many tokens/second you can decode, and how many
joules each token costs. Papers usually report a single "best" architecture at one size; this study
instead asks a deployment-shaped question:

- **Which architectural choice buys the most quality per unit of memory / latency / energy?**
- **Do those rankings hold under a fixed parameter budget**, or are apparent wins just "more params"?
- **Do they hold across model sizes and across real devices**, or only on the training GPU?

The deliverable is not one model. It is a set of **Pareto frontiers** — e.g. *perplexity vs decode
tokens/sec*, *perplexity vs peak memory*, *perplexity vs energy/token*, *KV-cache bytes/token vs
decode tokens/sec* — from which a practitioner can pick the right architecture for a given device
budget.

---

## 2. Core principle: one axis at a time, matched parameter budget

Two methodological rules make a result *causal* rather than coincidental.

**(a) One variable at a time.** Every run changes exactly one architectural axis; all other settings
stay at the baseline. So a difference in the outcome is attributable to that one change, not a tangle
of confounded changes.

**(b) Matched total parameter budget.** Bigger models are usually better — that is uninteresting. To
isolate the *architecture*, we hold the **total parameter count fixed** and let the architectures
differ in *how* they spend that budget. The matched-parameter solver (`toaster/models/sizing.py`,
`estimate_params` + `solve_for_target_params`) computes the exact parameter count analytically and
adjusts one free dimension — width (`n_embd`) or depth (`n_blocks`) — so every run lands at the
`target_params` you set.

**What "matched params" controls — and what it deliberately does not.** This is the most important
nuance to report honestly:

- It **controls** model *capacity* (parameter count), removing the trivial "more params → better"
  confound.
- It **does not** equalise FLOPs/token or KV-cache bytes/token — and that is the point. Example: GQA
  has fewer key/value parameters than MHA, so to hit the same total budget the solver makes the GQA
  run slightly **wider** (`n_embd` grows). The comparison is therefore *"same parameter budget, spent
  differently"* — exactly the deployment question. When you report a GQA win, state that it reinvested
  the saved KV parameters into width.
- Held fixed across the whole study (so they cannot confound): **dataset + shard order, tokenizer,
  context length (`seq_len`), optimizer + schedule, token budget, and precision.**

---

## 3. What we measure and why

Every run self-describes in `metrics.jsonl` (an `architecture` row + per-step `step` rows); the
aggregator collates them, and the on-device benchmark adds deployment numbers. Each metric maps to one
axis of the Pareto trade-off:

| Metric | Pareto axis | Where it comes from | Why it matters |
| --- | --- | --- | --- |
| **Validation loss / perplexity** | Quality | training `metrics.jsonl` (val) | The thing we're trading everything else against. |
| **Total params** | Capacity (held fixed) | architecture row | The matched budget; confirms the solver hit `target_params`. |
| **FLOPs/token** | Compute | architecture row | Architecture-independent cost proxy (`~6N + attention term`). |
| **KV-cache bytes/token** | Memory (inference) | architecture row | `2·n_blocks·kv_heads·head_dim·2 B`; the dominant runtime memory at long context — set by MHA/GQA/MQA. |
| **Peak memory (alloc / reserved)** | Memory (training) | step rows | Real VRAM pressure during training. |
| **Train tokens/sec** | Throughput | step rows | How fast the architecture trains on your GPU. |
| **MFU** | HW efficiency | step rows (needs `device_peak_flops`) | How well it uses the device. |
| **TTFT** | Latency (prefill) | `bench_device.py` | Interactivity — time to the first generated token. |
| **Decode tokens/sec** | Latency (generation) | `bench_device.py` | Deployment speed; improved by a smaller KV-cache. |
| **Energy/token** | Energy | `bench_device.py` (`tegrastats`/`nvidia-smi`) | Joules per token — the edge-device efficiency axis. |

Quality comes from the **training GPU**; latency/memory/energy come from running the trained
checkpoint **on each target device** (the laptop GPU and each Jetson). Decode uses the realistic
KV-cached path (`generate_cached`).

---

## 4. The axes: what we change, why, and the hypothesis

Each axis below is a separate sweep (change `axes:` in the spec). Hypotheses are *testable
expectations*, not assumed truths — the study exists to confirm or refute them.

### 4.1 Attention sharing — MHA → GQA → MQA (`model.num_key_value_heads`)

- **What we vary:** `num_key_value_heads ∈ {n_head, …, 2, 1}` (MHA → GQA → MQA). Query heads always
  number `n_head`; KV heads are shared.
- **Why:** the KV-cache is usually the dominant inference-memory and memory-bandwidth cost on-device;
  fewer KV heads shrink it linearly. This is the **headline axis** of the study.
- **Hypothesis:** as `kv_heads` decreases, **KV-cache bytes/token ↓, decode tok/s ↑, peak memory ↓,
  energy/token ↓**, while **perplexity stays roughly flat for GQA and degrades only at MQA**. At
  matched params the saved KV parameters are reinvested into width, which may even *help* quality.
- **Reveals it:** *KV-cache bytes/token vs decode tok/s* (the headline Pareto plot) and *perplexity vs
  `kv_heads`*. The cached-vs-uncached decode gap (`--no-cache`) quantifies the cache's real payoff.

### 4.2 Normalization — LayerNorm vs RMSNorm (`model.norm`)

- **What we vary:** `model.norm ∈ {layernorm, rmsnorm}`.
- **Why:** RMSNorm drops the mean-subtraction and the bias term (`E` params vs `2E`), so it is cheaper
  per token and the modern (LLaMA) default; the question is whether that simplification costs quality.
- **Hypothesis:** **RMSNorm ≈ LayerNorm in perplexity**, with marginally higher tokens/sec and a tiny
  parameter saving (which, at matched budget, the solver returns as width). I.e. RMSNorm is a "free"
  simplification.
- **Reveals it:** *perplexity* (should overlap within seed noise) and *tokens/sec*.

### 4.3 Feed-forward — GELU vs GEGLU vs SwiGLU (`model.ffn`)

- **What we vary:** `model.ffn ∈ {gelu, geglu, swiglu}`.
- **Why:** the FFN is ~half the parameters of a small model. Gated FFNs (GEGLU/SwiGLU) add a second
  input projection — they cost more params at equal `ffn_mult`, so the comparison *must* be at matched
  total params (the solver narrows `n_embd` for the gated variants).
- **Hypothesis:** **gated FFNs (SwiGLU ≈ GEGLU) beat plain GELU at matched params** — better quality
  per parameter — at the cost of an extra matmul (higher FLOPs/token), so the latency/quality trade
  is the interesting part.
- **Reveals it:** *perplexity at matched params* and *perplexity vs FLOPs/token*.

### 4.4 Positional encoding — learned vs RoPE vs NoPE (`model.position`)

- **What we vary:** `model.position ∈ {learned, rope, none}`.
- **Why:** learned absolute positions cost `seq_len·E` params and cap the context length; RoPE adds no
  parameters and extrapolates; NoPE (no positional signal) tests whether the causal mask alone leaks
  enough order for a decoder-only model.
- **Hypothesis:** **RoPE ≥ learned** in quality (and strictly better for length generalization),
  freeing the position-table parameters for width; **NoPE is worse but surprisingly viable**.
- **Reveals it:** *perplexity*; for the length-generalization claim, evaluate at sequence lengths
  beyond the training `seq_len` (RoPE should degrade more gracefully than learned positions).

### 4.5 Depth vs width (`model.n_blocks` with `vary: n_embd`, matched params)

- **What we vary:** `n_blocks` (number of layers), letting the solver adjust `n_embd` so total params
  stay matched — i.e. the **shape** of a fixed budget.
- **Why:** depth adds *serial* computation (each layer waits on the previous → higher decode latency,
  lower MFU); width adds *parallel* computation (better GPU utilisation) but more activation/KV
  memory. The best shape is budget- and device-dependent.
- **Hypothesis:** quality improves with depth up to a **sweet spot** then plateaus/regresses, while
  **deeper-narrower models are slower to decode** (more sequential layers) and **wider-shallower
  models are faster but heavier in memory**. The Pareto frontier picks the shape per device.
- **Reveals it:** *perplexity vs depth/width ratio* and *depth vs decode tok/s / MFU*.

### 4.6 Scale check (optional but important)

Re-run the winning axes at a larger `target_params` (e.g. 60M → 150M → 350M). **Hypothesis:** rankings
are scale-stable. If a choice that wins at 60M loses at 350M, that itself is a key finding and changes
the recommendation.

---

## 5. Experiment stages: sanity → pilot → main

Run the study in three escalating stages so you debug cheaply and spend GPU-hours only on the
comparisons that matter.

| Stage | Size / budget | Purpose | Spec |
| --- | --- | --- | --- |
| **Sanity** | tiny, CPU, ~seconds | Prove the train → metrics → aggregate loop works; never for the paper. | `config/sweeps/smoke.yaml` |
| **Pilot** | ~60M, short budget, 1–2 seeds | Quickly compare *all* variants on every axis; pick the strongest. | `config/sweeps/pilot_gqa.yaml`, `config/sweeps/example.yaml` |
| **Main** | larger (e.g. 150–350M), full budget, **3 seeds** | Final, credible results for the **strongest** comparisons only. | a spec you derive (raise `target_params`, `max_iter`, `seeds: [0,1,2]`) |

Do **not** run every variant at the main scale — use pilot results to choose. Reserve 3 seeds (mean ±
CI) for headline claims.

---

## 6. Execution, step by step

Tooling: `scripts/sweep.py`, `scripts/aggregate.py`, `scripts/bench_device.py`,
`scripts/describe_arch.py`, and the specs in `config/sweeps/`.

### 6.0 Setup

```bash
pip install -e ".[viz]"        # matplotlib, for the Pareto plots (optional)
```

### 6.1 Sanity-check the pipeline (CPU, ~10 s)

```bash
python scripts/sweep.py --spec config/sweeps/smoke.yaml
python scripts/aggregate.py --dir runs/smoke_sweep
```

A 2-row table confirms the train → metrics → aggregate loop works end to end.

### 6.2 Free the GPU

Stop any other training — a sweep needs the full GPU; concurrent runs compete for VRAM and will likely
OOM.

### 6.3 Run the sweep

```bash
tmux new -s sweep        # survives terminal/SSH disconnects
systemd-inhibit --what=sleep:idle:handle-lid-switch --mode=block --why="arch sweep" \
  python scripts/sweep.py --spec config/sweeps/pilot_gqa.yaml
```

`config/sweeps/pilot_gqa.yaml` runs the headline **KV-cache axis** — `num_key_value_heads ∈ {8,4,2,1}`
(MHA → GQA → MQA) × 2 seeds = **8 jobs**, each solved to **matched ~60M params** (the solver adjusts
`n_embd`), on a LLaMA-style baseline (RoPE / RMSNorm / SwiGLU). Each job writes
`runs/pilot_gqa/<axis>=<value>__seed<n>/`:

| file | what |
| --- | --- |
| `metrics.jsonl` | self-describing `architecture` row + one `step` row per logged step |
| `config.yaml` | the exact resolved config (use it for on-device benchmarking) |
| `ckpt` | trained checkpoint |
| `run.json` / `done` | run metadata / completion marker |

**Robustness:** the sweep is **resumable** — re-running skips jobs with a `done` marker (`--force` to
redo). Each non-CPU job trains in a **spawned subprocess**, so a hard GPU fault fails only that cell
(leaving a `failed` marker) and the sweep continues.

### 6.4 Aggregate → table + Pareto plots

```bash
python scripts/aggregate.py --dir runs/pilot_gqa --csv runs/pilot_gqa.csv --plot
```

Prints a Markdown table and writes a CSV with: params, FLOPs/token, **KV-cache bytes/token**, final
loss, perplexity, tok/s, peak memory, and MFU. With `--plot` it writes `runs/pilot_gqa/pareto_*.png`
(loss vs params / FLOPs / KV-cache).

### 6.5 Measure on-device (run ON each Jetson / the laptop)

Copy a run's `config.yaml` + `ckpt` to the device, then:

```bash
python scripts/bench_device.py \
  --config runs/pilot_gqa/num_key_value_heads=2__seed0/config.yaml \
  --checkpoint runs/pilot_gqa/num_key_value_heads=2__seed0/ckpt \
  --device-name jetson-nx --precision fp16
```

Writes `results/<device>_<precision>.jsonl` with time-to-first-token, decode tok/s, peak RAM, and
**energy/token** (`tegrastats` on Jetson, `nvidia-smi` on a desktop GPU). Repeat per device × variant.
`--no-cache` also measures the full-forward decode path (the cached-vs-uncached gap is itself a
result).

---

## 7. Designing your own sweep spec

A spec is a one-variable-at-a-time study from a base config. Annotated (`config/sweeps/example.yaml`):

```yaml
base_config: config/default_config.yaml   # the starting point; everything not overridden comes from here
output_dir: runs/ffn_ablation             # one subdir per (axis=value, seed)
seeds: [0, 1, 2]                           # ≥3 for final claims (mean ± CI)
target_params: 60_000_000                  # the matched budget
vary: n_embd                               # which dim the solver adjusts to hit target_params (n_embd | n_blocks)

overrides:                                 # the FIXED baseline applied to EVERY run (the controls)
  training.max_iter: 2000                  # token budget = max_iter × (batch_size × seq_len × n_batches)
  model.position: rope
  model.norm: rmsnorm
  model.num_key_value_heads: 2

axes:                                      # the ONE thing that varies; each value is a separate run
  model.ffn: [gelu, geglu, swiglu]
```

Rules of thumb:

- **Put the controls in `overrides:`** and the single studied dimension in `axes:`. Keep dataset,
  tokenizer, `seq_len`, optimizer/schedule, token budget, and precision identical across the spec.
- **For depth-vs-width**, set `vary: n_embd` and put `model.n_blocks: [...]` in `axes:` so total params
  stay matched while the shape changes.
- **Each `(axis, value, seed)` is one run dir.** Specs never overwrite completed runs (the `done`
  marker) unless you pass `--force`.

---

## 8. Reading the results & drawing conclusions

- **The table** ranks variants on every metric at once; sort by perplexity, then check the cost
  columns (KV-cache, FLOPs, memory).
- **The Pareto frontier** is the set of variants that are not beaten on *both* axes simultaneously
  (e.g. nothing has lower perplexity *and* fewer KV bytes/token). Recommend variants *on* the
  frontier; a variant strictly inside it is dominated.
- **Per-axis "win" looks like:** GQA/MQA → big drop in KV-cache bytes/token + decode tok/s gain for a
  small perplexity cost; gated FFN → lower perplexity at matched params; RMSNorm → perplexity parity
  with a small speed/param saving; RoPE → parity-or-better quality with graceful length extrapolation.
- **Use seeds.** Report **mean ± CI** across `seeds: [0,1,2]`; a gap smaller than the seed spread is
  not a real difference.
- **Confounds to keep checking:** confirm `params_total` is actually matched across the axis; remember
  matched-params means budget is *reallocated*, so attribute wins to "spent on width/depth," not magic.

---

## 9. Reproducibility & honest caveats

- **Determinism:** every run is seeded (`training.seed`, applied to Python/NumPy/torch+CUDA), uses the
  same data shards in the same order, the same tokenizer, and records its exact resolved config + a
  self-describing metrics file — so the study reproduces from the specs. (Strict *bitwise* determinism
  across hardware is a roadmap item; see `docs/architecture.md` §15.)
- **What matched-params does not equalise:** FLOPs/token and KV-cache bytes/token *vary by design* —
  report them, don't hide them.
- **Token budget ≠ convergence:** pilot budgets (~0.1B tokens) give a *signal*, not converged models.
  Raise `max_iter` for main-stage claims and verify rankings are stable as the budget grows.
- **Single-process, single-GPU:** training is single-GPU (DDP/FSDP not wired). The 250M+ main stage is
  slow on one 8–12 GB GPU — budget wall-clock with gradient accumulation, or scale the budget down.
- **Energy is device-dependent:** measured via `tegrastats` (Jetson) / `nvidia-smi` (desktop); absent
  tooling, energy fields are null. Report idle-power and measurement conditions with any energy claim.
- **Generate a card for any run** for the exact architecture + parameter table:
  `python scripts/describe_arch.py --config runs/<sweep>/<run>/config.yaml`.

---

## 10. Tuning knobs

- **Token budget / runtime.** `tokens/step = batch_size × seq_len × n_batches`. The pilot's
  `16 × 512 × 4 = 32,768`, so `max_iter: 3000` ≈ **98M tokens/run** (a quick first signal). For a
  publishable result raise it — e.g. `max_iter: 15000` ≈ ~0.5B tokens/run. Runtime ≈ `tokens ÷ tok/s`.
- **OOM.** Lower `training.batch_size` and/or `training.n_batches` (gradient accumulation keeps the
  effective batch).
- **MFU.** Set `logging.device_peak_flops` to your GPU's peak FLOP/s (e.g. ~3.0e13 for a 4070-class
  card).
- **Scales.** Re-run with a larger `target_params` (e.g. 150M, 350M) to check whether rankings hold
  across sizes.
- **Other axes.** Swap `axes:` to study a different knob, holding the rest at the baseline:
  - `model.norm: [rmsnorm, layernorm]`
  - `model.ffn: [swiglu, gelu, geglu]`
  - `model.position: [rope, learned, none]`
  - depth-vs-width: `vary: n_embd` + `model.n_blocks: [...]` in `axes:`.
