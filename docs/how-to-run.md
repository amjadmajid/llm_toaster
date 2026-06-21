# How to run LLM Toaster (Colab · Lambda · AWS · Leonardo/EuroHPC · local GPU)

This guide shows how to install and run LLM Toaster — training, sweeps, inference, and on-device
benchmarking — across environments. The **primary target is NVIDIA/CUDA**, with notes for **AMD/ROCm**
and **Apple Silicon/macOS (MPS)**.

> See [`docs/architecture.md`](architecture.md) for the system design and
> [`docs/running_sweeps.md`](running_sweeps.md) for the study methodology. This file is purely
> operational: *where* to run and *how* to set it up.

**Contents**
- [0. Universal setup (read first)](#0-universal-setup-read-first)
- [1. Local computer with a GPU](#1-local-computer-with-a-gpu) — NVIDIA · AMD/ROCm · macOS/MPS
- [2. Google Colab](#2-google-colab)
- [3. Lambda Cloud](#3-lambda-cloud)
- [4. AWS](#4-aws)
- [5. Leonardo / EuroHPC (CINECA, SLURM)](#5-leonardo--eurohpc-cineca-slurm)
- [6. Inference & on-device benchmarking](#6-inference--on-device-benchmarking)
- [7. Troubleshooting](#7-troubleshooting)
- [Appendix: config override snippets](#appendix-config-override-snippets)

---

## 0. Universal setup (read first)

The same project runs everywhere; only the **torch wheel**, **device**, and **precision** differ.

### 0.1 Install

```bash
git clone https://github.com/amjadmajid/llm_toaster && cd llm_toaster
python -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip

# 1) Install torch for YOUR accelerator FIRST (so the editable install reuses it) — see the per-platform sections:
#    NVIDIA: pip install torch --index-url https://download.pytorch.org/whl/cu124
#    AMD:    pip install torch --index-url https://download.pytorch.org/whl/rocm6.2
#    macOS:  pip install torch            # default wheel includes Metal/MPS

# 2) Then install the project + the extras you need:
pip install -e ".[data,viz]"      # data = HF download/tokenize; viz = Pareto plots
#   ".[hf]"  -> Hugging Face tokenizer backend     ".[dev]" -> pytest/ruff/mypy
```

**Why torch first:** `pip install -e .` only requires `torch>=2.1`; installing the correct accelerator
wheel beforehand prevents pip from pulling a wheel that doesn't match your hardware.

Verify the accelerator is visible:

```bash
python -c "import torch; print('cuda', torch.cuda.is_available(), '| mps', torch.backends.mps.is_available())"
```

### 0.2 Validate the install (CPU, seconds — works with no GPU)

```bash
python trainer.py --config config/smoke_test_config.yaml --mode pretrain
pytest -q     # if you installed ".[dev]"
```

### 0.3 Get training data

Real pretraining reads tokenized `.npy` shards. Build them once (needs internet + `".[data]"`):

```bash
python dataspace/src/download_tokenize_hf.py     # tokenizes fineweb-edu into dataspace/fineweb/*.npy
```

For offline/airgapped nodes (HPC), do this on a node *with* internet and copy the shards over (see §5).
Tiny `.txt`/`.tokens` fixtures under `tests/fixtures/tokenized_data/` let you exercise the full
pipeline with no download.

### 0.4 Pick device & precision

The engine auto-selects the device (`cuda` → `mps` → `cpu`); override with `training.device`. Precision
is `distributed.mixed_precision` (autocast only activates on CUDA). Choose by GPU generation:

| Hardware | `mixed_precision` | Note |
| --- | --- | --- |
| NVIDIA Ampere+ (A100, A10/A10G, L4, 30xx/40xx, H100) | `bf16` | Default; no loss scaling needed. |
| NVIDIA Turing/Volta (T4, V100, 20xx) | `fp16` | No bf16 tensor cores; fp16 uses a GradScaler automatically. |
| AMD (MI200/MI300, RDNA) via ROCm | `bf16` (MI2xx+) or `fp16` | torch reports the device as `cuda`; see §1b. |
| Apple Silicon (MPS) | `no` | Autocast is CUDA-only here → runs fp32. See §1c. |
| CPU only | `no` | Smoke/tests only. |

See the [Appendix](#appendix-config-override-snippets) for the exact YAML to drop in.

### 0.5 Crash-safe long runs (matters for cloud/spot/HPC)

LLM Toaster is built to survive interruptions — essential on preemptible cloud instances and
time-limited HPC jobs:

- **Atomic checkpoints** (temp → `fsync` → rename): an interrupted save never corrupts the live file.
- **Graceful SIGINT/SIGTERM**: on interrupt (incl. spot reclamation / SLURM `--time` hit) the engine
  writes `emergency.pt` at the next step boundary.
- **Exact resume**: `python trainer.py -ct` restores model/optimizer/scheduler/RNG/data-cursor/step and
  continues. For sweeps, re-running skips `done` cells.

Put `checkpoints/`, `logs/`, and `runs/` on **persistent storage** (Drive / EBS / `$WORK`) so resume
works after a restart.

> **One run = one GPU.** Distributed training (DDP/FSDP) is intentionally not wired (`validate()`
> rejects `distributed.backend != none`). To use *multiple* GPUs, run **independent jobs** — sweep
> cells pinned per GPU, or SLURM array tasks (§5) — not data-parallel on a single run.

---

## 1. Local computer with a GPU

### 1a. NVIDIA / CUDA (primary target)

```bash
# match the CUDA build to your driver (cu121 / cu124 / cu128 ...); see https://pytorch.org/get-started/locally/
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[data,viz]"
python -c "import torch; print(torch.cuda.get_device_name(0))"

python dataspace/src/download_tokenize_hf.py          # one-time data prep
python trainer.py --config config/default_config.yaml --mode pretrain
python trainer.py -ct                                 # resume after any stop
```

Tune `training.batch_size` / `training.n_batches` to your VRAM (effective tokens/step =
`batch_size × seq_len × n_batches`; lower the first two on OOM, raise `n_batches` to keep the budget).
Set `logging.device_peak_flops` to your card's peak FLOP/s to get MFU.

### 1b. AMD / ROCm (Linux)

```bash
pip install torch --index-url https://download.pytorch.org/whl/rocm6.2   # match your ROCm version
pip install -e ".[data,viz]"
python -c "import torch; print('hip', torch.version.hip, '| cuda?', torch.cuda.is_available())"
```

On ROCm, torch presents AMD GPUs through the **`cuda`** device API, so the framework's auto-detection
and `training.device: cuda` work unchanged. Notes:

- Precision: `bf16` on MI200/MI300; `fp16` on older parts. If SDPA misbehaves, set
  `attention.backend: eager` (Appendix).
- Select a GPU with `HIP_VISIBLE_DEVICES=0` (the ROCm analogue of `CUDA_VISIBLE_DEVICES`).
- Energy/peak-memory: the benchmark's samplers are NVIDIA/Jetson-only — run `bench_device.py` with
  `--energy none` (decode tok/s and TTFT still work; `rocm-smi`-based energy is a roadmap item).

### 1c. macOS / Apple Silicon (MPS)

Great for development and inference; fine for small-model training.

```bash
pip install torch            # default macOS wheel ships Metal/MPS support
pip install -e ".[data,viz]"
export PYTORCH_ENABLE_MPS_FALLBACK=1     # let unsupported ops fall back to CPU instead of erroring
python -c "import torch; print('mps', torch.backends.mps.is_available())"
```

In your config set `training.device: mps` and `distributed.mixed_precision: "no"` (autocast is
CUDA-only; MPS runs fp32). `peak_mem_bytes`, MFU, and energy are CUDA/Jetson-only and will read 0/None
on MPS — quality (loss/perplexity) and decode tok/s are still valid. Keep `distributed.compile: false`.

---

## 2. Google Colab

Colab gives a free T4 (A100/L4 on Pro). The catch is **ephemeral disk + session limits**, so persist
to Google Drive and lean on resume.

```python
# Cell 1 — GPU + Drive
from google.colab import drive; drive.mount('/content/drive')
!nvidia-smi -L
%cd /content
!git clone https://github.com/amjadmajid/llm_toaster && cd llm_toaster
%cd /content/llm_toaster

# Cell 2 — install (Colab already has a CUDA torch; DON'T reinstall torch)
!pip install -e ".[data,viz]" -q
import torch; print(torch.cuda.get_device_name(0))

# Cell 3 — point all outputs at Drive so they survive disconnects, then train.
#          On a free T4 use fp16 (no bf16 tensor cores). See the Appendix for the YAML.
!python trainer.py --config config/smoke_test_config.yaml --mode pretrain    # quick check
# ... edit a config to set checkpoint/log paths under /content/drive/MyDrive/llm_toaster_runs ...
!python trainer.py --config /content/drive/MyDrive/llm_toaster_runs/my_config.yaml --mode pretrain

# Cell 4 — after a disconnect, reconnect, re-run Cells 1–2, then resume:
!python trainer.py --config /content/drive/MyDrive/llm_toaster_runs/my_config.yaml -ct
```

Tips: **Runtime → Change runtime type → GPU**; set `checkpointing.output_dir`, `training.ckpt`,
`training.ckpt_config`, and `logging.*` to a Drive path; choose a token budget you can finish in one
session (`training.max_iter`) and rely on `-ct` to continue across sessions. Free T4 → `fp16`.

---

## 3. Lambda Cloud

Lambda on-demand instances (A10, A100, H100, GH200) ship with NVIDIA drivers and often "Lambda Stack"
(a system torch). Attach **persistent storage** so checkpoints outlive the instance.

```bash
ssh ubuntu@<instance-ip>
git clone https://github.com/amjadmajid/llm_toaster && cd llm_toaster
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu124   # or reuse Lambda Stack's torch
pip install -e ".[data,viz]"
python dataspace/src/download_tokenize_hf.py        # or rsync pre-tokenized shards to the persistent volume

tmux new -s train       # survives SSH drops
python trainer.py --config config/default_config.yaml --mode pretrain
# detach: Ctrl-b d ; reattach: tmux attach -t train
```

**Multi-GPU instances** (e.g. 8×A100): run one *independent* job per GPU rather than DDP. For a sweep,
launch several sweep processes, each pinned to a GPU and given a disjoint slice of the axis:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/sweep.py --spec config/sweeps/spec_a.yaml &
CUDA_VISIBLE_DEVICES=1 python scripts/sweep.py --spec config/sweeps/spec_b.yaml &
wait
```

Point `output_dir` at the persistent volume and **terminate the instance when done** (storage keeps the
checkpoints; resume later with `-ct`).

---

## 4. AWS

### EC2 (recommended)

Use a GPU instance — `g4dn` (T4), `g5` (A10G), `p4d/p4de` (A100), `p5` (H100) — with the **Deep Learning
AMI** (preinstalled CUDA + drivers). Put work on an **EBS** volume; sync data/checkpoints to **S3**.

```bash
ssh ubuntu@<ec2-ip>
git clone https://github.com/amjadmajid/llm_toaster && cd llm_toaster
python -m venv .venv && source .venv/bin/activate && pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[data,viz]"

aws s3 sync s3://my-bucket/fineweb dataspace/fineweb     # stage pre-tokenized shards (or run the tokenizer)
tmux new -s train
python trainer.py --config config/default_config.yaml --mode pretrain
```

Sync checkpoints to S3 periodically (and after the run) so they survive instance teardown:

```bash
aws s3 sync checkpoints/ s3://my-bucket/checkpoints/      # and runs/, logs/
```

### Spot instances — a perfect fit for the crash-safe design

Spot is far cheaper but interruptible (≈2-minute SIGTERM warning). LLM Toaster handles this directly:
the SIGTERM triggers a graceful `emergency.pt` (atomic), and on a fresh spot instance you resume with
`-ct` after re-syncing from S3. Sync checkpoints to S3 on a timer/`checkpointing.save_every_steps` so a
reclaim loses at most a few steps.

### SageMaker (optional)

Run as a **custom training job**: package this repo in a container whose entrypoint is
`python trainer.py --config <cfg> --mode pretrain`, mount an FSx/S3 channel for `dataspace/`, and write
checkpoints to `/opt/ml/checkpoints` (SageMaker syncs it to S3, enabling managed-spot resume).

---

## 5. Leonardo / EuroHPC (CINECA, SLURM)

Leonardo's **Booster** partition has nodes of 4× NVIDIA A100 (64 GB). It is a SLURM cluster:
account-based, batch-scheduled, and compute nodes are **offline** — so you stage data/tokenizer caches
on a login node first and keep caches on `/leonardo_work` (HOME has a tight quota). The patterns below
mirror [slimx-ai/EuroHPC-Qwen-Tool-Calling](https://github.com/slimx-ai/EuroHPC-Qwen-Tool-Calling),
adapted to LLM Toaster's single-process design.

### 5.1 One-time setup on a login node (has internet)

```bash
export PROJECT=AIFAC_P01_006                      # <-- your CINECA project account
export WORK=/leonardo_work/$PROJECT/$USER
mkdir -p "$WORK" && cd "$WORK"

git clone https://github.com/amjadmajid/llm_toaster && cd llm_toaster
module load python/3.11        # or the site's recommended python; module list is site-specific
python -m venv "$WORK/.venv" && source "$WORK/.venv/bin/activate" && pip install -U pip
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[data,viz]"

# Stage data + tokenizer caches onto $WORK so OFFLINE compute nodes can use them:
export TIKTOKEN_CACHE_DIR="$WORK/.cache/tiktoken"; mkdir -p "$TIKTOKEN_CACHE_DIR"
python -c "import tiktoken; tiktoken.get_encoding('gpt2').encode('warm cache')"   # caches the real GPT-2 BPE
python dataspace/src/download_tokenize_hf.py     # writes dataspace/fineweb/*.npy (move to $SCRATCH for speed if desired)
```

> Without the warmed `TIKTOKEN_CACHE_DIR`, offline nodes silently fall back to a byte-level tokenizer —
> fine for tests, **wrong for real training**. Always warm it on the login node.

### 5.2 Single-GPU training job (`train.sbatch`)

```bash
#!/bin/bash
#SBATCH --job-name=toaster-pretrain
#SBATCH -A AIFAC_P01_006
#SBATCH -p boost_usr_prod
#SBATCH --qos=boost_qos_dbg          # debug QoS for short jobs; drop for long runs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=toaster-%j.out
#SBATCH --error=toaster-%j.err
set -euo pipefail

export PROJECT=AIFAC_P01_006
export WORK=/leonardo_work/$PROJECT/$USER
export TIKTOKEN_CACHE_DIR="$WORK/.cache/tiktoken"     # use the warmed cache; nodes are offline
cd "$WORK/llm_toaster"
source "$WORK/.venv/bin/activate"

# Resume if a checkpoint exists, else start fresh (lets you re-queue past the time limit):
if [ -f checkpoints/base_ckpt ]; then RESUME="-ct"; else RESUME=""; fi
srun python trainer.py --config config/default_config.yaml --mode pretrain $RESUME
```

Submit with `sbatch train.sbatch`. Because checkpoints are atomic and `-ct` resumes exactly, a job that
hits `--time` simply **re-queues and continues** — set `checkpointing.save_every_steps` (or, on the
roadmap, `checkpoint_interval_minutes`) comfortably under your wall-clock limit. A100 64 GB → keep
`bf16`.

### 5.3 Parallel sweeps via a SLURM **job array** (the EuroHPC-native way)

LLM Toaster runs one GPU per process, so parallelise a study across Leonardo's GPUs with a job array —
one array task per variant, each on its own GPU. Create one single-value spec per variant that share an
`output_dir` (cells write disjoint subdirs; `done` markers make it safe and resumable), then:

```bash
#!/bin/bash
#SBATCH --job-name=toaster-sweep
#SBATCH -A AIFAC_P01_006
#SBATCH -p boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --array=0-3                    # 4 variants -> 4 GPUs in parallel
#SBATCH --output=sweep-%A_%a.out
set -euo pipefail
export PROJECT=AIFAC_P01_006; export WORK=/leonardo_work/$PROJECT/$USER
export TIKTOKEN_CACHE_DIR="$WORK/.cache/tiktoken"
cd "$WORK/llm_toaster"; source "$WORK/.venv/bin/activate"

SPECS=(config/sweeps/mha.yaml config/sweeps/gqa4.yaml config/sweeps/gqa2.yaml config/sweeps/mqa.yaml)
srun python scripts/sweep.py --spec "${SPECS[$SLURM_ARRAY_TASK_ID]}"
```

After all tasks finish, aggregate on a login node:
`python scripts/aggregate.py --dir runs/pilot_gqa --csv runs/pilot_gqa.csv --plot`.

> For a full multi-GPU node in **one** job, you can instead launch four background sweep processes pinned
> with `CUDA_VISIBLE_DEVICES=0..3` (as in §3). Job arrays are usually cleaner on a shared scheduler.

---

## 6. Inference & on-device benchmarking

Inference runs anywhere the model is built (CUDA/ROCm/MPS/CPU):

```bash
python extract_inference_model.py --config checkpoints/base_config.yaml \
  --output model/babyGPT/babyGPT_base.llm --output-config model/babyGPT/babyGPT_base.yaml
python inference.py -p "Your prompt" --config model/babyGPT/babyGPT_base.yaml \
  --model model/babyGPT/babyGPT_base.llm
```

Deployment **benchmarks** (TTFT, decode tok/s, peak memory, energy/token) must run **on the target
device**:

```bash
# Desktop NVIDIA GPU (energy via nvidia-smi):
python scripts/bench_device.py --config <run>/config.yaml --checkpoint <run>/ckpt \
  --device-name laptop-4070 --precision fp16

# NVIDIA Jetson (energy via tegrastats) — the canonical edge target:
python scripts/bench_device.py --config <run>/config.yaml --checkpoint <run>/ckpt \
  --device-name jetson-nx --precision fp16

# AMD / macOS / CPU (no power sampler):
python scripts/bench_device.py --config <run>/config.yaml --checkpoint <run>/ckpt \
  --device-name macbook --energy none
```

Energy and peak-GPU-memory sampling are NVIDIA (`nvidia-smi`) and Jetson (`tegrastats`) only; latency
and throughput are measured everywhere. See [`docs/running_sweeps.md`](running_sweeps.md) §6.5.

---

## 7. Troubleshooting

| Symptom | Cause → Fix |
| --- | --- |
| `torch.cuda.is_available()` is `False` | Wrong/missing torch wheel or driver. Reinstall the matching CUDA/ROCm wheel; check `nvidia-smi`/`rocm-smi`. |
| bf16 errors or is slow (T4/V100/20xx) | No bf16 tensor cores. Set `distributed.mixed_precision: fp16`. |
| CUDA out of memory | Lower `training.batch_size` and/or `training.seq_len`; raise `training.n_batches` to keep the token budget. |
| `NaN`/`inf` loss | fp16 instability → switch to `bf16`, or lower `optimizer.lr` / warm up longer. |
| Tokenizer outputs look byte-level / vocab tiny | tiktoken couldn't fetch GPT-2 assets (offline). Warm `TIKTOKEN_CACHE_DIR` on a node with internet (§5.1). |
| HF `datasets` fails offline (HPC) | Pre-tokenize on a login node; copy `dataspace/fineweb/*.npy` to the compute filesystem. |
| macOS: "op not implemented for MPS" | `export PYTORCH_ENABLE_MPS_FALLBACK=1`; keep `mixed_precision: no` and `compile: false`. |
| `peak_mem_bytes` / `mfu` / energy are 0 / null | Expected off NVIDIA-CUDA / Jetson; quality + decode tok/s are still valid. |
| Run died (spot reclaim / SLURM time / Ctrl-C) | An `emergency.pt` was saved; resume with `python trainer.py -ct` (re-sync from S3/Drive/`$WORK` first). |
| `NotImplementedError: distributed.backend ...` | DDP/FSDP isn't wired. Use single-GPU jobs + arrays/parallel processes (§3, §5.3). |

---

## Appendix: config override snippets

`trainer.py` takes `--config`, `--mode`, and `-ct`; per-environment knobs live in the YAML. Copy a base
config and edit (or append) these sections.

**Free Colab T4 / Turing / Volta — fp16 + smaller batch + Drive paths:**

```yaml
training:
  device: cuda
  batch_size: 4
  n_batches: 8
  ckpt: /content/drive/MyDrive/llm_toaster_runs/base_ckpt
  ckpt_config: /content/drive/MyDrive/llm_toaster_runs/base_config.yaml
distributed:
  mixed_precision: fp16
checkpointing:
  output_dir: /content/drive/MyDrive/llm_toaster_runs
logging:
  log_file: /content/drive/MyDrive/llm_toaster_runs/train.log
  metrics_file: /content/drive/MyDrive/llm_toaster_runs/metrics.jsonl
```

**Apple Silicon (MPS) — fp32:**

```yaml
training:
  device: mps
distributed:
  mixed_precision: "no"
  compile: false
```

**AMD/ROCm — bf16, eager attention if SDPA misbehaves:**

```yaml
training:
  device: cuda            # ROCm exposes AMD GPUs via the cuda API
distributed:
  mixed_precision: bf16
attention:
  backend: eager          # only if sdpa is flaky on your ROCm build
```

**A100/Ada/H100 (HPC, cloud, high-end desktop) — bf16, larger budget, MFU:**

```yaml
distributed:
  mixed_precision: bf16
logging:
  device_peak_flops: 3.12e14   # A100 bf16 peak (set to your card) -> enables MFU
```
