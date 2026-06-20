# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LLM Toaster is an educational library for pretraining, supervised fine-tuning (SFT),
and inference of decoder-only GPT-style Transformers. Training routes through a
modular engine under `llm_toaster/toaster/`; legacy top-level packages remain as
thin compatibility shims for the inference path. All commands are run from the repo root.

## Commands

```bash
# Install (editable). Base deps (torch, numpy, tiktoken, pyyaml) come from pyproject.toml.
pip install -e .
pip install -e ".[data]"   # extra: datasets + tqdm — only for the HF download/tokenize script
pip install -e ".[dev]"    # extra: pytest, pytest-cov, ruff, mypy, pre-commit

# Tests (unittest-style cases; run with pytest for the coverage gate, or unittest)
pip install -e ".[dev]"
pytest                                   # config in pyproject.toml; enforces coverage >= 80%
pytest tests/test_engine_components.py   # single file
pytest tests/test_models_and_lora.py::LoRATests::test_merge_removes_lora_modules_and_preserves_output
python -m unittest discover -s tests -v  # still works (tests are unittest.TestCase)

# Lint + format (also wired as pre-commit hooks)
ruff check .
ruff format --check .

# Fast end-to-end smoke runs on tiny fixtures (no GPU, ~seconds)
python trainer.py --config config/smoke_test_config.yaml --mode pretrain
python trainer.py --config config/smoke_test_config.yaml --mode finetune

# Real training (base pretrain -> extract -> finetune -> extract)
python trainer.py --config config/default_config.yaml --mode pretrain
python trainer.py -ct                                   # resume from training.ckpt
python extract_inference_model.py --config checkpoints/base_config.yaml \
  --output model/babyGPT/babyGPT_base.llm --output-config model/babyGPT/babyGPT_base.yaml
python trainer.py --config config/default_config.yaml --mode finetune

# Inference and generation (both share the engine model + tokenizer + sampler)
python inference.py -p "Your prompt" --config model/babyGPT/babyGPT_base.yaml --model model/babyGPT/babyGPT_base.llm
python scripts/generate.py --config config/default_config.yaml --prompt "Hello"

# Dataset download + tokenization (fineweb-edu from HF)
python dataspace/src/download_tokenize_hf.py
```

CI lives in `.github/workflows/ci.yml`: a lint job (`ruff check` + `ruff format --check`,
plus advisory `mypy`) and a CPU test matrix (Python 3.11/3.12) running pytest with the
coverage gate. Some legacy tests still skip when optional deps (e.g. numpy) are absent;
CI installs the full dependency set so nothing is skipped there.

## Architecture

### Two parallel stacks (read this first)

The repo deliberately keeps two import paths alive. Know which one you are editing:

1. **Modular engine — `llm_toaster/toaster/`** (canonical, used for all training):
   - `config/schema.py` — dataclass `ConfigHandler` with sections: `training`, `model`,
     `tokenizer`, `data`, `optimizer`, `scheduler`, `attention`, `peft`, `checkpointing`,
     `logging`, `distributed`, `evaluation`, `finetune`, `inference`. `from_yaml()` loads
     only the sections present, then runs `apply_backward_compatibility()` and `validate()`.
   - `training/engine.py` — `TrainingEngine`, the heart of the system. Explicit lifecycle
     hooks: `setup_tokenizer` / `setup_model` / `setup_dataloaders` / `setup_optimizer` /
     `setup_scheduler` / `train_step` / `eval_step` / `save_checkpoint` / `load_checkpoint` /
     `train`. Extend behavior by overriding these.
   - `training/checkpointing.py` (`save_checkpoint`/`load_checkpoint`/`rotate_checkpoints`),
     `training/optim.py` (`build_optimizer`/`build_scheduler`).
   - `models/` — `registry.build_model(config)` dispatches on `model.architecture`; pieces
     are `attention.py`, `feedforward.py`, `norms.py`, `transformer.py`.
   - `data/adapters.py` — `DataAdapterRegistry` + `JsonlSFTDataLoader` (SFT).
   - `peft/lora.py` — `inject_lora`, `lora_state_dict`. `tokenizers.py` — `build_tokenizer`.

2. **Legacy top-level packages** (still used for data prep and config re-export):
   - `config/` — re-exports the engine schema (`from llm_toaster.toaster.config.schema import *`),
     so `config.ConfigHandler` IS the engine config.
   - `dataspace/src/data_loader.py` — `DataLoaderLite` (pretrain shard loader) and
     `InstructionDataLoader`; `download_tokenize_hf.py` builds shards. `tokenizer_lib/functional.py`
     provides the gpt2 `encode`/`decode` used by that offline tokenization pipeline.
   - `model/model.py` (`TransformerModel` positional-constructor shim) and `utils/utils.py`
     (`load_checkpoint_`, `save_model`) are **deprecated**: nothing in the repo imports them
     anymore, and `model/model.py` emits a `DeprecationWarning`. Build models via
     `toaster.models.registry.build_model` instead.

   Generation is now unified: `inference.py`, `extract_inference_model.py`, and
   `scripts/generate.py` all build the model via `build_model`, tokenize via `build_tokenizer`,
   and sample through the single `toaster.generation.generate` helper (which calls
   `TransformerModel.generate_text`, supporting `temperature`/`top_k`/`top_p`/`eos`). `trainer.py`
   is a thin wrapper that builds a `ConfigHandler` and calls `TrainingEngine(config).train()`;
   `scripts/train.py` re-invokes `trainer.main`.

### Config backward-compatibility (common gotcha)

Legacy YAMLs (`config/default_config.yaml`, `config/smoke_test_config.yaml`) put model
dimensions and several knobs under the flat `training.*` section. `apply_backward_compatibility()`
in `config/schema.py` mirrors them into the newer sections **only when the newer section is
absent from the YAML**:
- `training.{n_embd,n_head,n_blocks,seq_len,dropout_rate,vocab_size}` → `model.*`
- `training.lr` → `optimizer.lr`; `training.log_*` → `logging.*`; `training.eval_*` → `evaluation.*`
- `training.tokenizer_type` (`gpt2`) → `tokenizer.type` (`tiktoken`)

So when changing model size, edit `training.*` in the legacy YAMLs (not `model.*`), or the
mirror will not apply. `validate()` enforces invariants (e.g. `n_embd % n_head == 0`, allowed
enum values) and raises clear `ValueError`s.

### Training modes

`TrainingEngine.is_finetune_mode` is true when `training.mode in {finetune, sft}` or
`finetune.enabled`. It selects the data path:
- **pretrain** → `DataLoaderLite` over tokenized shards in `training.data_dir`
  (`.npy`, or `.txt`/`.tokens` integer-token text shards used for fixtures).
- **finetune/sft** → `JsonlSFTDataLoader` over `finetune.dataset_path`. Prompt tokens are
  masked to `-100` (loss on response only) unless `finetune.train_on_prompt: true`.
  `DataAdapterRegistry` auto-detects JSONL schemas: `text`, `prompt`/`completion`,
  `instruction`/`response`, Alpaca, OpenAI `messages`, ShareGPT `conversations`,
  preference `chosen`/`rejected` (SFT trains on `chosen`).

Finetune loads `finetune.base_ckpt` before training and writes to `finetune.output_ckpt`.

### Not implemented / placeholders (validate() rejects these at config load)

`ConfigHandler.validate()` (via `_reject_unimplemented`) raises `NotImplementedError` for
options that are spelled correctly (so they pass the enum checks) but have no working code
path — a config can't silently claim a capability the engine lacks:
- `attention.sliding_window` (non-null)
- `distributed.backend` other than `none` (training is single-process only; DDP/accelerate
  are not wired)
- `model.ffn: moe`, `model.position: rope`
- `attention.backend: flash_attn_2` / `xformers`
- `tokenizer.type: sentencepiece`

The real, working options: SDPA/eager attention, learned positions, gelu/geglu/swiglu FFNs,
tiktoken/hf tokenizers, and `mixed_precision: no|fp16|bf16` (fp16 uses a GradScaler).

### Testing layout

All tests run CPU-only on tiny configs/fixtures (`config/smoke_test_config.yaml` +
`tests/fixtures/`) and most avoid the network by using fake tokenizers; the real tokenizer
falls back to a byte-level encoder when tiktoken assets can't be downloaded.

- `tests/test_config_validation.py` — config validation matrix + legacy backward-compat
  mirroring (torch-free).
- `tests/test_config_and_data.py` — legacy stack (`config`, `InstructionDataLoader`); skips if
  numpy is absent.
- `tests/test_models_and_lora.py` — `build_model` across norm × ffn × backend, GQA, MoE/RoPE
  `NotImplementedError`, and LoRA inject/state_dict/merge/adapter-roundtrip.
- `tests/test_engine_components.py` — data adapters/masking, scaler (disabled off-CUDA),
  `train_step`, checkpoint resume, and `generate_text` top_k/top_p.
- `tests/test_training_smoke.py` — full `TrainingEngine.train()` for pretrain + finetune and a
  resume that restores trained weights exactly.
- `tests/test_inference_roundtrip.py` — extract → load (`weights_only`) → `generate` over the
  unified path.
