# LLM Toaster

<img style="max-width:400px" src="assets/images/llmtoster.jpg" alt="LLM Toaster Logo">

## Overview
LLM Toaster is a library designed for training, fine-tuning, and running inference on Transformer-based language models. It streamlines the process of dataset loading, model configuration, and execution.

## Installation
From the repository root (Python 3.11+; packaging is defined in `pyproject.toml`):
```bash
pip install -e .            # core deps: torch, numpy, tiktoken, pyyaml
pip install -e ".[data]"    # extra: datasets + tqdm (only for the HF download/tokenize script)
pip install -e ".[dev]"     # extra: pytest, ruff, mypy, pre-commit (development)
```

## Configuration
Configs live in `config/`: `config/default_config.yaml` (full size) and
`config/smoke_test_config.yaml` (tiny, for fast local checks). See
[Model architecture & options](#model-architecture--options) for the tunable fields.

## Model architecture & options

The model is a **dense, decoder-only Transformer** (GPT-style, pre-norm, tied embeddings by
default, GPT-2 style weight init so training starts near `ln(vocab)`). Architecture is driven by
the `model:` / `attention:` config sections, so you can vary **one axis at a time** for comparison.

| Field | Options | Notes |
|---|---|---|
| `model.n_blocks` / `model.n_embd` / `model.n_head` | ints | depth / width / heads |
| `model.position` | `learned` \| `rope` \| `none` | RoPE needs an even head_dim; `none` = no positions |
| `model.norm` | `layernorm` \| `rmsnorm` | RMSNorm is cheaper |
| `model.ffn` | `gelu` \| `geglu` \| `swiglu` | `geglu`/`swiglu` are gated (extra projection) |
| `model.ffn_mult` | int (default 4) | FFN hidden width = `ffn_mult * n_embd` |
| `model.num_key_value_heads` | int (default = `n_head`) | GQA/MQA — fewer KV heads → smaller KV-cache |
| `model.tie_embeddings` | bool (default `true`) | share input/output embedding weights |

Reserved but **rejected at config load** (clear `NotImplementedError`): MoE FFN,
sliding-window attention, `flash_attn_2`/`xformers` backends, SentencePiece tokenizer,
and non-`none` `distributed.backend`.

## Logging & metrics

Every run logs a startup architecture summary plus an aligned per-step line, and writes a
machine-readable `logs/metrics.jsonl` (a self-describing `architecture` row + one row per logged
step) for comparing/plotting runs (`logging.metrics_file: ""` disables it):

```text
architecture: decoder_transformer (dense) | positions=learned | norm=layernorm | ffn=gelu
params: 254.1M total | 52.6M embedding | 201.5M non-embedding | ~969.3 MB fp32
attention: n_head=8 kv_heads=8 head_dim=128 (MHA) | KV-cache 64.0 KB/token, 64.0 MB @ seq_len=1024 (fp16)
compute: ~1.72 GFLOP/token (train fwd+bwd)
step    140/100,000 | loss 7.5664 | lr 6.00e-04 | gnorm 1.23 | 21,098 tok/s | seen 11.4M | 11m51s | eta 2h09m
```

Run `python scripts/describe_arch.py --config <cfg>` for a full **architecture card** (Mermaid
dataflow + decoder-block diagrams and a per-component parameter table with exact numbers). See
[`docs/architecture.md`](docs/architecture.md) for the design overview and per-variant diagrams, and
[`docs/running_sweeps.md`](docs/running_sweeps.md) to run a controlled architecture sweep
(matched-parameter training → Pareto table → on-device measurement).

## Training
### Step 1) data: materialize token shards
Pretraining reads **immutable, manifest-described token shards**. Install the data extra
(`pip install -e ".[data]"`), then materialize a dataset described by your config's `data:` section
(streams the source while writing shards; **append-safe** to re-run):
```bash
python scripts/data.py prepare --config config/default_config.yaml --dry-run   # plan: tokens, shards, storage, steps
python scripts/data.py prepare --config config/default_config.yaml             # write manifest.json + shards/
python scripts/data.py inspect  --manifest dataspace/fineweb/manifest.json
```
Pick a token budget in the config (`training.max_tokens` or `max_iter`) so you materialize only what
you need. For Colab / limited disk, set `data.materialization.mode: prefetch` to stream shards in the
background **during** training instead of downloading upfront. Already have loose `.npy` shards?
Migrate them once (no retokenize): `python scripts/data.py migrate-legacy --data-dir dataspace/fineweb
--manifest dataspace/fineweb/manifest.json`. Full guide: [`docs/data-pipeline.md`](docs/data-pipeline.md).

### Step 2) Training
To train the model, navigate to the `llm_toaster` directory and run:
```bash
python trainer.py
```
### Step 3) Continue training 
You can decide to stop the training and continue afterwards. 
To continue training from the last checkpoint run: 
```
python trainer.py -ct
```
NOTE: progress is saved only during a checkpoint which is automatically taken when the loss is reduced. 

## Inference
### Option 1) Train your model
If you just trained your model, then run the following script to extract the model from the checkpoint (a checkpoint consists of a model, an optimizer, and a scaler)
```
python extract_inference_model.py
```
This will extract the model from a checkpoint saved in `model/checkpoints` and save it under the `model/babyGPT`
### Option 2) Download a pretrained model
You can download a pretrained babyGPT model from [HERE](https://huggingface.co/AmjadMajid/BabyGPT/tree/main) and save it under `llm_toaster/model/babyGPT` directory. 
To prompt the model, use the following command:
```bash
python inference.py -p="Your prompt here"
```
`inference.py` builds the model via the same registry/tokenizer as training, and `--model` accepts
either an extracted `.llm` state_dict or a full training checkpoint.

## Troubleshooting
If you encounter any issues, please check the following:
- Ensure all dependencies are installed.
- Verify the configuration in `config/default_config.yaml`.
- Make sure data is materialized: `python scripts/data.py validate --manifest dataspace/fineweb/manifest.json`.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your improvements or bug fixes.

## License
LLM Toaster is released under the MIT License. See `LICENSE` for more details.

## Base training and supervised fine-tuning

LLM Toaster now has two explicit training modes:

- **Base pretraining** (`training.mode: pretrain`) trains a causal language model from tokenized web/text shards and writes a base checkpoint such as `checkpoints/base_ckpt`.
- **Instruction fine-tuning** (`training.mode: finetune` or `python trainer.py --mode finetune`) loads `finetune.base_ckpt`, trains on JSONL instruction data, and writes a separate instructed checkpoint such as `checkpoints/instruct_ckpt`.

A supervised fine-tuning JSONL file can contain any of these schemas:

```jsonl
{"instruction": "Explain gradient accumulation.", "response": "Gradient accumulation ..."}
{"prompt": "Write a haiku about GPUs", "completion": "Silent tensor cores ..."}
{"text": "A fully formatted single training sample."}
```

By default, instruction fine-tuning masks prompt tokens with `-100`, so the loss is applied only to response tokens. Set `finetune.train_on_prompt: true` if you want the model to learn the prompt formatting too.

Typical workflow:

```bash
# 1. Train the base model.
python trainer.py --config config/default_config.yaml --mode pretrain

# 2. Extract a compact inference artifact for the base model.
python extract_inference_model.py \
  --config checkpoints/base_config.yaml \
  --output model/babyGPT/babyGPT_base.llm \
  --output-config model/babyGPT/babyGPT_base.yaml

# 3. Fine-tune from the base checkpoint into a separate instructed model.
python trainer.py --config config/default_config.yaml --mode finetune

# 4. Extract the instructed model.
python extract_inference_model.py \
  --config checkpoints/instruct_config.yaml \
  --output model/babyGPT/babyGPT_instruct.llm \
  --output-config model/babyGPT/babyGPT_instruct.yaml
```

## Development smoke checks

A small config is available for fast local validation without using the full model size:

```bash
python -m unittest tests.test_config_and_data
python trainer.py --config config/smoke_test_config.yaml --mode pretrain
python trainer.py --config config/smoke_test_config.yaml --mode finetune
```

The unit test suite includes config loading checks and supervised fine-tuning loader checks. In minimal environments without optional numeric dependencies installed, data-loader tests are skipped rather than failing at import time.

## Generation controls

Inference exposes common sampling controls:

```bash
python inference.py \
  --config model/babyGPT/babyGPT_instruct.yaml \
  --model model/babyGPT/babyGPT_instruct.llm \
  --prompt "Instruction:\nExplain attention.\n\nResponse:\n" \
  --max-new-tokens 128 \
  --temperature 0.8 \
  --top-k 40 \
  --top-p 0.95
```

## Modular Training Engine

LLM Toaster now keeps the original `trainer.py` entrypoint as a compatibility wrapper while routing training through a modular engine under `llm_toaster/toaster/`.  The engine exposes explicit setup and lifecycle hooks (`setup_model`, `setup_tokenizer`, `setup_dataloaders`, `setup_optimizer`, `setup_scheduler`, `train_step`, `eval_step`, `save_checkpoint`, `load_checkpoint`, and `train`) so the project can stay educational while supporting extension points.

### Install

See [Installation](#installation). Hugging Face tokenizers come from the `[hf]` extra. Options the
engine does not implement are **rejected at config load** with a clear `NotImplementedError` (see
[Model architecture & options](#model-architecture--options)), so a config never silently claims a
capability that isn't there.

### Quick smoke test

```bash
python -m unittest discover -s tests -v
python trainer.py --config config/smoke_test_config.yaml --mode pretrain
python trainer.py --config config/smoke_test_config.yaml --mode finetune
python scripts/train.py --config config/smoke_test_config.yaml --mode pretrain
```

### Pretraining and SFT

`training.mode: pretrain` uses tokenized `.npy` shards through the existing lightweight loader.  `training.mode: finetune` uses the JSONL SFT loader and can still be launched with:

```bash
python trainer.py --config config/default_config.yaml --mode pretrain
python trainer.py --config config/default_config.yaml --mode finetune
```

The richer config schema adds `model`, `tokenizer`, `data`, `optimizer`, `scheduler`, `attention`, `peft`, `checkpointing`, `logging`, `distributed`, and `evaluation` sections while preserving legacy fields such as `training.batch_size`, `training.seq_len`, `training.lr`, `training.ckpt`, `finetune.dataset_path`, `finetune.base_ckpt`, and `finetune.output_ckpt`.

### Supported SFT data formats

The data adapter registry supports:

- Pretraining/text rows: `{"text": "..."}`
- Prompt/completion: `{"prompt": "...", "completion": "..."}`
- Instruction/response: `{"instruction": "...", "response": "..."}`
- Alpaca: `{"instruction": "...", "input": "...", "output": "..."}`
- OpenAI messages: `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`
- ShareGPT: `{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}`
- Preference/DPO rows: `{"prompt": "...", "chosen": "...", "rejected": "..."}`

Prompt tokens are masked with `-100` by default for SFT loss.  Set `finetune.train_on_prompt: true` to include prompt tokens in the loss.

### Attention backends

Configure attention with:

```yaml
attention:
  backend: "sdpa"      # eager | sdpa | sdpa_auto | sdpa_flash | sdpa_mem_efficient | sdpa_math
  sdpa_kernel: "auto"
  sliding_window: null # reserved; a non-null value is rejected at config load
  dropout: 0.0
```

SDPA-style backends use PyTorch `scaled_dot_product_attention`; `eager` is an educational reference
implementation. **GQA/MQA is controlled by `model.num_key_value_heads`** (not an attention flag).
`flash_attn_2` and `xformers` are not wired and are rejected at config load.

### LoRA finetuning

Enable local LoRA without a hard Hugging Face PEFT dependency:

```yaml
peft:
  enabled: true
  method: "lora"
  r: 16
  alpha: 32
  dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj"]
```

When enabled, base parameters are frozen, LoRA adapter parameters remain trainable, and adapter-only state can be saved beside the normal checkpoint.

### Checkpoint and resume

Checkpoints are managed by `llm_toaster/toaster/training/checkpointing.py` and include model, optimizer, scheduler, scaler, config, global step, tokens seen, RNG state, best metric, data state where available, the current git commit when available, and a `format_version`. Resuming with `python trainer.py -ct` restores model + optimizer + scheduler + scaler + RNG state. `load_state_dict_any` also lets tools load weights from a full checkpoint, a legacy `model_state_dict`, or a bare `.llm`.

```yaml
checkpointing:
  output_dir: "checkpoints"
  save_every_steps: 500
  save_best: true
  save_total_limit: 3
  resume_from_checkpoint: null
```

### Planned MoE roadmap

MoE is intentionally not implemented yet. The feed-forward factory includes a placeholder `MoEFFN` so future MoE work can plug into `TransformerBlock` without changing the block interface.
