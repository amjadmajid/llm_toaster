# LLM Toaster

<img style="max-width:400px" src="assets/images/llmtoster.jpg" alt="LLM Toster Logo">

## Overview
LLM Toster is a library designed for training, fine-tuning, and running inference on Transformer-based language models. It streamlines the process of dataset loading, model configuration, and execution.

## Installation
To install the LLM Toster library, navigate to the root directory and run the following command:
```bash
pip install -e .
```
This will install all necessary packages and dependencies.

## Configuration
All configurations can be found in the `llm_toaster/config` directory. Ensure to review and modify these configurations according to your setup and requirements.

## Training
### Step 1) data download and tokenization
To download and tokenize the [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset from Hugging Face, navigate to the `dataspace/src` directory and run:
```bash
python download_tokenize_hf.py
```
This command will take a file to download and tokenize about 27GB (after tokenization, the size will be reduced to about 10GB)

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
You can continue interacting with the model by providing new prompts, or type `exit` to quit.

## Troubleshooting
If you encounter any issues, please check the following:
- Ensure all dependencies are installed.
- Verify the configurations in `config/config.yaml`.
- Make sure the dataset is downloaded and tokenized correctly.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your improvements or bug fixes.

## License
LLM Toster is released under the MIT License. See `LICENSE` for more details.

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

```bash
pip install -e .
pip install torch numpy pyyaml tiktoken
```

Optional integrations such as Hugging Face tokenizers, SentencePiece, FlashAttention 2, and xFormers are detected lazily and fail with clear errors if they are unavailable.

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
  use_gqa: false
  sliding_window: null
  dropout: 0.0
```

The model uses PyTorch scaled-dot-product attention for SDPA-style backends and an educational eager implementation for reference. Optional `flash_attn_2` and `xformers` paths are placeholders that report missing integrations cleanly.

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

Checkpoints are managed by `llm_toaster/toaster/training/checkpointing.py` and include model, optimizer, scheduler, scaler, config, global step, tokens seen, RNG state, best metric, data state where available, and the current git commit when available.

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
