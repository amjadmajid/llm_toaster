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
