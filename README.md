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
All configurations are specified in the `config/config.yaml` file. Ensure to review and modify this file according to your setup and requirements.

## Loading Dataset
To download and tokenize the [fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset from Hugging Face, navigate to the `dataspace/loaders` directory and run:
```bash
python download_tokenize_hf.py
```

## Training
To train the model, navigate to the `llm_toster` directory and run:
```bash
python trainer.py
```
or to continue training from the last checkpoint run: 
```
python trainer.py -ct
```

## Inference
To prompt the model, use the following command:
```bash
python inference.py -p="Your prompt here"
```
You can continue interacting with the model by providing new prompts, or type `exit` to quit.

## BabyGPT
Download a pre-trained babyGPT (a 210M parameters model) from [Here](https://drive.google.com/drive/folders/140u7ScsKUuTnj103B9vTY6YbhU_GOykH?usp=sharing)

## Troubleshooting
If you encounter any issues, please check the following:
- Ensure all dependencies are installed.
- Verify the configurations in `config/config.yaml`.
- Make sure the dataset is downloaded and tokenized correctly.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request with your improvements or bug fixes.

## License
LLM Toster is released under the MIT License. See `LICENSE` for more details.
