# tokenizer_lib.py

from transformers import GPT2Tokenizer

def init_tokenizer():
    """
    Initializes the GPT2 tokenizer with additional special tokens.
    
    Returns:
        tokenizer (GPT2Tokenizer): The initialized tokenizer.
    """
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    special_tokens = {
        'additional_special_tokens': ['<|user|>', '<|assistant|>', '<|endoftext|>', '<|pad|>'],
        'pad_token': '<|pad|>'
    }
    tokenizer.add_special_tokens(special_tokens)
    return tokenizer


def tokenize_sample(sample, tokenizer):
    """
    Tokenizes a single sample containing 'question' and 'answer' fields.
    
    Args:
        sample (dict): A dictionary with 'question' and 'answer' keys.
        tokenizer (GPT2Tokenizer): The tokenizer to use.
    
    Returns:
        tokens (list): List of token IDs.
    """
    if 'question' not in sample or 'answer' not in sample:
        raise ValueError("Sample must have 'question' and 'answer' keys")

    question = sample['question'].strip()
    answer = sample['answer'].strip()

    # Format the text with special tokens
    text = f"<|user|>\n{question}\n<|assistant|>\n{answer}<|endoftext|>"

    # Tokenize the text
    tokens = tokenizer.encode(text, add_special_tokens=False, 
                              truncation=True, 
                              #TODO: set the max length to the seq_len from the config file 
                              max_length=1024) # False: means we are manually inserting the special toekens
    return tokens
