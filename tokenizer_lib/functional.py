import tiktoken
import numpy as np

def init_gpt2_tokenizer():
    """
    Initialize the GPT-2 tokenizer.
    
    This sets up the global encoder (`enc`) and the end-of-text token (`eot`).
    """
    global enc, eot
    try:
        enc = tiktoken.get_encoding("gpt2")
        eot = enc._special_tokens['<|endoftext|>']  # End of text token
    except Exception as e:
        raise RuntimeError(f"Failed to initialize GPT-2 tokenizer: {e}")

def gpt2_encode(doc, dtype=np.uint16):
    """
    Encode a document using the GPT-2 tokenizer.
    
    Args:
    - doc (str): The document to encode.
    - dtype (numpy.dtype): The dtype for the output numpy array (default: np.uint16).
    
    Returns:
    - numpy.ndarray: The encoded tokens as a numpy array.
    """
    prompt = {'text': doc}
    return gpt2_encode_hf(prompt, dtype=dtype)

def gpt2_encode_hf(doc, dtype=np.uint16):
    """
    Tokenizes a single document and returns a numpy array of uint16 tokens.
    
    Args:
    - doc (dict): A dictionary with the document text under the 'text' key.
    - dtype (numpy.dtype): The dtype for the output numpy array (default: np.uint16).
    
    Returns:
    - numpy.ndarray: The encoded tokens as a numpy array of dtype uint16.
    """
    global enc, eot
    if 'text' not in doc:
        raise ValueError("Document must have a 'text' key")
    
    tokens = [eot]  # The special token delimits all documents
    tokens.extend(enc.encode_ordinary(doc['text']))

    tokens_np = np.array(tokens, dtype=dtype)
    if not ((0 <= tokens_np).all() and (tokens_np < 2**16).all()):
        raise ValueError("Token dictionary too large for uint16")

    return tokens_np

def gpt2_decode(tokens):
    """
    Decode a numpy array of uint16 tokens into a string.
    
    Args:
    - tokens (numpy.ndarray): The array of tokens to decode.
    
    Returns:
    - str: The decoded string.
    """
    global enc, eot
    if len(tokens) == 0 or tokens[0] != eot:
        raise ValueError("Invalid token array: must start with the end-of-text token")
    
    return enc.decode(tokens[1:])

# Example usage
if __name__ == "__main__":
    init_gpt2_tokenizer()
    test_text = "Hello, this is a test document."
    encoded = gpt2_encode(test_text)
    print(f"Encoded: {encoded}")
    decoded = gpt2_decode(encoded)
    print(f"Decoded: {decoded}")
