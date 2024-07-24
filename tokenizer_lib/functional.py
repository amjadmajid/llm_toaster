import tiktoken
import numpy as np

def init_gpt2_tokenizer():
    global enc, eot
    enc = tiktoken.get_encoding("gpt2")
    eot = enc._special_tokens['<|endoftext|>']  # End of text token

def gpt2_encode(doc, dtype=np.uint16):
    prompt = {}
    prompt['text'] = doc 
    return gpt2_encode_hf(prompt, dtype=dtype)

def gpt2_encode_hf(doc, dtype=np.uint16):
    """Tokenizes a single document and returns a numpy array of uint16 tokens"""
    global enc, eot
    print(f"DOC {doc=}")
    tokens = [eot]  # The special token delimits all documents
    tokens.extend(enc.encode_ordinary(doc['text']))

    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "Token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(dtype)
    return tokens_np_uint16

def gpt2_decode(tokens):
    """Decodes a numpy array of uint16 tokens into a string"""
    global enc, eot
    assert tokens[0] == eot, "Invalid start token"
    return enc.decode(tokens[1:])
