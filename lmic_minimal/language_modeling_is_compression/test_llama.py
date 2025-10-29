import sys
sys.path.insert(0, "/home/pnlong/lnac/lmic_minimal")
from language_modeling_is_compression.compressors_audio import llama
from language_modeling_is_compression import constants_audio
from language_modeling_is_compression import constants
from language_modeling_is_compression import arithmetic_coder
from language_modeling_is_compression import utils
from collections.abc import Iterator

import numpy as np
import torch
from tqdm import tqdm

data = np.random.randint(low=-100, high=100, size = (2048), dtype = np.int8).tobytes()
llama_model = 'llama-2-7b'
return_num_padded_bits = True

ascii_text = llama._bytes_to_ascii_text(data)
# print(f"ascii_text: {ascii_text}")

model_info = llama._load_llama_model(llama_model)
model = model_info['model']
tokenizer = model_info['tokenizer']
T = len(tokenizer) # vocabulary size
print(f"T: {T}")
encoded = tokenizer(ascii_text, return_tensors='pt', add_special_tokens=False).to(model.device)
# print(f"encoded keys: {encoded.keys()}")
input_ids = encoded['input_ids']
# print(f"tokens: {tokens}")
l = input_ids.shape[1] # "the length of the sequence has now completely changed"
print(f"l: {l}")
print(f"input_ids shape: {input_ids.shape}")
# print(f"input_ids: {input_ids}")
with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits[0]  # Shape: (l, T) where T=32000
    print(f"logits dtype: {logits.dtype}")
    logits = logits.to(torch.float16)
    print(f"logits dtype: {logits.dtype}")
    assert logits.shape == (l, T), f"logits shape: {logits.shape} does not match (l, T): {(l, T)}"
    print(f"logits shape: {logits.shape}")
log_probs = torch.nn.functional.log_softmax(logits, dim=-1).cpu().numpy()  
print(f"log_probs shape: {log_probs.shape}")

log_probs_topk = np.full((l, T), -np.inf) # (l, T) array filled with -inf
for i in range(l):
    pos_log_probs = log_probs[i] # Shape: (T,); for each position i, get the conditional distribution
    top_k_indices = np.argsort(pos_log_probs)[-constants_audio.TOP_K:] # get top-k indices (k most likely tokens) (indices of the top-k probabilities)
    log_probs_topk[i, top_k_indices] = pos_log_probs[top_k_indices] # set top-k probabilities to the original probabilities  

print(f"log_probs_topk shape: {log_probs_topk.shape}")

probs = np.exp(log_probs_topk) # convert to probabilities from log-probabilities
probs = probs / np.sum(probs, axis=-1, keepdims=True) # renormalize each row to sum to 1
print(f"probs sums to 1: {np.allclose(np.sum(probs, axis=-1), 1)}")

print(f"COMPRESSING DATA...")
compressed_data = llama.compress(data, llama_model, return_num_padded_bits=False)
print(f"compressed data: {compressed_data}")



print(f"DECOMPRESSING DATA...")
post_tokenization_length = l
num_padded_bits = 0
model_info = llama._load_llama_model(llama_model)
model = model_info['model']
tokenizer = model_info['tokenizer']
T = len(tokenizer) # vocabulary size
data_iter = iter(utils.bytes_to_bits(compressed_data, num_padded_bits=num_padded_bits))
# the decoder requires a function that reads digits from {0, 1, ..., base - 1}
# from the compressed input and returns `None` when the input is exhausted.
def _input_fn(bit_sequence: Iterator[str] = data_iter) -> int | None:
    try:
        return int(next(bit_sequence))
    except StopIteration:
        return None
decoder = arithmetic_coder.Decoder( # initialize arithmetic decoder
  base=constants.ARITHMETIC_CODER_BASE,
  precision=constants.ARITHMETIC_CODER_PRECISION,
  input_fn=_input_fn,
)

# STEP 2: Reconstruct tokens one by one using dummy token approach
# We need a dummy token because the language model right-shifts the sequence
# by one when computing the conditional probabilities. Concretely, at every
# step, we need the pdf of the next token given all currently decompressed
# tokens, but without a dummy token, the last pdf would be that of the last
# already decompressed token. The value of the dummy token is irrelevant.
def _get_token_predictions(token_array: np.ndarray) -> np.ndarray: # helper function to get token predictions from model for given token array
    """Get token predictions from model for given token array.
    Args:
        token_array: Array of token IDs (including dummy token at the end)
    Returns:
        Array of shape (len(token_array), T) with token probabilities after top-k filtering
    """
    input_ids = torch.tensor([token_array.tolist()], device=model.device)
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0] # Shape: (len(token_array), T)
        logits = logits.to(torch.float16)
        assert logits.shape == (len(token_array), T), f"logits shape: {logits.shape} does not match (len(token_array), T): {(len(token_array), T)}"
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).cpu().numpy()
    probs_list = []
    for i in range(len(token_array)): # apply top-k filtering
        pos_log_probs = log_probs[i]
        top_k_indices = np.argsort(pos_log_probs)[-constants_audio.TOP_K:]
        filtered = np.full(T, -np.inf)
        filtered[top_k_indices] = pos_log_probs[top_k_indices]
        probs = np.exp(filtered)
        probs = probs / np.sum(probs)  # Renormalize
        probs_list.append(probs)
    probs_list = np.array(probs_list)
    # probs_list = np.clip(probs_list, 0, 1 - 1e-10) # clamp to valid range (avoid numerical issues)
    return probs_list
token_array = np.zeros((1,), dtype=np.int64) # initialize with dummy token (0)
probs = _get_token_predictions(token_array) # Shape: (1, T)
for idx in tqdm(range(post_tokenization_length)):
    # print(f"idx: {idx}, probs shape: {probs[idx].shape}")
    token_id = decoder.decode(utils.normalize_pdf_for_arithmetic_coding(probs[idx])) # decode next token
    # print(f"token_id: {token_id}, dtype: {token_id.dtype}")
    token_array = np.insert(token_array, -1, token_id) # insert token before the last position (before dummy token)
    probs = _get_token_predictions(token_array) # get predictions for the updated sequence

# STEP 3: Convert tokens back to ASCII string
print(f"token_array shape: {token_array.shape}")
decoded_tokens = token_array[:-1].tolist() # remove the dummy token and convert to list
ascii_text = tokenizer.decode(decoded_tokens) # convert tokens back to ASCII string
print(f"ascii_text: {ascii_text}")
assert data == llama._ascii_text_to_bytes(ascii_text) # convert ASCII string back to byte array


# try to decompress for real
decompressed_data = llama.decompress(compressed_data, post_tokenization_length = l, llama_model = llama_model)
print(f"decompressed data: {decompressed_data}")
