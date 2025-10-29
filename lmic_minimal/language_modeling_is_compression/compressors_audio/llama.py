"""Implements a lossless compressor with language models (arithmetic coding) for audio data."""

from collections.abc import Iterator
import warnings
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from language_modeling_is_compression import constants_audio
from language_modeling_is_compression import arithmetic_coder
from language_modeling_is_compression import constants
from language_modeling_is_compression import utils

if constants_audio.QUANTIZE_LLAMA_MODEL:
  from torchao.quantization import quantize_, Int4WeightOnlyConfig


# map model names to HuggingFace identifiers
_HF_MODEL_MAP = {
  'llama-2-7b': 'meta-llama/Llama-2-7b-hf',
  'llama-2-13b': 'meta-llama/Llama-2-13b-hf',
  'llama-2-70b': 'meta-llama/Llama-2-70b-hf',
}
assert set(_HF_MODEL_MAP.keys()) == set(constants_audio.VALID_LLAMA_MODELS), f'_HF_MODEL_MAP keys: {_HF_MODEL_MAP.keys()} do not match constants_audio.VALID_LLAMA_MODELS: {constants_audio.VALID_LLAMA_MODELS}'


_HF_MODEL_CACHE = dict()


def _load_llama_model(
    llama_model: str,
) -> dict[str, Any]:
  """Loads HuggingFace Llama model with quantization.
  
  Args:
    llama_model: Model identifier ('llama-2-7b', 'llama-2-13b', 'llama-2-70b')

  Returns:
    Dictionary with model, tokenizer, and device info.
  """
  if llama_model not in _HF_MODEL_MAP.keys(): 
    raise ValueError(f'Unknown Llama model: {llama_model}. Valid options: {list(_HF_MODEL_MAP.keys())}')
  hf_model_name = _HF_MODEL_MAP[llama_model] # map model names to HuggingFace identifiers
  if hf_model_name in _HF_MODEL_CACHE.keys():
    return _HF_MODEL_CACHE[hf_model_name]
  model = AutoModelForCausalLM.from_pretrained(
      hf_model_name,
      dtype=torch.bfloat16,
      device_map='auto',
  )
  if constants_audio.QUANTIZE_LLAMA_MODEL:
    try:
      quantize_(model, Int4WeightOnlyConfig(group_size=128)) # load with quantization for memory efficiency
    except Exception as e:
      warnings.warn(f"Could not apply quantization: {e}. Continuing without quantization...")
  tokenizer = AutoTokenizer.from_pretrained(hf_model_name)
  model_info = {
      'model': model,
      'tokenizer': tokenizer,
      'llama_model': llama_model,
  }
  _HF_MODEL_CACHE[hf_model_name] = model_info
  return model_info


def _bytes_to_ascii_text(byte_array: bytes) -> str:
  """Converts byte array to ASCII string (only 7 LSB since MSB=0 after masking)."""
  # Each byte is in range 0-127 (ASCII) after masking
  return byte_array.decode('ascii', errors='ignore')


def _ascii_text_to_bytes(text: str) -> bytes:
  """Converts ASCII string back to byte array."""
  return text.encode('ascii', errors='ignore')


def compress(
    data: bytes,
    llama_model: str,
    return_num_padded_bits: bool = False,
) -> bytes | tuple[bytes, int]:
  """Compresses the `data` using arithmetic coding and a Llama model.

  Args:
    data: The data to be compressed.
    llama_model: The Llama model to use.
    return_num_padded_bits: Whether to return the number of zeros added to the
      encoded bitstream in order to make it byte-decodeable (i.e., divisible by
      8). Usually, this is used when the encoded data has to be decoded again.

  Returns:
    The compressed data.
  """
    
  # STEP 1: Convert bytes to ASCII string (exactly constants.CHUNK_SIZE_BYTES characters)
  ascii_text = _bytes_to_ascii_text(data) # constants.CHUNK_SIZE_BYTES chars, each in range [0, 127 terminated by newline])
  
  # STEP 2: Tokenize using SentencePiece -> l tokens (l ≤ constants.CHUNK_SIZE_BYTES, variable length)
  # "the models immediately tokenize the string using SentencePiece"
  model_info = _load_llama_model(llama_model)
  model = model_info['model']
  tokenizer = model_info['tokenizer']
  V = len(tokenizer) # vocabulary size
  encoded = tokenizer(ascii_text, return_tensors='pt', add_special_tokens=False).to(model.device)
  input_ids = encoded['input_ids'] # Shape: (1, l)
  l = input_ids.shape[1] # length of the post-tokenization sequence
  
  # STEP 3: Feed to Llama -> get l * V log-probabilities
  # "This sequence is fed into the big pretrained Transformer model"
  # "we obtain l * V log-probabilities"
  with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits[0] # Shape: (l, V) where V=32000
    logits = logits.to(torch.float16)
    assert logits.shape == (l, V), f"logits shape: {logits.shape} does not match (l, V): {(l, V)}"
  log_probs = torch.nn.functional.log_softmax(logits, dim=-1).cpu().numpy() # Shape: (l, V)
  
  # STEP 4: Apply top-k filtering (k=100)
  # "the large models had only access to the top-k next token log-probabilities"
  # "We chose k = 100"
  log_probs_topk = np.full((l, V), -np.inf) # (l, V) array filled with -inf
  for i in range(l):
    pos_log_probs = log_probs[i] # Shape: (V,); for each position i, get the conditional distribution
    top_k_indices = np.argsort(pos_log_probs)[-constants_audio.TOP_K:] # get top-k indices (k most likely tokens) (indices of the top-k probabilities)
    log_probs_topk[i, top_k_indices] = pos_log_probs[top_k_indices] # set top-k probabilities to the original probabilities  
  
  # STEP 5: Renormalize top-k probabilities
  # "Accordingly, we renormalize the top-k log-probabilities"
  probs = np.exp(log_probs_topk) # convert to probabilities from log-probabilities
  probs = probs / np.sum(probs, axis=-1, keepdims=True) # renormalize each row to sum to 1
  
  # STEP 6: Use arithmetic coding in token space (vocab size V=32000)
  # "We can pass them to an arithmetic encoder of vocabulary size V"
  output = []
  encoder = arithmetic_coder.Encoder(
      base=constants.ARITHMETIC_CODER_BASE,
      precision=constants.ARITHMETIC_CODER_PRECISION,
      output_fn=output.append,
  )
  for i, token_id in enumerate(input_ids[0].tolist()):
    pdf = probs[i] # Shape: (V,); conditional probability for this token
    encoder.encode( # Encode this token using its distribution
        utils.normalize_pdf_for_arithmetic_coding(pdf),
        token_id # encode token_id from vocabulary size V
    )
  encoder.terminate()
  
  # STEP 7: Return compressed output
  # "Size in bytes compared with initial size, i.e., 2048 bytes"
  compressed_bits = ''.join(map(str, output))
  compressed_bytes, num_padded_bits = utils.bits_to_bytes(compressed_bits)
  if return_num_padded_bits: # return number of padded bits if requested
    return compressed_bytes, num_padded_bits
  return compressed_bytes


def decompress(
    data: bytes,
    post_tokenization_length: int,
    llama_model: str = constants_audio.LLAMA_MODEL,
    num_padded_bits: int = 0,
) -> bytes:
  """Decompresses the `data` using arithmetic coding and a Llama model.

  See https://en.wikipedia.org/wiki/Arithmetic_coding for details.

  Args:
    data: The data to be decompressed.
    post_tokenization_length: The number of tokens in the original sequence after ASCII mapping and tokenization.
    llama_model: The Llama model to use.
    num_padded_bits: The number of zeros added to the encoded bitstream in order
      to make it byte-decodeable (i.e., divisble by 8).

  Returns:
    The decompressed data.
  """

  # STEP 1: Setup decoder
  model_info = _load_llama_model(llama_model)
  model = model_info['model']
  tokenizer = model_info['tokenizer']
  V = len(tokenizer) # vocabulary size
  data_iter = iter(utils.bytes_to_bits(data, num_padded_bits=num_padded_bits))
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
      Array of shape (len(token_array), V) with token probabilities after top-k filtering
    """
    input_ids = torch.tensor([token_array.tolist()], device=model.device)
    with torch.no_grad():
      outputs = model(input_ids)
      logits = outputs.logits[0] # Shape: (len(token_array), V)
      logits = logits.to(torch.float16)
      assert logits.shape == (len(token_array), V), f"logits shape: {logits.shape} does not match (len(token_array), V): {(len(token_array), V)}"
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1).cpu().numpy()
    probs_list = []
    for i in range(len(token_array)): # apply top-k filtering
      pos_log_probs = log_probs[i]
      top_k_indices = np.argsort(pos_log_probs)[-constants_audio.TOP_K:]
      filtered = np.full(V, -np.inf)
      filtered[top_k_indices] = pos_log_probs[top_k_indices]
      probs = np.exp(filtered)
      probs = probs / np.sum(probs)  # Renormalize
      probs_list.append(probs)
    probs_list = np.array(probs_list)
    # probs_list = np.clip(probs_list, 0, 1 - 1e-10) # clamp to valid range (avoid numerical issues)
    return probs_list
  token_array = np.zeros((1,), dtype=np.int64) # initialize with dummy token (0)
  probs = _get_token_predictions(token_array) # Shape: (1, V)
  for i in range(post_tokenization_length):
    token_id = decoder.decode(
        utils.normalize_pdf_for_arithmetic_coding(probs[i])
    ) # decode next token
    token_array = np.insert(token_array, -1, token_id) # insert token before the last position (before dummy token)
    probs = _get_token_predictions(token_array) # get predictions for the updated sequence
  
  # STEP 3: Convert tokens back to ASCII string
  decoded_tokens = token_array[:-1].tolist() # remove the dummy token and convert to list
  ascii_text = tokenizer.decode(decoded_tokens) # convert tokens back to ASCII string
  
  # STEP 4: Convert ASCII back to bytes
  data = _ascii_text_to_bytes(ascii_text) # convert ASCII string back to byte array
  return data