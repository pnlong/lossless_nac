"""Implements a lossless compressor with language models (arithmetic coding) for audio data."""

from collections.abc import Iterator
import warnings
from typing import Any
from math import ceil

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaTokenizerFast
from transformers.utils import logging as transformers_logging

from language_modeling_is_compression import constants_audio
from language_modeling_is_compression import arithmetic_coder
from language_modeling_is_compression import constants
from language_modeling_is_compression import utils
from language_modeling_is_compression import utils_audio

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
    llama_model: str = constants_audio.DEFAULT_LLAMA_MODEL,
    silence_progress_bar: bool = True,
) -> dict[str, Any]:
  """Loads HuggingFace Llama model with quantization.
  
  Args:
    llama_model: Model identifier ('llama-2-7b', 'llama-2-13b', 'llama-2-70b')
    silence_progress_bar: Whether to silence the progress bar.

  Returns:
    Dictionary with model, tokenizer, and device info.
  """
  if silence_progress_bar:
    transformers_logging.disable_progress_bar()
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
  if silence_progress_bar:
    transformers_logging.enable_progress_bar()
  return model_info


def _bytes_to_ascii_text(byte_array: bytes) -> str:
  """Converts byte array to ASCII string (only 7 LSB since MSB=0 after masking)."""
  # Each byte is in range 0-127 (ASCII) after masking
  return byte_array.decode('ascii', errors='ignore')


def _ascii_text_to_bytes(text: str) -> bytes:
  """Converts ASCII string back to byte array."""
  return text.encode('ascii', errors='ignore')


def _get_dummy_token_for_lossless_compression() -> torch.Tensor:
  """Gets dummy token for lossless compression."""
  return torch.zeros((1,), dtype=torch.int64)


def _predict_fn(
    model: LlamaForCausalLM,
    tokenizer: LlamaTokenizerFast,
    token_array: torch.Tensor,
    use_top_k_filtering: bool = constants_audio.LLAMA_USE_TOP_K,
) -> torch.Tensor:
  """Get token predictions from model for given token array.
  Args:
    model: The Llama model to use.
    tokenizer: The tokenizer to use.
    token_array: Array of token IDs (including dummy token at the end) of shape (1, l)
    use_top_k_filtering: Whether to use top-k filtering.
  Returns:
    Tensor of shape (1, l, V) with token log-probabilities after top-k filtering
  """

  # get log probs from model, which yields shape (l, V)
  input_ids = token_array.to(model.device)
  V = len(tokenizer) # vocabulary size
  l = token_array.shape[-1]
  with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits[0] # Shape: (l, V)
    logits = logits.to(torch.float32)
    assert logits.shape == (l, V), f"logits shape: {logits.shape} does not match (l, V): {(l, V)}"
  log_probs = torch.nn.functional.log_softmax(logits, dim=-1) # Shape: (l, V)

  # apply top-k filtering
  if use_top_k_filtering and constants_audio.TOP_K < V: # if top-k is less than the vocabulary size, apply top-k filtering
    log_probs_topk = torch.full((l, V), -torch.inf, dtype=log_probs.dtype).to(log_probs.device) # (l, V) tensor filled with -inf
    top_k_indices = torch.argsort(log_probs, dim=-1)[:, -constants_audio.TOP_K:] # get top-k indices (k most likely tokens) (indices of the top-k probabilities), Shape: (l, constants_audio.TOP_K)
    log_probs_topk[:, top_k_indices] = log_probs[:, top_k_indices] # set top-k probabilities to the original probabilities
    log_probs = log_probs_topk - torch.logsumexp(log_probs_topk, dim=-1, keepdim=True)
    del log_probs_topk, top_k_indices # free memory
  else: # if top-k is greater than the vocabulary size, use the full distribution
    pass

  # return log-probabilities, which is shape (1, l, V)
  log_probs = torch.unsqueeze(log_probs, dim=0) # Shape: (1, l, V), add batch dimension
  return log_probs


def compress(
    data: bytes,
    llama_model: str = constants_audio.DEFAULT_LLAMA_MODEL,
    return_num_padded_bits: bool = False,
    use_slow_lossless_compression: bool = constants_audio.USE_SLOW_LOSSLESS_COMPRESSION_FOR_EVALS,
) -> bytes | tuple[bytes, int]:
  """Compresses the `data` using arithmetic coding and a Llama model.

  Args:
    data: The data to be compressed.
    llama_model: The Llama model to use.
    return_num_padded_bits: Whether to return the number of zeros added to the
      encoded bitstream in order to make it byte-decodeable (i.e., divisible by
      8). Usually, this is used when the encoded data has to be decoded again.
    use_slow_lossless_compression: Whether to compute the `pdf`s for all tokens
      in the data stream in one go or separately for every proper subsequence.
      When only compressing data (i.e., without decompression) use the first
      approach (i.e., `False`) since it has an O(n) runtime complexity, while
      the latter is O(n^2). However, the goal is to losslessly decompress the
      compressed output, use the second option (i.e., `True`) since this is what
      happens in the decoder (which iteratively reconstructs the sequence).

  Returns:
    The compressed data.
  """

  raise NotImplementedError("Trilobyte is not implemented yet.")

  # STEP 0: Convert the data into ASCII-decodable bytes
  data, discarded_lsbs, _ = utils_audio.right_shift_bytes_by_one(data)
    
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
  sequence_array = input_ids[0] # Shape: (l,)
  post_tokenization_length = len(sequence_array) # post_tokenization_length = l

  # get probabilities from model
  dummy_token = _get_dummy_token_for_lossless_compression().to(sequence_array.device) # Shape: (1,)
  if use_slow_lossless_compression: # slow lossless compression that matches the decompressor, O(n^2) runtime complexity
    log_probs = torch.zeros((post_tokenization_length, V), dtype=torch.float32).to(sequence_array.device) # Shape: (l, V)
    for subsequence_length in range(post_tokenization_length):
      subsequence_probs = _predict_fn(
          model,
          tokenizer,
          torch.cat((sequence_array[:subsequence_length], dummy_token), dim=0)[None] # Shape: (1, subsequence_length + 1)
      )
      log_probs[subsequence_length, :] = subsequence_probs[0, -1]
  else: # fast lossy compression, O(n) runtime complexity    
    log_probs = _predict_fn(
        model,
        tokenizer,
        torch.cat((sequence_array, dummy_token), dim=0)[None] # Shape: (1, l + 1)
    ) # Shape: (1, l + 1, V)
    log_probs = log_probs[0, 1:, :] # Shape: (l, V)
  if constants_audio.OUTPUT_LOSS_AND_BPB_TO_VERIFY_COMPRESSION: # output loss and bits per byte for each chunk
    loss = F.nll_loss(input=log_probs, target=sequence_array).item() # loss is the negative log-likelihood of the predicted tokens
    bpb = loss / np.log(2).item()
    print(f"Loss: {loss:8.4f}, BPB: {bpb:8.4f}")
    del loss, bpb # free memory
  log_probs = log_probs.cpu().numpy() # convert to numpy array
  probs = np.exp(log_probs)
  del dummy_token, log_probs # free memory
 
  # STEP 6: Use arithmetic coding in token space (vocab size V=32000)
  # "We can pass them to an arithmetic encoder of vocabulary size V"
  output = []
  encoder = arithmetic_coder.Encoder(
      base=constants.ARITHMETIC_CODER_BASE,
      precision=constants.ARITHMETIC_CODER_PRECISION,
      output_fn=output.append,
  )
  for pdf, symbol in zip(probs, sequence_array.tolist()):
    encoder.encode( # encode this token using its distribution
        utils.normalize_pdf_for_arithmetic_coding(pdf), # Shape: (V,); conditional probability for this token
        symbol # encode token_id from vocabulary size V
    )
  encoder.terminate()
  
  # STEP 7: Return compressed output
  # "Size in bytes compared with initial size, i.e., 2048 bytes"
  compressed_bits = ''.join(map(str, output))
  compressed_bytes, num_padded_bits = utils.bits_to_bytes(compressed_bits)

  # Add the discarded LSBs and post tokenization length to the compressed data
  post_tokenization_length_bytes = post_tokenization_length.to_bytes(constants_audio.POST_TOKENIZATION_LENGTH_BYTES, byteorder=constants_audio.POST_TOKENIZATION_LENGTH_ENDIANNESS, signed=False)
  compressed_bytes = post_tokenization_length_bytes + compressed_bytes + discarded_lsbs

  if return_num_padded_bits: # return number of padded bits if requested
    return compressed_bytes, num_padded_bits
  return compressed_bytes


def decompress(
    data: bytes,
    llama_model: str = constants_audio.DEFAULT_LLAMA_MODEL,
    num_padded_bits: int = 0,
    uncompressed_length: int = constants.CHUNK_SIZE_BYTES,
) -> bytes:
  """Decompresses the `data` using arithmetic coding and a Llama model.

  See https://en.wikipedia.org/wiki/Arithmetic_coding for details.

  Args:
    data: The data to be decompressed.
    llama_model: The Llama model to use.
    num_padded_bits: The number of zeros added to the encoded bitstream in order
      to make it byte-decodeable (i.e., divisble by 8).
    uncompressed_length: The length of the original data stream (in bytes).

  Returns:
    The decompressed data.
  """

  raise NotImplementedError("Trilobyte is not implemented yet.")

  # STEP 0: Extract the post tokenization length from the compressed data
  post_tokenization_length = int.from_bytes(data[:constants_audio.POST_TOKENIZATION_LENGTH_BYTES], byteorder=constants_audio.POST_TOKENIZATION_LENGTH_ENDIANNESS, signed=False)
  data = data[constants_audio.POST_TOKENIZATION_LENGTH_BYTES:] # remove the post tokenization length from the compressed data
  discarded_lsbs_length = ceil(uncompressed_length / 8) # length of the discarded LSBs in bytes
  data, discarded_lsbs = data[:-discarded_lsbs_length], data[-discarded_lsbs_length:] # extract the discarded LSBs from the compressed data

  # STEP 1: Setup decoder
  model_info = _load_llama_model(llama_model)
  model = model_info['model']
  tokenizer = model_info['tokenizer']
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
  dummy_token = _get_dummy_token_for_lossless_compression().to(model.device) # Shape: (1,)
  sequence_array = torch.empty((post_tokenization_length,), dtype=dummy_token.dtype).to(model.device) # Shape: (l,); all zeros
  for subsequence_length in range(post_tokenization_length):
    log_probs = _predict_fn(
        model,
        tokenizer,
        torch.cat((sequence_array[:subsequence_length], dummy_token), dim=0)[None] # Shape: (1, subsequence_length + 1)
    )[0, ...] # Shape: (subsequence_length + 1, V)
    probs_for_token = np.exp(log_probs[-1].cpu().numpy()) # Shape: (V,); conditional probability for this token
    token = decoder.decode(
        utils.normalize_pdf_for_arithmetic_coding(probs_for_token)
    )
    sequence_array[subsequence_length] = token
  del dummy_token # free memory
  
  # STEP 3: Convert tokens back to ASCII string
  decoded_tokens = sequence_array.cpu().numpy().tolist() # convert to list
  ascii_text = tokenizer.decode(decoded_tokens) # convert tokens back to ASCII string
  del sequence_array, decoded_tokens # free memory
  
  # STEP 4: Convert ASCII back to bytes
  reconstructed_data = _ascii_text_to_bytes(ascii_text) # convert ASCII string back to byte array

  # Add the discarded LSBs to the reconstructed data
  reconstructed_data = utils_audio.add_discarded_lsbs_back(reconstructed_data, discarded_lsbs)

  return reconstructed_data