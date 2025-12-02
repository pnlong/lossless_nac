"""Implements a lossless compressor with language models (arithmetic coding) for audio data."""

from collections.abc import Iterator
import functools
from typing import Callable
from math import ceil

import haiku as hk
import numpy as np
import torch.nn.functional as F

from language_modeling_is_compression import arithmetic_coder
from language_modeling_is_compression import constants
from language_modeling_is_compression import transformer
from language_modeling_is_compression import utils
from language_modeling_is_compression import utils_audio
from language_modeling_is_compression import constants_audio


def _retrieve_model_params() -> hk.Params:
  """Returns the trained model parameters.

  Raises:
    FileNotFoundError if the file params.npz does not exist yet, in which case
    the user should launch a training with train.py first.
  """
  try:
    with np.load('params.npz', allow_pickle=True) as data:
      return {key: data[key].item() for key in data.files}
  except FileNotFoundError as exc:
    raise FileNotFoundError(
        'You must train a model first, the parameters file params.npz does not'
        ' exist yet.'
    ) from exc


def _retrieve_predict_fn(
    params: hk.Params,
) -> Callable[[np.ndarray], np.ndarray]:
  """Returns the prediction function for the trained model."""
  config = transformer.TransformerConfig(vocab_size=constants.ALPHABET_SIZE)
  model = hk.transform(
      functools.partial(transformer.transformer_decoder, config=config)
  )
  return lambda x: model.apply(params, None, x)


def compress(
    data: bytes,
    return_num_padded_bits: bool = False,
    return_loss_and_bpb: bool = False,
    use_slow_lossless_compression: bool = False,
) -> bytes | tuple[bytes, int]:
  """Compresses the `data` using arithmetic coding and a pretrained model.

  Args:
    data: The data to be compressed.
    return_num_padded_bits: Whether to return the number of zeros added to the
      encoded bitstream in order to make it byte-decodeable (i.e., divisible by
      8). Usually, this is used when the encoded data has to be decoded again.
    return_loss_and_bpb: Whether to return the loss and bits per byte for the
      compressed data.
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
  params = _retrieve_model_params()
  predict_fn = _retrieve_predict_fn(params)

  # Convert the data into ASCII-decodable bytes
  data, discarded_lsbs, _ = utils_audio.right_shift_bytes_by_one(data)

  # Convert the `data` into an array of integers (representing the bytes).
  sequence_array = np.frombuffer(data, dtype=np.uint8)

  if use_slow_lossless_compression:
    log_probs = list()
    for subsequence_length in range(len(sequence_array)):
      subsequence_probs = predict_fn(
          sequence_array[None, : subsequence_length + 1]
      )
      log_probs.append(subsequence_probs[0, -1])
    log_probs = np.vstack(log_probs)
  else:
    log_probs = predict_fn(sequence_array[None])[0, ...]
  if return_loss_and_bpb: # output loss and bits per byte for each chunk
    compressed_loss = F.nll_loss(input=log_probs, target=sequence_array).item() # loss is the negative log-likelihood of the predicted tokens
    compressed_bpb = compressed_loss / np.log(2).item()
  probs = np.exp(log_probs)

  output = list()
  encoder = arithmetic_coder.Encoder(
      base=constants.ARITHMETIC_CODER_BASE,
      precision=constants.ARITHMETIC_CODER_PRECISION,
      output_fn=output.append,
  )
  for pdf, symbol in zip(probs, sequence_array):
    encoder.encode(utils.normalize_pdf_for_arithmetic_coding(pdf), symbol)
  encoder.terminate()

  compressed_bits = ''.join(map(str, output))
  compressed_bytes, num_padded_bits = utils.bits_to_bytes(compressed_bits)

  # Add the discarded LSBs to the compressed data
  compressed_bytes = compressed_bytes + discarded_lsbs

  # Return requested output
  if return_loss_and_bpb: # return loss and bits per byte for the compressed data
    if return_num_padded_bits: # return number of padded bits if requested
      return compressed_bytes, num_padded_bits, compressed_loss, compressed_bpb
    else: # don't return number of padded bits
      return compressed_bytes, compressed_loss, compressed_bpb
  else:
    if return_num_padded_bits: # return number of padded bits if requested
      return compressed_bytes, num_padded_bits
    else: # don't return number of padded bits or compression loss and bits per byte
      return compressed_bytes


def decompress(
    data: bytes,
    num_padded_bits: int = 0,
    uncompressed_length: int = constants.CHUNK_SIZE_BYTES,
) -> bytes:
  """Decompresses the `data` using arithmetic coding and a pretrained model.

  See https://en.wikipedia.org/wiki/Arithmetic_coding for details.

  Args:
    data: The data to be decompressed.
    num_padded_bits: The number of zeros added to the encoded bitstream in order
      to make it byte-decodeable (i.e., divisble by 8).
    uncompressed_length: The length of the original data stream (in bytes).

  Returns:
    The decompressed data.
  """
  params = _retrieve_model_params()
  predict_fn = _retrieve_predict_fn(params)

  # Extract the discarded LSBs from the compressed data
  discarded_lsbs_length = ceil(uncompressed_length / 8) # length of the discarded LSBs in bytes
  data, discarded_lsbs = data[:-discarded_lsbs_length], data[-discarded_lsbs_length:]

  data_iter = iter(utils.bytes_to_bits(data, num_padded_bits=num_padded_bits))

  # The decoder requires a function that reads digits from {0, 1, ..., base - 1}
  # from the compressed input and returns `None` when the input is exhausted.
  def _input_fn(bit_sequence: Iterator[str] = data_iter) -> int | None:
    try:
      return int(next(bit_sequence))
    except StopIteration:
      return None

  decoder = arithmetic_coder.Decoder(
      base=constants.ARITHMETIC_CODER_BASE,
      precision=constants.ARITHMETIC_CODER_PRECISION,
      input_fn=_input_fn,
  )
  # We need a dummy token because the language model right-shifts the sequence
  # by one when computing the conditional probabilities. Concretely, at every
  # step, we need the `pdf` of the next token given all currently decompressed
  # tokens, but without a dummy token, the last `pdf` would be that of the last
  # already decompressed token. The value of the dummy token is irrelevant.
  sequence_array = np.empty((1,), dtype=np.uint8)
  probs = np.exp(predict_fn(sequence_array[None])[0, ...])

  for idx in range(uncompressed_length):
    token = decoder.decode(
        utils.normalize_pdf_for_arithmetic_coding(probs[idx])
    )
    sequence_array = np.insert(sequence_array, -1, token)
    probs = np.exp(predict_fn(sequence_array[None])[0, ...])

  # Remove the dummy token and convert to bytes.
  reconstructed_data = sequence_array[:-1].tobytes()

  # Add the discarded LSBs to the reconstructed data
  reconstructed_data = utils_audio.add_discarded_lsbs_back(reconstructed_data, discarded_lsbs)

  return reconstructed_data
