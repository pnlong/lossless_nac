"""Evaluates a compressor, designed for audio data."""

from collections.abc import Generator
import functools
import time
from typing import Callable

from absl import app
from absl import flags
from absl import logging
import tqdm

from language_modeling_is_compression import constants
from language_modeling_is_compression import constants_audio
from language_modeling_is_compression import data_loaders_audio
from language_modeling_is_compression import utils
from language_modeling_is_compression.compressors_audio import compressor


_COMPRESSOR = flags.DEFINE_enum(
    'compressor',
    'flac',
    compressor.COMPRESSOR_TYPES['classical'] + compressor.COMPRESSOR_TYPES['arithmetic_coding'],
    'Compressor to use.',
)
_DATASET = flags.DEFINE_enum(
    'dataset',
    'musdb18mono',
    data_loaders_audio.GET_AUDIO_DATA_GENERATOR_FN_DICT.keys(),
    'Dataset to use.',
)
_NUM_CHUNKS = flags.DEFINE_integer(
    'num_chunks',
    constants.NUM_CHUNKS,
    'Number of chunks.',
)
_BIT_DEPTH = flags.DEFINE_integer(
    'bit_depth',
    constants_audio.BIT_DEPTH,
    'Bit depth (8, 16, or 24).',
)
_SAMPLE_RATE = flags.DEFINE_integer(
    'sample_rate',
    constants_audio.SAMPLE_RATE,
    'Sample rate (Hz).',
)
_LLAMA_MODEL = flags.DEFINE_enum(
    'llama_model',
    constants_audio.LLAMA_MODEL,
    constants_audio.VALID_LLAMA_MODELS,
    'Llama model to use.',
)


def evaluate_compressor_chunked(
    compress_fn: compressor.Compressor,
    get_data_generator_fn: Callable[[], Generator[bytes, None, None]],
    num_chunks: int,
    count_header_only_once: bool = True,
    mask_fn: Callable[[bytes], tuple[bytes, int]] | None = None,
    use_tqdm: bool = True,
) -> tuple[float, float]:
  """Evaluates the compressor on the chunked dataset.

  Args:
    compress_fn: The function that evaluates data.
    get_data_generator_fn: The function that creates a data generator.
    num_chunks: The number of chunks to consider
    count_header_only_once: Whether to count the header as part of the
      compressed output only once for the whole dataset or for every chunk
      individually.
    mask_fn: The function that masks the data in case the compressor cannot
      handle all possible byte values (e.g., language models can only process
      ASCII-decodable data).
    use_tqdm: Whether to use a progress bar or not.

  Returns:
    The compression rate and the total running time.
  """
  num_missed_bits = running_time = raw_length = compressed_length = 0

  data_generator = get_data_generator_fn()
  if use_tqdm:
    data_generator = tqdm.tqdm(data_generator, desc='Compressing data, chunked', total=num_chunks)

  for data in data_generator:
    if mask_fn is not None:
      data, missed_bits = mask_fn(data)
      num_missed_bits += missed_bits

    t0 = time.perf_counter()
    compressed_data = compress_fn(data)
    t1 = time.perf_counter()

    running_time += t1 - t0
    raw_length += len(data)
    compressed_length += len(compressed_data)

  # Since language models are trained on ASCII strings, they cannot handle all
  # byte values. Thus, we mask the data to be ASCII-decodable by zeroing
  # `num_missed_bits` of the most significant bits. However, this means that we
  # are effectively only compressing `num_bits - num_missed_bits` bits, so we
  # rescale the `compressed_length` to account for this.
  if mask_fn is not None:
    num_bits = 8 * num_chunks * constants.CHUNK_SIZE_BYTES
    compressed_length *= num_bits / (num_bits - num_missed_bits)

  # We only count the header once for classical compressors.
  if count_header_only_once:
    header_length = len(compress_fn((0).to_bytes(1, 'little')))
    compressed_length -= header_length * (num_chunks - 1)

  return compressed_length / raw_length, running_time


def evaluate_compressor_unchunked(
    compress_fn: compressor.Compressor,
    get_data_generator_fn: Callable[[], Generator[bytes, None, None]],
    num_chunks: int,
) -> tuple[float, float]:
  """Evaluates the compressor on the unchunked dataset.

  Args:
    compress_fn: The function that compresses data.
    get_data_generator_fn: The function that creates a data generator.
    num_chunks: The number of chunks to consider.

  Returns:
    The compression rate and the total running time.
  """
  all_data = bytearray()
  for data in tqdm.tqdm(get_data_generator_fn(), desc='Compressing data, unchunked', total=num_chunks):
    all_data += data
  all_data = bytes(all_data)
  t0 = time.perf_counter()
  compressed_data = compress_fn(all_data)
  t1 = time.perf_counter()
  return len(compressed_data) / len(all_data), t1 - t0


def main(_) -> None:
  # log the command line arguments, only logging certain arguments for certain compressors
  logging.info('Compressor: %s', _COMPRESSOR.value)
  logging.info('Dataset: %s', _DATASET.value)
  logging.info('Num chunks: %s', _NUM_CHUNKS.value)
  if _COMPRESSOR.value == 'flac':
    assert _BIT_DEPTH.value in constants_audio.VALID_BIT_DEPTHS, f"Invalid bit depth: {_BIT_DEPTH.value}. Valid bit depths are {constants_audio.VALID_BIT_DEPTHS}."
    logging.info('Bit depth: %s', _BIT_DEPTH.value)
  if _COMPRESSOR.value == 'flac':
    assert _SAMPLE_RATE.value > 0, f"Sample rate must be greater than 0. Provided sample rate: {_SAMPLE_RATE.value}."
    logging.info('Sample rate: %s', _SAMPLE_RATE.value)
  if _COMPRESSOR.value == 'llama': # if the compressor is llama, we need to check if the llama model is valid
    assert _LLAMA_MODEL.value in constants_audio.VALID_LLAMA_MODELS, f"Invalid llama model: {_LLAMA_MODEL.value}. Valid llama models are {constants_audio.VALID_LLAMA_MODELS}."
    logging.info('Llama model: %s', _LLAMA_MODEL.value)

  # get the compress function and data generator function
  compress_fn_dict = compressor.get_compress_fn_dict( # get compress function dictionary
    bit_depth=_BIT_DEPTH.value,
    sample_rate=_SAMPLE_RATE.value,
    llama_model=_LLAMA_MODEL.value,
  )
  compress_fn = compress_fn_dict[_COMPRESSOR.value]
  get_data_generator_fn = functools.partial(
      data_loaders_audio.GET_AUDIO_DATA_GENERATOR_FN_DICT[_DATASET.value],
      num_chunks=_NUM_CHUNKS.value,
      bit_depth=_BIT_DEPTH.value,
  )

  # for classical compressors, we evaluate the compressor on the unchunked and chunked data
  if _COMPRESSOR.value in compressor.COMPRESSOR_TYPES['classical']:
    unchunked_rate, unchunked_time = evaluate_compressor_unchunked(
        compress_fn=compress_fn,
        get_data_generator_fn=get_data_generator_fn,
        num_chunks=_NUM_CHUNKS.value,
    )
    chunked_rate, chunked_time = evaluate_compressor_chunked(
        compress_fn=compress_fn,
        get_data_generator_fn=get_data_generator_fn,
        num_chunks=_NUM_CHUNKS.value,
        count_header_only_once=True,
        mask_fn=None,
    )
    logging.info('Unchunked: %.1f [%.1fs]', 100 * unchunked_rate, unchunked_time)
    logging.info('Chunked: %.1f [%.1fs]', 100 * chunked_rate, chunked_time)

  # for arithmetic coding compressors, we evaluate the compressor on only the chunked data
  elif _COMPRESSOR.value in compressor.COMPRESSOR_TYPES['arithmetic_coding']:
    chunked_rate, chunked_time = evaluate_compressor_chunked(
        compress_fn=compress_fn,
        get_data_generator_fn=get_data_generator_fn,
        num_chunks=_NUM_CHUNKS.value,
        count_header_only_once=False,
        mask_fn=utils.right_shift_bytes_by_one,
    )
    logging.info('Chunked: %.1f [%.1fs]', 100 * chunked_rate, chunked_time)

  # unknown compressor
  else:
    raise ValueError(f"Unknown compressor: {_COMPRESSOR.value}. For classical compressors, use one of {compressor.COMPRESSOR_TYPES['classical']}. For arithmetic coding compressors, use one of {compressor.COMPRESSOR_TYPES['arithmetic_coding']}.")

if __name__ == '__main__':
  app.run(main)
