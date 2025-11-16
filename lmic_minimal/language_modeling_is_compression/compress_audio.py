"""Evaluates a compressor, designed for audio data."""

from collections.abc import Generator
import functools
import time
from typing import Callable
import inspect
import pandas as pd
from os.path import exists

from absl import app
from absl import flags
from absl import logging
import tqdm

from language_modeling_is_compression import constants
from language_modeling_is_compression import constants_audio
from language_modeling_is_compression import data_loaders_audio
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
_CHUNK_SIZE = flags.DEFINE_integer(
    'chunk_size',
    constants.CHUNK_SIZE_BYTES,
    'Chunk size (number of bytes).',
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


def evaluate_compressor_chunked(
    compress_fn: compressor.Compressor,
    get_data_generator_fn: Callable[[], Generator[bytes, None, None]],
    num_chunks: int,
    count_header_only_once: bool = True,
    use_tqdm: bool = constants_audio.USE_TQDM,
    bit_depth: int = constants_audio.BIT_DEPTH,
) -> tuple[float, float]:
  """Evaluates the compressor on the chunked dataset.

  Args:
    compress_fn: The function that evaluates data.
    get_data_generator_fn: The function that creates a data generator.
    num_chunks: The number of chunks to consider
    count_header_only_once: Whether to count the header as part of the
      compressed output only once for the whole dataset or for every chunk
    use_tqdm: Whether to use a progress bar or not.
    bit_depth: Bit depth of the audio data (used to create minimal valid sample for header calculation).

  Returns:
    The compression rate and the total running time.
  """
  running_time = raw_length = compressed_length = 0

  data_generator = get_data_generator_fn()
  if use_tqdm:
    data_generator = tqdm.tqdm(data_generator, desc='Compressing data, chunked', total=num_chunks, miniters=max(num_chunks // 100, 1), maxinterval=3600)

  for data in data_generator:

    t0 = time.perf_counter()
    if 'use_slow_lossless_compression' in inspect.signature(compress_fn).parameters.keys(): # if the compress function has a use_slow_lossless_compression keyword argument, pass it
      compressed_data = compress_fn(data, use_slow_lossless_compression=constants_audio.USE_SLOW_LOSSLESS_COMPRESSION_FOR_EVALS)
    else:
      compressed_data = compress_fn(data)
    t1 = time.perf_counter()

    running_time += t1 - t0
    raw_length += len(data)
    compressed_length += len(compressed_data)

  # We only count the header once for classical compressors.
  if count_header_only_once:
    minimal_sample = (0).to_bytes(bit_depth // 8, byteorder='little', signed=False) # create a minimal valid sample based on bit depth (at least one sample worth of bytes)
    header_length = len(compress_fn(minimal_sample))
    compressed_length -= header_length * (num_chunks - 1)

  return compressed_length / raw_length, running_time


def evaluate_compressor_unchunked(
    compress_fn: compressor.Compressor,
    get_data_generator_fn: Callable[[], Generator[bytes, None, None]],
    num_chunks: int,
    use_tqdm: bool = constants_audio.USE_TQDM,
) -> tuple[float, float]:
  """Evaluates the compressor on the unchunked dataset.

  Args:
    compress_fn: The function that compresses data.
    get_data_generator_fn: The function that creates a data generator.
    num_chunks: The number of chunks to consider.
    use_tqdm: Whether to use a progress bar or not.

  Returns:
    The compression rate and the total running time.
  """
  all_data = bytearray()
  data_generator = get_data_generator_fn()
  if use_tqdm:
    data_generator = tqdm.tqdm(data_generator, desc='Compressing data, unchunked', total=num_chunks, miniters=max(num_chunks // 100, 1), maxinterval=3600)
  for data in data_generator:
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
  assert _CHUNK_SIZE.value > 0, f"Chunk size must be greater than 0. Provided chunk size: {_CHUNK_SIZE.value}."
  logging.info('Chunk size: %s', _CHUNK_SIZE.value)
  assert _NUM_CHUNKS.value > 0, f"Number of chunks must be greater than 0. Provided number of chunks: {_NUM_CHUNKS.value}."
  logging.info('Num chunks: %s', _NUM_CHUNKS.value)
  if _COMPRESSOR.value == 'flac':
    assert _BIT_DEPTH.value in constants_audio.VALID_BIT_DEPTHS, f"Invalid bit depth: {_BIT_DEPTH.value}. Valid bit depths are {constants_audio.VALID_BIT_DEPTHS}."
    logging.info('Bit depth: %s', _BIT_DEPTH.value)
  if _COMPRESSOR.value == 'flac':
    assert _SAMPLE_RATE.value > 0, f"Sample rate must be greater than 0. Provided sample rate: {_SAMPLE_RATE.value}."
    logging.info('Sample rate: %s', _SAMPLE_RATE.value)
  
  # get the compress function and data generator function
  compress_fn_dict = compressor.get_compress_fn_dict( # get compress function dictionary
    bit_depth=_BIT_DEPTH.value,
    sample_rate=_SAMPLE_RATE.value,
  )
  compress_fn = compress_fn_dict[_COMPRESSOR.value]
  get_data_generator_fn = functools.partial(
      data_loaders_audio.GET_AUDIO_DATA_GENERATOR_FN_DICT[_DATASET.value],
      chunk_size=_CHUNK_SIZE.value,
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
        bit_depth=_BIT_DEPTH.value,
    )
    logging.info('Unchunked: %.1f (%.1fx) [%.1fs]', 100 * unchunked_rate, 1 / unchunked_rate, unchunked_time)
    logging.info('Chunked: %.1f (%.1fx) [%.1fs]', 100 * chunked_rate, 1 / chunked_rate, chunked_time)

  # for arithmetic coding compressors, we evaluate the compressor on only the chunked data
  elif _COMPRESSOR.value in compressor.COMPRESSOR_TYPES['arithmetic_coding']:
    if not exists(constants_audio.LLAMA_EVAL_OUTPUT_FILEPATH): # write column names to output file if it doesn't exist
      pd.DataFrame(
        columns = [
          'compressor', 'dataset', 'chunk_size', 'num_chunks', 'bit_depth', 'compression_rate', 'compression_time'
        ]).to_csv(
          path_or_buf=constants_audio.LLAMA_EVAL_OUTPUT_FILEPATH,
          sep=',',
          na_rep='NA',
          header=True,
          index=False,
          mode='w',
        )
    chunked_rate, chunked_time = evaluate_compressor_chunked(
        compress_fn=compress_fn,
        get_data_generator_fn=get_data_generator_fn,
        num_chunks=_NUM_CHUNKS.value,
        count_header_only_once=False,
        bit_depth=_BIT_DEPTH.value,
    )
    logging.info('Chunked: %.1f (%.1fx) [%.1fs]', 100 * chunked_rate, 1 / chunked_rate, chunked_time)
    pd.DataFrame(data = [{
      'compressor': _COMPRESSOR.value,
      'dataset': _DATASET.value,
      'chunk_size': _CHUNK_SIZE.value,
      'num_chunks': _NUM_CHUNKS.value,
      'bit_depth': _BIT_DEPTH.value,
      'compression_rate': 1 / chunked_rate,
      'compression_time': chunked_time,
    }]).to_csv(
      path_or_buf=constants_audio.LLAMA_EVAL_OUTPUT_FILEPATH,
      sep=',',
      na_rep='NA',
      header=False,
      index=False,
      mode='a',
    )

  # unknown compressor
  else:
    raise ValueError(f"Unknown compressor: {_COMPRESSOR.value}. For classical compressors, use one of {compressor.COMPRESSOR_TYPES['classical']}. For arithmetic coding compressors, use one of {compressor.COMPRESSOR_TYPES['arithmetic_coding']}.")

if __name__ == '__main__':
  app.run(main)
