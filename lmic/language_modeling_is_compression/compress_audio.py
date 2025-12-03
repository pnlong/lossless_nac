"""Evaluates a compressor, designed for audio data."""

from collections.abc import Generator
import functools
import time
from typing import Callable
import inspect
import pandas as pd
from os.path import exists

import argparse
from absl import logging
import tqdm

from language_modeling_is_compression import constants
from language_modeling_is_compression import constants_audio
from language_modeling_is_compression import data_loaders_audio
from language_modeling_is_compression.compressors_audio import compressor


def parse_args():
  """Parse command line arguments."""
  # parse arguments
  parser = argparse.ArgumentParser(description='Evaluates a compressor, designed for audio data.')
  parser.add_argument(
      '--output_filepath',
      type=str,
      default=constants_audio.EVAL_OUTPUT_FILEPATH,
      help='Output filepath (CSV file).',
  )
  parser.add_argument(
      '--loss_bpb_output_filepath',
      type=str,
      default=constants_audio.LOSS_BPB_OUTPUT_FILEPATH,
      help='Output filepath (CSV file) for loss and bits per byte. Only necessary for arithmetic coding compressors.',
  )
  parser.add_argument(
      '--compressor',
      type=str,
      default='flac',
      choices=compressor.COMPRESSOR_TYPES['classical'] + compressor.COMPRESSOR_TYPES['arithmetic_coding'],
      help='Compressor to use.',
  )
  parser.add_argument(
      '--dataset',
      type=str,
      default='musdb18mono',
      choices=list(data_loaders_audio.GET_AUDIO_DATA_GENERATOR_FN_DICT.keys()),
      help='Dataset to use.',
  )
  parser.add_argument(
      '--chunk_size',
      type=int,
      default=constants.CHUNK_SIZE_BYTES,
      help='Chunk size (number of bytes).',
  )
  parser.add_argument(
      '--num_chunks',
      type=int,
      default=constants.NUM_CHUNKS,
      help='Number of chunks.',
  )
  parser.add_argument(
      '--bit_depth',
      type=int,
      default=None,
      help='Bit depth (8, 16, or 24). Default is None, which means the bit depth is determined by the dataset.',
  )
  parser.add_argument(
      '--is_mu_law',
      type=bool,
      default=None,
      help='Whether to use mu-law encoding. Default is None, which means the is_mu_law is determined by the dataset.',
  )
  parser.add_argument(
      '--sample_rate',
      type=int,
      default=constants_audio.SAMPLE_RATE,
      help='Sample rate (Hz). Only necessary for FLAC compression.',
  )
  parser.add_argument(
      '--use_slow_lossless_compression',
      action='store_true',
      help='Whether to use slow lossless compression.',
  )
  args = parser.parse_args()

  # set default arguments if not provided
  if args.bit_depth is None:
    native_bit_depth = data_loaders_audio.get_native_bit_depth(dataset=args.dataset)
    logging.info(f"Bit depth is not set. Using native bit depth {native_bit_depth} for dataset {args.dataset}.")
    args.bit_depth = native_bit_depth
  if args.is_mu_law is None:
    native_is_mu_law = data_loaders_audio.get_is_mu_law(dataset=args.dataset)
    logging.info(f"is_mu_law is not set. Using native is_mu_law {native_is_mu_law} for dataset {args.dataset}.")
    args.is_mu_law = native_is_mu_law
  
  return args


def evaluate_compressor_chunked(
    compress_fn: compressor.Compressor,
    get_data_generator_fn: Callable[[], Generator[bytes, None, None]],
    num_chunks: int,
    count_header_only_once: bool = True,
    use_tqdm: bool = constants_audio.USE_TQDM,
    bit_depth: int = constants_audio.BIT_DEPTH,
    use_slow_lossless_compression: bool = constants_audio.USE_SLOW_LOSSLESS_COMPRESSION_FOR_EVALS,
    return_loss_and_bpb: bool = False,
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
    use_slow_lossless_compression: Whether to use slow lossless compression.
    return_loss_and_bpb: Whether to return the loss and bits per byte for the compressed data.

  Returns:
    The compression rate and the total running time.
  """
  running_time = raw_length = compressed_length = 0

  data_generator = get_data_generator_fn()
  if use_tqdm:
    data_generator = tqdm.tqdm(data_generator, desc='Compressing data, chunked', total=num_chunks, miniters=max(num_chunks // 100, 1), maxinterval=3600)

  compressed_losses = [float('inf')] * num_chunks
  compressed_bpbs = [float('inf')] * num_chunks
  for i, data in enumerate(data_generator):

    t0 = time.perf_counter()
    kwargs = dict()
    has_use_slow_lossless_compression_keyword_argument = 'use_slow_lossless_compression' in inspect.signature(compress_fn).parameters.keys()
    if has_use_slow_lossless_compression_keyword_argument:
      kwargs['use_slow_lossless_compression'] = use_slow_lossless_compression
    has_return_loss_and_bpb_keyword_argument = 'return_loss_and_bpb' in inspect.signature(compress_fn).parameters.keys()
    if has_return_loss_and_bpb_keyword_argument:
      kwargs['return_loss_and_bpb'] = return_loss_and_bpb
    if has_return_loss_and_bpb_keyword_argument and return_loss_and_bpb: # return loss and bits per byte for the compressed data
      compressed_data, compressed_loss, compressed_bpb = compress_fn(data, **kwargs)
      compressed_losses[i] = compressed_loss
      compressed_bpbs[i] = compressed_bpb
    else: # don't return loss and bits per byte for the compressed data, either because the compress function does not have a return_loss_and_bpb keyword argument or because return_loss_and_bpb is False
      compressed_data = compress_fn(data, **kwargs)
    t1 = time.perf_counter()

    running_time += t1 - t0
    raw_length += len(data)
    compressed_length += len(compressed_data)

  # Convert loss and bpb into pandas DataFrame
  loss_and_bpb_df = pd.DataFrame(data = {
    'i': list(range(num_chunks)),
    'loss': compressed_losses,
    'bpb': compressed_bpbs,
  })

  # We only count the header once for classical compressors.
  if count_header_only_once:
    minimal_sample = (0).to_bytes(bit_depth // 8, byteorder='little', signed=False) # create a minimal valid sample based on bit depth (at least one sample worth of bytes)
    header_length = len(compress_fn(minimal_sample))
    compressed_length -= header_length * (num_chunks - 1)

  # Calculate compression rate
  compression_rate = compressed_length / raw_length

  # Return specified output
  if return_loss_and_bpb:
    return compression_rate, running_time, loss_and_bpb_df
  else:
    return compression_rate, running_time


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


def main(args) -> None:
  # log the command line arguments, only logging certain arguments for certain compressors
  logging.info('Compressor: %s', args.compressor)
  logging.info('Dataset: %s', args.dataset)
  assert args.chunk_size > 0, f"Chunk size must be greater than 0. Provided chunk size: {args.chunk_size}."
  logging.info('Chunk size: %s', args.chunk_size)
  assert args.num_chunks > 0, f"Number of chunks must be greater than 0. Provided number of chunks: {args.num_chunks}."
  logging.info('Num chunks: %s', args.num_chunks)
  assert args.bit_depth in constants_audio.VALID_BIT_DEPTHS, f"Invalid bit depth: {args.bit_depth}. Valid bit depths are {constants_audio.VALID_BIT_DEPTHS}."
  logging.info('Bit depth: %s', args.bit_depth)
  assert args.is_mu_law is not None, f"is_mu_law is not set. Provided is_mu_law: {args.is_mu_law}."
  logging.info('Is mu-law: %s', args.is_mu_law)
  if args.compressor == 'flac':
    assert args.sample_rate > 0, f"Sample rate must be greater than 0. Provided sample rate: {args.sample_rate}."
    logging.info('Sample rate: %s', args.sample_rate)
  if args.use_slow_lossless_compression:
    logging.info('Using slow lossless compression.')
  
  # get the compress function and data generator function
  compress_fn_dict = compressor.get_compress_fn_dict( # get compress function dictionary
    bit_depth=args.bit_depth,
    sample_rate=args.sample_rate,
  )
  compress_fn = compress_fn_dict[args.compressor]
  get_data_generator_fn = functools.partial(
      data_loaders_audio.GET_AUDIO_DATA_GENERATOR_FN_DICT[args.dataset],
      chunk_size=args.chunk_size,
      num_chunks=args.num_chunks,
      bit_depth=args.bit_depth,
      is_mu_law=args.is_mu_law,
  )

  # for classical compressors, we evaluate the compressor on the unchunked and chunked data
  if args.compressor in compressor.COMPRESSOR_TYPES['classical']:
    unchunked_rate, unchunked_time = evaluate_compressor_unchunked(
        compress_fn=compress_fn,
        get_data_generator_fn=get_data_generator_fn,
        num_chunks=args.num_chunks,
    )
    chunked_rate, chunked_time = evaluate_compressor_chunked(
        compress_fn=compress_fn,
        get_data_generator_fn=get_data_generator_fn,
        num_chunks=args.num_chunks,
        count_header_only_once=True,
        bit_depth=args.bit_depth,
    )
    logging.info('Unchunked: %.1f (%.1fx) [%.1fs]', 100 * unchunked_rate, 1 / unchunked_rate, unchunked_time)
    logging.info('Chunked: %.1f (%.1fx) [%.1fs]', 100 * chunked_rate, 1 / chunked_rate, chunked_time)

  # for arithmetic coding compressors, we evaluate the compressor on only the chunked data
  elif args.compressor in compressor.COMPRESSOR_TYPES['arithmetic_coding']:

    # introduce column names that describe the data being evaluated
    description_columns = [
      'compressor',
      'dataset',
      'chunk_size',
      'num_chunks',
      'bit_depth',
      'is_native_bit_depth',
      'is_mu_law',
      'matches_native_quantization',
      'use_slow_lossless_compression',
    ]

    # calculate compression rate and loss and bits per byte
    chunked_rate, chunked_time, loss_and_bpb_df = evaluate_compressor_chunked(
        compress_fn=compress_fn,
        get_data_generator_fn=get_data_generator_fn,
        num_chunks=args.num_chunks,
        count_header_only_once=False,
        bit_depth=args.bit_depth,
        use_slow_lossless_compression=args.use_slow_lossless_compression,
        return_loss_and_bpb=True,
    )
    logging.info('Chunked: %.1f (%.1fx) [%.1fs]', 100 * chunked_rate, 1 / chunked_rate, chunked_time)
    is_native_bit_depth = args.bit_depth == data_loaders_audio.get_native_bit_depth(dataset=args.dataset)
    matches_native_quantization = args.is_mu_law == data_loaders_audio.get_is_mu_law(dataset=args.dataset)

    # deal with output data frame for compression data (i.e., compression rate and compression time)
    if not exists(args.output_filepath): # write column names to output file if it doesn't exist
      pd.DataFrame(
        columns = description_columns + [
          'compression_rate',
          'compression_time',
        ]).to_csv(
          path_or_buf=args.output_filepath,
          sep=',',
          na_rep='NA',
          header=True,
          index=False,
          mode='w',
        )
    compression_df = pd.DataFrame(data = [{
      'compressor': args.compressor,
      'dataset': args.dataset,
      'chunk_size': args.chunk_size,
      'num_chunks': args.num_chunks,
      'bit_depth': args.bit_depth,
      'is_native_bit_depth': is_native_bit_depth,
      'is_mu_law': args.is_mu_law,
      'matches_native_quantization': matches_native_quantization,
      'use_slow_lossless_compression': args.use_slow_lossless_compression,
      'compression_rate': 1 / chunked_rate, # > 1 indicates compression, < 1 indicates expansion
      'compression_time': chunked_time,
    }])
    compression_df = compression_df[description_columns + ['compression_rate', 'compression_time']] # reorder columns
    compression_df.to_csv(
      path_or_buf=args.output_filepath,
      sep=',',
      na_rep='NA',
      header=False,
      index=False,
      mode='a',
    )

    # deal with loss and bits per byte data frame
    loss_and_bpb_columns = loss_and_bpb_df.columns.tolist()
    if not exists(args.loss_bpb_output_filepath): # write column names to output file if it doesn't exist
      pd.DataFrame(
        columns = description_columns + loss_and_bpb_columns).to_csv(
          path_or_buf=args.loss_bpb_output_filepath,
          sep=',',
          na_rep='NA',
          header=True,
          index=False,
          mode='w',
        )
    rep_for_loss_and_bpb_df = lambda x: [x] * len(loss_and_bpb_df) # helper function to repeat a value for the length of the loss and bits per byte data frame
    loss_and_bpb_df = loss_and_bpb_df.assign(**{
      'compressor': rep_for_loss_and_bpb_df(x=args.compressor),
      'dataset': rep_for_loss_and_bpb_df(x=args.dataset),
      'chunk_size': rep_for_loss_and_bpb_df(x=args.chunk_size),
      'num_chunks': rep_for_loss_and_bpb_df(x=args.num_chunks),
      'bit_depth': rep_for_loss_and_bpb_df(x=args.bit_depth),
      'is_native_bit_depth': rep_for_loss_and_bpb_df(x=is_native_bit_depth),
      'is_mu_law': rep_for_loss_and_bpb_df(x=args.is_mu_law),
      'matches_native_quantization': rep_for_loss_and_bpb_df(x=matches_native_quantization),
      'use_slow_lossless_compression': rep_for_loss_and_bpb_df(x=args.use_slow_lossless_compression),
    })
    loss_and_bpb_df = loss_and_bpb_df[description_columns + loss_and_bpb_columns] # reorder columns    
    loss_and_bpb_df = loss_and_bpb_df.drop(columns=['num_chunks']) # num chunks is irrelevant for loss and bits per byte data
    loss_and_bpb_df.to_csv(
      path_or_buf=args.loss_bpb_output_filepath,
      sep=',',
      na_rep='NA',
      header=False,
      index=False,
      mode='a',
    )

  # unknown compressor
  else:
    raise ValueError(f"Unknown compressor: {args.compressor}. For classical compressors, use one of {compressor.COMPRESSOR_TYPES['classical']}. For arithmetic coding compressors, use one of {compressor.COMPRESSOR_TYPES['arithmetic_coding']}.")

if __name__ == '__main__':
  # Initialize absl logging to ensure INFO messages are displayed
  logging.use_absl_handler()
  logging.set_verbosity(logging.INFO)
  
  args = parse_args()
  main(args)
