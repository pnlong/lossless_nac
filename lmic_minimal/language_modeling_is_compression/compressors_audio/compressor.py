"""Defines the compressor interface for audio data."""

import functools
import gzip
import lzma
from typing import Mapping, Protocol

from language_modeling_is_compression import constants_audio
from language_modeling_is_compression.compressors_audio import flac
from language_modeling_is_compression.compressors_audio import language_model
from language_modeling_is_compression.compressors_audio import png
from language_modeling_is_compression.compressors_audio import llama


class Compressor(Protocol):

  def __call__(self, data: bytes, *args, **kwargs) -> bytes | tuple[bytes, int]:
    """Returns the compressed version of `data`, with optional padded bits."""


COMPRESSOR_TYPES = {
    'classical': ['flac', 'gzip', 'lzma', 'png'],
    'arithmetic_coding': ['language_model'] + constants_audio.VALID_LLAMA_MODELS,
}

def get_compress_fn_dict(
    bit_depth: int = constants_audio.BIT_DEPTH,
    sample_rate: int = constants_audio.SAMPLE_RATE,
  ) -> Mapping[str, Compressor]:
  """Returns the compress function dictionary."""
  compress_fn_dict = {
    'flac': functools.partial(flac.compress, bit_depth=bit_depth, sample_rate=sample_rate),
    'gzip': functools.partial(gzip.compress, compresslevel=9),
    'language_model': language_model.compress,
    'lzma': lzma.compress,
    'png': png.compress,
  }
  for llama_model in constants_audio.VALID_LLAMA_MODELS: # add llama models to the compress function dictionary
    compress_fn_dict[llama_model] = functools.partial(llama.compress, llama_model=llama_model)
  return compress_fn_dict
