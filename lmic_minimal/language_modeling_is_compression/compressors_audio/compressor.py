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
    'arithmetic_coding': ['language_model', 'llama'],
}

# COMPRESS_FN_DICT: Mapping[str, Compressor] = { # using a function instead
#     'flac': flac.compress,
#     'gzip': functools.partial(gzip.compress, compresslevel=9),
#     'language_model': language_model.compress,
#     'lzma': lzma.compress,
#     'png': png.compress,
# }
def get_compress_fn_dict(
    bit_depth: int = constants_audio.BIT_DEPTH,
    sample_rate: int = constants_audio.SAMPLE_RATE,
    llama_model: str = constants_audio.LLAMA_MODEL,
  ) -> Mapping[str, Compressor]:
  """Returns the compress function dictionary."""
  return {
    'flac': functools.partial(flac.compress, bit_depth=bit_depth, sample_rate=sample_rate),
    'gzip': functools.partial(gzip.compress, compresslevel=9),
    'language_model': language_model.compress,
    'lzma': lzma.compress,
    'png': png.compress,
    'llama': functools.partial(llama.compress, llama_model=llama_model),
  }
