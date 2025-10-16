# Copyright 2024 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Defines the compressor interface."""

import functools
import gzip
import lzma
from typing import Mapping, Protocol

from language_modeling_is_compression.compressors import flac
from language_modeling_is_compression.compressors import language_model
from language_modeling_is_compression.compressors import png


class Compressor(Protocol):

  def __call__(self, data: bytes, *args, **kwargs) -> bytes | tuple[bytes, int]:
    """Returns the compressed version of `data`, with optional padded bits."""


COMPRESSOR_TYPES = {
    'classical': ['flac', 'gzip', 'lzma', 'png'],
    'arithmetic_coding': ['language_model'],
}

# Create FLAC wrapper that can accept bit_depth parameter
def flac_compress_wrapper(data: bytes, bit_depth: int = 8) -> bytes:
    """Wrapper for FLAC compression that accepts bit_depth parameter."""
    return flac.compress(data, bit_depth=bit_depth)

COMPRESS_FN_DICT: Mapping[str, Compressor] = {
    'flac': flac_compress_wrapper,
    'gzip': functools.partial(gzip.compress, compresslevel=9),
    'language_model': language_model.compress,
    'lzma': lzma.compress,
    'png': png.compress,
}
