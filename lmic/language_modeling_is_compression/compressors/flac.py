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

"""Implements a lossless compressor with FLAC."""

import audioop
import io
import numpy as np

import pydub


def compress(data: bytes, bit_depth: int = 16) -> bytes:
  """Returns data compressed with the FLAC codec.

  Args:
    data: Audio data bytes
    bit_depth: Bit depth of the audio data (8, 16, 24, or 32)
  """
  if bit_depth == 8:
    sample_width = 1
  elif bit_depth == 16:
    sample_width = 2
  elif bit_depth == 24:
    raise NotImplementedError("24-bit audio not supported by FLAC compressor")
  elif bit_depth == 32:
    raise NotImplementedError("32-bit audio not supported by FLAC compressor")
  else:
    raise ValueError(f"Unsupported bit depth: {bit_depth}")
  
  # Ensure data length is aligned to sample width
  if len(data) == 0:
    return b""
  
  remainder = len(data) % sample_width
  if remainder != 0:
    # Truncate trailing bytes to enforce alignment
    data = data[:len(data) - remainder]
  
  sample = pydub.AudioSegment(
      data=data,
      channels=1,
      sample_width=sample_width,
      frame_rate=16000,
  )
  return sample.export(
      format='flac',
      parameters=['-compression_level', '12'],
  ).read()


def decompress(data: bytes, bit_depth: int = 16) -> bytes:
  """Decompresses `data` losslessly using the FLAC codec.

  Args:
    data: The data to be decompressed
    bit_depth: Bit depth of the original audio data (8, 16, 24, or 32)

  Returns:
    The decompressed data in the original bit depth format
  """
  if bit_depth == 24:
    raise NotImplementedError("24-bit audio not supported by FLAC decompressor")
  elif bit_depth == 32:
    raise NotImplementedError("32-bit audio not supported by FLAC decompressor")
  elif bit_depth not in [8, 16]:
    raise ValueError(f"Unsupported bit depth: {bit_depth}")
  
  sample = pydub.AudioSegment.from_file(io.BytesIO(data), format='flac')
  
  if bit_depth == 8:
    # Convert from 16-bit to 8-bit: FLAC outputs 16-bit, but original was 8-bit
    # Convert samples from 16 bit to 8 bit and add 128 since 16 bit is signed
    return audioop.bias(audioop.lin2lin(sample.raw_data, 2, 1), 1, 128)
  elif bit_depth == 16:
    # FLAC outputs 16-bit, which matches our original format
    return sample.raw_data
