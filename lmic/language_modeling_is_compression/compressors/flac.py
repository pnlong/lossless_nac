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


def compress(data: bytes, bit_depth: int = 8) -> bytes:
  """Returns data compressed with the FLAC codec.

  Args:
    data: Audio data bytes
    bit_depth: Bit depth of the audio data (8, 16, 24, or 32)
  """
  # Determine sample width and frame rate based on bit depth
  if bit_depth == 8:
    sample_width = 1
    frame_rate = 16000
  elif bit_depth == 16:
    sample_width = 2
    frame_rate = 16000
  elif bit_depth == 24:
    # FLAC doesn't natively support 24-bit, so we'll treat it as 32-bit
    sample_width = 4
    frame_rate = 16000
    # Convert 24-bit to 32-bit by padding with zeros
    data_24bit = np.frombuffer(data, dtype=np.uint8)
    num_samples = len(data_24bit) // 3
    data_32bit = np.zeros(num_samples * 4, dtype=np.uint8)
    
    for i in range(num_samples):
      start_idx = i * 3
      end_idx = start_idx + 3
      if end_idx <= len(data_24bit):
        # Convert 3 bytes to 4 bytes (24-bit to 32-bit)
        sample_24bit = int.from_bytes(data_24bit[start_idx:end_idx], byteorder='little', signed=False)
        sample_32bit = sample_24bit << 8  # Left shift to make it 32-bit
        sample_32bit_bytes = sample_32bit.to_bytes(4, byteorder='little')
        data_32bit[i*4:(i*4)+4] = list(sample_32bit_bytes)
    
    data = data_32bit.tobytes()
  elif bit_depth == 32:
    sample_width = 4
    frame_rate = 16000
  else:
    raise ValueError(f"Unsupported bit depth: {bit_depth}")
  
  # Defensive alignment: ensure data length is a multiple of sample_width
  # Some upstream generators may yield a final chunk that is not perfectly
  # aligned. FLAC/pydub requires exact alignment.
  if len(data) == 0:
    return b""  # Nothing to compress

  remainder = len(data) % sample_width
  if remainder != 0:
    # Truncate trailing bytes to enforce alignment
    data = data[: len(data) - remainder]
  
  sample = pydub.AudioSegment(
      data=data,
      channels=1,
      sample_width=sample_width,
      frame_rate=frame_rate,
  )
  return sample.export(
      format='flac',
      parameters=['-compression_level', '12'],
  ).read()


def decompress(data: bytes) -> bytes:
  """Decompresses `data` losslessly using the FLAC codec.

  Args:
    data: The data to be decompressed. Assumes 2 bytes per sample (16 bit).

  Returns:
    The decompressed data. Assumes 1 byte per sample (8 bit).
  """
  sample = pydub.AudioSegment.from_file(io.BytesIO(data), format='flac')
  # FLAC assumes that data is 16 bit. However, since our original data is 8 bit,
  # we need to convert the samples from 16 bit to 8 bit (i.e., changing from two
  # channels to one channel with `lin2lin`) and add 128 since 16 bit is signed
  # (i.e., adding 128 using `bias`).
  return audioop.bias(audioop.lin2lin(sample.raw_data, 2, 1), 1, 128)
