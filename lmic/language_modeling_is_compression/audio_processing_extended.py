"""Extended audio processing functions for multiple bit depths."""

import numpy as np
from typing import Iterator
from . import ascii_mapping


def convert_to_target_bit_depth_extended(audio_data: np.ndarray, bit_depth: int) -> bytes:
  """Convert audio data to target bit depth and return as bytes.
  
  Args:
    audio_data: Audio data as 1D numpy array (float values in [-1, 1])
    bit_depth: Target bit depth (8, 16, 24, or 32)
    
  Returns:
    Audio data as bytes in the target bit depth
  """
  if bit_depth == 8:
    # Convert to 8-bit unsigned integers
    # Map [-1, 1] to [0, 255]
    audio_8bit = ((audio_data + 1.0) * 127.5).astype(np.uint8)
    return audio_8bit.tobytes()
  elif bit_depth == 16:
    # Convert to 16-bit signed integers
    # Map [-1, 1] to [-32768, 32767]
    audio_16bit = (audio_data * 32767).astype(np.int16)
    return audio_16bit.tobytes()
  elif bit_depth == 24:
    # Convert to 24-bit signed integers
    # Map [-1, 1] to [-8388608, 8388607]
    audio_24bit = (audio_data * 8388607).astype(np.int32)
    # Convert to 24-bit bytes (3 bytes per sample)
    audio_bytes = []
    for sample in audio_24bit:
      # Convert to 24-bit signed representation
      if sample < 0:
        sample = sample + (1 << 24)  # Convert to unsigned 24-bit
      # Convert numpy int32 to Python int, then to bytes
      sample_int = int(sample)
      audio_bytes.extend(sample_int.to_bytes(3, byteorder='little'))
    return bytes(audio_bytes)
  elif bit_depth == 32:
    # Convert to 32-bit signed integers
    # Map [-1, 1] to [-2147483648, 2147483647]
    audio_32bit = (audio_data * 2147483647).astype(np.int32)
    return audio_32bit.tobytes()
  else:
    raise ValueError(f"Unsupported bit depth: {bit_depth}")


def extract_audio_chunks_extended(audio_bytes: bytes, chunk_size: int, bit_depth: int) -> Iterator[bytes]:
  """Extract audio chunks with proper alignment for different bit depths.
  
  Args:
    audio_bytes: Audio data as bytes
    chunk_size: Size of each chunk in bytes
    bit_depth: Bit depth of the audio data
    
  Yields:
    Audio chunks as bytes
  """
  bytes_per_sample = ascii_mapping.calculate_bytes_per_sample(bit_depth)
  
  # Ensure chunk_size is aligned to sample boundaries
  aligned_chunk_size = (chunk_size // bytes_per_sample) * bytes_per_sample
  
  for i in range(0, len(audio_bytes), aligned_chunk_size):
    chunk = audio_bytes[i:i + aligned_chunk_size]
    if len(chunk) == aligned_chunk_size:
      yield chunk


def get_custom_audio_iterator_extended(
    audio_files: list[str],
    num_chunks: int = 1000,
    bit_depth: int = 8,
    blocking_size: int = 1024,
    chunk_size_bytes: int = 2048,
) -> Iterator[bytes]:
  """Extended version of get_custom_audio_iterator with bit depth support.
  
  Args:
    audio_files: List of audio file paths
    num_chunks: Maximum number of chunks to process
    bit_depth: Bit depth for audio processing (8, 16, 24, or 32)
    blocking_size: Size of blocks for stereo processing in samples
    chunk_size_bytes: Size of each chunk in bytes
    
  Yields:
    Audio chunks as bytes
  """
  from scipy.io import wavfile
  import random
  
  chunk_count = 0
  random.shuffle(audio_files)
  
  for audio_file in audio_files:
    if chunk_count >= num_chunks:
      break
      
    try:
      # Load audio file using scipy
      sr, audio_data = wavfile.read(audio_file)
      
      # Convert to mono if stereo
      if len(audio_data.shape) > 1:
        audio_data = audio_data.mean(axis=1)
      
      # Normalize to [-1, 1] range
      if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / 32767.0
      elif audio_data.dtype == np.int32:
        audio_data = audio_data.astype(np.float32) / 2147483647.0
      elif audio_data.dtype == np.uint8:
        audio_data = (audio_data.astype(np.float32) - 128.0) / 128.0
      else:
        # Assume already in float format
        audio_data = audio_data.astype(np.float32)
      
      # Convert to target bit depth
      audio_bytes = convert_to_target_bit_depth_extended(audio_data, bit_depth)
      
      # Extract chunks with proper alignment
      for chunk in extract_audio_chunks_extended(audio_bytes, chunk_size_bytes, bit_depth):
        if chunk_count >= num_chunks:
          break
        yield chunk
        chunk_count += 1
        
    except Exception as e:
      print(f"Error processing {audio_file}: {e}")
      continue
