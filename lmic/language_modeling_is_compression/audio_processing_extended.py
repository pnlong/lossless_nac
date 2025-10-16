"""Extended audio processing functions for multiple bit depths."""

import logging
import os
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


def process_stereo_blocking_extended(audio_data: np.ndarray, blocking_size: int) -> np.ndarray:
  """Process stereo audio using interleaved blocking strategy.
  
  Args:
    audio_data: Audio data with shape (samples, channels) from scipy.wavfile
    blocking_size: Size of each block in samples
    
  Returns:
    Processed audio data as 1D array with interleaved blocks
    Pattern: [L_block1, R_block1, L_block2, R_block2, ...]
    This doubles the effective sequence length compared to averaging.
  """
  if len(audio_data.shape) == 1:
    # Mono audio - return as is
    return audio_data
  
  elif len(audio_data.shape) == 2:
    # Stereo audio - apply blocking strategy
    samples, channels = audio_data.shape
    
    if channels == 2:
      left_channel = audio_data[:, 0]  # Shape: (samples,)
      right_channel = audio_data[:, 1]  # Shape: (samples,)
      
      # Split channels into blocks
      left_blocks = []
      right_blocks = []
      
      for i in range(0, samples, blocking_size):
        left_block = left_channel[i:i + blocking_size]
        right_block = right_channel[i:i + blocking_size]
        
        # Add all blocks, including the final partial block
        left_blocks.append(left_block)
        right_blocks.append(right_block)
      
      # Interleave blocks: [L_block1, R_block1, L_block2, R_block2, ...]
      interleaved_blocks = []
      for left_block, right_block in zip(left_blocks, right_blocks):
        interleaved_blocks.append(left_block)
        interleaved_blocks.append(right_block)
      
      # Concatenate all blocks
      return np.concatenate(interleaved_blocks)
    
    else:
      raise ValueError(f"Unsupported number of channels: {channels}")
  
  else:
    raise ValueError(f"Unsupported audio data shape: {audio_data.shape}")


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
  
  for file_index, audio_file in enumerate(audio_files):
    if chunk_count >= num_chunks:
      break
      
    try:
      # Log when a new file is loaded
      logging.debug(f"Loading audio file #{file_index + 1}: {os.path.basename(audio_file)}")
      
      # Load audio file using scipy
      sr, audio_data = wavfile.read(audio_file)
      
      # Handle stereo audio with interleaved blocking (doubles sequence length)
      if len(audio_data.shape) > 1:
        # Use interleaved blocking for stereo: [L_block1, R_block1, L_block2, R_block2, ...]
        # This doubles the effective sequence length compared to averaging
        audio_data = process_stereo_blocking_extended(audio_data, blocking_size)
      else:
        # Mono audio - keep as is
        audio_data = audio_data
      
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
      
      # Extract chunks with proper alignment - use random sampling
      # First, collect all possible chunks
      all_chunks = list(extract_audio_chunks_extended(audio_bytes, chunk_size_bytes, bit_depth))
      total_possible_chunks = len(all_chunks)
      
      if total_possible_chunks == 0:
        continue
      
      # Calculate how many chunks to take from this file (limit to 10 per file)
      remaining_chunks_needed = num_chunks - chunk_count
      chunks_to_take = min(remaining_chunks_needed, 10, total_possible_chunks)  # Max 10 chunks per file
      
      # Randomly sample chunks instead of taking first N
      import random
      selected_chunk_indices = random.sample(range(total_possible_chunks), chunks_to_take)
      selected_chunk_indices.sort()  # Sort for consistent ordering
      
      # Yield the randomly selected chunks
      logging.debug(f"Yielding {len(selected_chunk_indices)} chunks from {os.path.basename(audio_file)}")
      for local_chunk_index, chunk_index in enumerate(selected_chunk_indices):
        if chunk_count >= num_chunks:
          break
        
        chunk = all_chunks[chunk_index]
        logging.debug(f"Yielding chunk {local_chunk_index + 1}/{len(selected_chunk_indices)} from {os.path.basename(audio_file)} (global chunk #{chunk_count + 1})")
        yield chunk
        chunk_count += 1
      
      logging.debug(f"Finished processing {os.path.basename(audio_file)} - yielded {len(selected_chunk_indices)} chunks")
        
    except Exception as e:
      logging.warning(f"Error processing {audio_file}: {e}")
      continue
