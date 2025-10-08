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

"""Implements data loaders."""

import audioop
from collections.abc import Iterator
import itertools
import logging
import os.path
import random
import urllib.request
import warnings
import zipfile
from typing import List

import numpy as np
import tensorflow_datasets as tfds

from language_modeling_is_compression import constants


def _get_librispeech_dataset():
  return tfds.load('librispeech', split='train_clean100')


def _get_imagenet_dataset():
  return tfds.load('imagenet2012')['full']


def get_enwik9_iterator(
    num_chunks: int = constants.NUM_CHUNKS,
    sequence_length: int = constants.CHUNK_SIZE_BYTES,
) -> Iterator[bytes]:
  """Returns an iterator for enwik9 data."""
  if not os.path.exists('enwik9'):
    # Downloading and extracting the dataset.
    urllib.request.urlretrieve(
        'https://mattmahoney.net/dc/enwik9.zip',
        'enwik9.zip',
    )
    with zipfile.ZipFile('enwik9.zip', 'r') as zip_ref:
      zip_ref.extract('enwik9')

  all_chunks = []
  with open('enwik9', 'rb') as file:
    for _ in range(num_chunks):
      all_chunks.append(file.read(sequence_length))
  return iter(all_chunks)


def _extract_audio_patches(sample: bytes) -> Iterator[bytes]:
  patches = np.array_split(
      np.frombuffer(sample, dtype=np.uint8),
      range(
          constants.CHUNK_SIZE_BYTES,
          len(sample),
          constants.CHUNK_SIZE_BYTES,
      ),
  )
  if len(patches[-1]) != constants.CHUNK_SIZE_BYTES:
    patches.pop()
  return map(lambda x: x.tobytes(), patches)


def get_librispeech_iterator(
    num_chunks: int = constants.NUM_CHUNKS,
) -> Iterator[bytes]:
  """Returns an iterator for librispeech data."""
  # Convert samples from 16 bit to 8 bit (i.e., changing from two channels to
  # one channel with `lin2lin`), adding 128 since 16 bit is signed (i.e., adding
  # 128 using `bias`).
  librispeech_dataset = map(
      lambda x: audioop.bias(audioop.lin2lin(x['speech'], 2, 1), 1, 128),
      _get_librispeech_dataset().as_numpy_iterator(),
  )
  idx = 0
  for data in librispeech_dataset:
    for patch in _extract_audio_patches(data):
      if idx == num_chunks:
        return
      yield patch
      idx += 1


def get_random_iterator(
    num_chunks: int = constants.NUM_CHUNKS,
) -> Iterator[bytes]:
  """Returns an iterator for random data."""
  for _ in range(num_chunks):
    yield random.randbytes(constants.CHUNK_SIZE_BYTES)


def _rgb_to_grayscale(image: np.ndarray) -> np.ndarray:
  return np.mean(image, axis=-1).astype(np.uint8)


def _extract_image_patches(image: np.ndarray) -> Iterator[bytes]:
  h, w = constants.CHUNK_SHAPE_2D
  height, width = image.shape

  for row, col in itertools.product(range(height // h), range(width // w)):
    yield image[row * h : (row + 1) * h, col * w : (col + 1) * w].tobytes()


def get_imagenet_iterator(
    num_chunks: int = constants.NUM_CHUNKS,
) -> Iterator[bytes]:
  """Returns a iterator for imagenet data."""
  imagenet_dataset = map(
      lambda x: _rgb_to_grayscale(x['image']),
      _get_imagenet_dataset().as_numpy_iterator(),
  )
  idx = 0
  for data in imagenet_dataset:
    for patch in _extract_image_patches(data):
      if idx == num_chunks:
        return
      yield patch
      idx += 1




# ==============================================================================
# Audio Data Loader Helper Functions
# ==============================================================================

def load_wav_file(file_path: str) -> tuple[np.ndarray, int, int]:
    """Load WAV file and return audio data, sample rate, and number of channels.
    
    Args:
        file_path: Path to the WAV file
        
    Returns:
        Tuple of (audio_data, sample_rate, num_channels)
        audio_data: Shape (channels, samples) for stereo, (1, samples) for mono
        sample_rate: Sample rate in Hz
        num_channels: Number of audio channels (1 for mono, 2 for stereo)
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is not a valid WAV file
    """
    from scipy.io import wavfile
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    if not file_path.lower().endswith('.wav'):
        raise ValueError(f"File must be a WAV file: {file_path}")
    
    try:
        # Load audio file using scipy.wavfile
        # scipy.wavfile returns (sample_rate, audio_data) where audio_data is (samples, channels)
        sample_rate, audio_data = wavfile.read(file_path)
        
        # Handle mono vs stereo
        if audio_data.ndim == 1:
            # Mono audio
            audio_data = audio_data.reshape(1, -1)  # Shape: (1, samples)
            num_channels = 1
        else:
            # Stereo audio - transpose to (channels, samples)
            audio_data = audio_data.T  # Shape: (channels, samples)
            num_channels = audio_data.shape[0]
        
        # Convert to float32 and normalize to [-1, 1] range
        if audio_data.dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32767.0
        elif audio_data.dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483647.0
        elif audio_data.dtype == np.uint8:
            audio_data = (audio_data.astype(np.float32) - 128.0) / 128.0
        else:
            # Assume already in float format
            audio_data = audio_data.astype(np.float32)
        
        return audio_data, sample_rate, num_channels
        
    except Exception as e:
        raise ValueError(f"Error loading WAV file {file_path}: {str(e)}")




def process_stereo_blocking(audio_data: np.ndarray, blocking_size: int) -> np.ndarray:
    """Process stereo audio using interleaved blocking strategy.
    
    Args:
        audio_data: Audio data with shape (channels, samples)
        blocking_size: Size of each block in samples
        
    Returns:
        Processed audio data as 1D array with interleaved blocks
        Pattern: [L_block1, R_block1, L_block2, R_block2, ...]
    """
    channels, total_samples = audio_data.shape
    
    if channels == 1:
        # Mono audio - return as is
        return audio_data[0]  # Return as 1D array
    
    elif channels == 2:
        # Stereo audio - apply blocking strategy
        left_channel = audio_data[0]  # Shape: (samples,)
        right_channel = audio_data[1]  # Shape: (samples,)
        
        # Split channels into blocks
        left_blocks = []
        right_blocks = []
        
        for i in range(0, total_samples, blocking_size):
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


def convert_to_target_bit_depth(audio_data: np.ndarray, use_16bit: bool) -> bytes:
    """Convert audio data to target bit depth and return as bytes.
    
    Args:
        audio_data: Audio data as 1D numpy array (float values in [-1, 1])
        use_16bit: If True, convert to 16-bit, else convert to 8-bit
        
    Returns:
        Audio data as bytes in the target bit depth
    """
    if use_16bit:
        # Convert to 16-bit signed integers
        # Map [-1, 1] to [-32768, 32767]
        audio_16bit = (audio_data * 32767).astype(np.int16)
        return audio_16bit.tobytes()
    else:
        # Convert to 8-bit unsigned integers
        # Map [-1, 1] to [0, 255]
        audio_8bit = ((audio_data + 1.0) * 127.5).astype(np.uint8)
        return audio_8bit.tobytes()


def extract_audio_chunks(audio_bytes: bytes, chunk_size: int) -> Iterator[bytes]:
    """Extract fixed-size chunks from audio bytes.
    
    Args:
        audio_bytes: Audio data as bytes
        chunk_size: Size of each chunk in bytes
        
    Yields:
        Chunks of audio data as bytes
    """
    total_bytes = len(audio_bytes)
    
    for i in range(0, total_bytes, chunk_size):
        chunk = audio_bytes[i:i + chunk_size]
        
        # Only yield complete chunks
        if len(chunk) == chunk_size:
            yield chunk
        # Skip incomplete chunks at the end


def validate_blocking_size(blocking_size: int, chunk_size: int) -> None:
    """Validate blocking size against chunk size and issue warnings if needed.
    
    Args:
        blocking_size: Size of each block in samples
        chunk_size: Size of each chunk in bytes
        
    Raises:
        Warning if blocking_size * 2 < chunk_size
    """
    # For stereo audio, each block contributes 2 * blocking_size samples
    # If blocking_size * 2 < chunk_size, chunks won't contain complete L/R block pairs
    if blocking_size * 2 < chunk_size:
        warnings.warn(
            f"Blocking size {blocking_size} is too small for chunk size {chunk_size}. "
            f"Chunks will not contain complete L/R block pairs. "
            f"Consider increasing blocking_size to at least {chunk_size // 2}.",
            UserWarning
        )


def validate_audio_config(
    use_16bit: bool, 
    blocking_size: int, 
    chunk_size: int
) -> None:
    """Validate audio configuration parameters.
    
    Args:
        use_16bit: Whether to use 16-bit audio
        blocking_size: Size of each block in samples
        chunk_size: Size of each chunk in bytes
        
    Raises:
        ValueError: If any parameter is invalid
    """
    if blocking_size <= 0:
        raise ValueError(f"Blocking size must be positive, got {blocking_size}")
    
    if chunk_size <= 0:
        raise ValueError(f"Chunk size must be positive, got {chunk_size}")
    
    # Validate blocking size against chunk size
    validate_blocking_size(blocking_size, chunk_size)
    
    # Additional validation for 16-bit mode
    if use_16bit:
        # For 16-bit audio, chunk_size should be even (since each sample is 2 bytes)
        if chunk_size % 2 != 0:
            raise ValueError(
                f"For 16-bit audio, chunk_size must be even, got {chunk_size}"
            )


def get_custom_audio_iterator(
    audio_files: List[str],
    num_chunks: int = constants.NUM_CHUNKS,
    use_16bit: bool = False,
    blocking_size: int = 1024,
    chunk_size_bytes: int = constants.CHUNK_SIZE_BYTES,
) -> Iterator[bytes]:
    """Custom audio iterator for WAV files with configurable bit depth and stereo blocking.
    
    Args:
        audio_files: List of paths to WAV files (provided by calling code)
        num_chunks: Maximum number of chunks to generate
        use_16bit: If True, use 16-bit audio (vocab size 65536), else 8-bit (vocab size 256)
        blocking_size: Size of blocks for stereo processing (in samples)
        chunk_size_bytes: Size of each chunk in bytes
        
    Yields:
        Chunks of audio data as bytes
        
    Raises:
        ValueError: If configuration is invalid
        FileNotFoundError: If audio files don't exist
    """
    # Validate configuration
    validate_audio_config(use_16bit, blocking_size, chunk_size_bytes)
    
    # Validate audio files
    for file_path in audio_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    logging.info(f"Processing {len(audio_files)} audio files")
    logging.info(f"Configuration: 16-bit={use_16bit}, blocking_size={blocking_size}, "
                f"chunk_size={chunk_size_bytes} bytes")
    
    chunks_generated = 0
    file_index = 0
    
    while chunks_generated < num_chunks and file_index < len(audio_files):
        audio_file = audio_files[file_index]
        
        try:
            # Load audio file
            audio_data, sample_rate, num_channels = load_wav_file(audio_file)
            logging.info(f"Loaded {audio_file}: {num_channels} channels, {sample_rate}Hz, "
                        f"{audio_data.shape[1]} samples")
            
            # Process stereo blocking
            processed_audio = process_stereo_blocking(audio_data, blocking_size)
            
            # Convert to target bit depth
            audio_bytes = convert_to_target_bit_depth(processed_audio, use_16bit)
            
            # Extract chunks
            for chunk in extract_audio_chunks(audio_bytes, chunk_size_bytes):
                if chunks_generated >= num_chunks:
                    break
                    
                yield chunk
                chunks_generated += 1
                
                if chunks_generated % 1000 == 0:
                    logging.info(f"Generated {chunks_generated} chunks")
            
            logging.info(f"Processed {audio_file}: generated chunks from this file")
            
        except Exception as e:
            logging.warning(f"Error processing {audio_file}: {str(e)}")
            # Continue with next file
        
        file_index += 1
    
    logging.info(f"Total chunks generated: {chunks_generated}")
    
    if chunks_generated < num_chunks:
        warnings.warn(
            f"Only generated {chunks_generated} chunks out of requested {num_chunks}. "
            f"Consider adding more audio files.",
            UserWarning
        )


# Data generator function dictionary - defined after all functions
GET_DATA_GENERATOR_FN_DICT = {
    'enwik9': get_enwik9_iterator,
    'imagenet': get_imagenet_iterator,
    'librispeech': get_librispeech_iterator,
    'custom_audio': get_custom_audio_iterator,
    'random': get_random_iterator,
}
