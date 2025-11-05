"""Implements data loaders for new audio datasets (excluding LibriSpeech, which is already in data_loaders.py)."""

# To add a new audio data loader:
# 1. Create _get_<dataset name>_dataset() function (see _get_librispeech_dataset() for reference in data_loaders.py).
# 2. Create get_<dataset name>_iterator(num_chunks: int = constants.NUM_CHUNKS, bit_depth: int = constants_audio.BIT_DEPTH) -> Iterator[bytes] function (see get_librispeech_iterator() for reference in data_loaders.py).
# 3. Add key '<dataset name>' with value get_<dataset name>_iterator to this file's GET_AUDIO_DATA_GENERATOR_FN_DICT, which will be merged with GET_DATA_GENERATOR_FN_DICT in data_loaders.py.

import audioop
from collections.abc import Iterator
import pandas as pd
import numpy as np
import scipy.io.wavfile
import functools

from language_modeling_is_compression import constants
from language_modeling_is_compression import constants_audio

np.random.seed(42)


def _get_musdb18mono_dataset():
  """Returns an iterator that yields numpy arrays, one per song."""
  # Load MUSDB18 dataset
  musdb18mono = pd.read_csv(filepath_or_buffer = f"{constants_audio.MUSDB18MONO_DATA_DIR}/mixes.csv", sep = ",", header = 0, index_col = False)
  musdb18mono["path"] = musdb18mono["path"].apply(lambda path: f"{constants_audio.MUSDB18MONO_DATA_DIR}/{path}")

  # filter dataset
  if constants_audio.MUSDB18MONO_MIXES_ONLY: # include only mixes, instead of everything
    musdb18mono = musdb18mono[musdb18mono["is_mix"]]
  if constants_audio.MUSDB18MONO_PARTITION == "train": # include only the "train" partition specified in constants_audio.MUSDB18MONO_PARTITION
    musdb18mono = musdb18mono[musdb18mono["is_train"]]
  elif constants_audio.MUSDB18MONO_PARTITION == "valid": # include only the "valid" partition specified in constants_audio.MUSDB18MONO_PARTITION
    musdb18mono = musdb18mono[~musdb18mono["is_train"]]

  # Return an iterator that yields one track at a time
  for path in musdb18mono["path"]:
    sample_rate, waveform = scipy.io.wavfile.read(filename = path) # get the mixture audio as a numpy array
    yield waveform


def _get_musdb18stereo_dataset():
  """Returns an iterator that yields numpy arrays, one per song."""
  # Load MUSDB18 dataset
  musdb18stereo = pd.read_csv(filepath_or_buffer = f"{constants_audio.MUSDB18STEREO_DATA_DIR}/mixes.csv", sep = ",", header = 0, index_col = False)
  musdb18stereo["path"] = musdb18stereo["path"].apply(lambda path: f"{constants_audio.MUSDB18STEREO_DATA_DIR}/{path}")

  # filter dataset
  if constants_audio.MUSDB18STEREO_MIXES_ONLY: # include only mixes, instead of everything
    musdb18stereo = musdb18stereo[musdb18stereo["is_mix"]]
  if constants_audio.MUSDB18STEREO_PARTITION == "train": # include only the "train" partition specified in constants_audio.MUSDB18STEREO_PARTITION
    musdb18stereo = musdb18stereo[musdb18stereo["is_train"]]
  elif constants_audio.MUSDB18STEREO_PARTITION == "valid": # include only the "valid" partition specified in constants_audio.MUSDB18STEREO_PARTITION
    musdb18stereo = musdb18stereo[~musdb18stereo["is_train"]]

  # Return an iterator that yields one track at a time
  for path in musdb18stereo["path"]:
    sample_rate, waveform = scipy.io.wavfile.read(filename = path) # get the mixture audio as a numpy array
    yield waveform


def _validate_arguments(
    chunk_size: int,
    num_chunks: int,
    bit_depth: int,
) -> None:
  """Validates arguments."""
  assert chunk_size > 0, f"Chunk size must be greater than 0. Provided chunk size: {chunk_size}."
  assert num_chunks > 0, f"Number of chunks must be greater than 0. Provided number of chunks: {num_chunks}."
  assert bit_depth in constants_audio.VALID_BIT_DEPTHS, f"Invalid bit depth: {bit_depth}. Valid bit depths are {constants_audio.VALID_BIT_DEPTHS}."
  assert (chunk_size / (bit_depth // 8) % 1) == 0, f"With given bit depth, cannot fit a whole number of samples into a chunk. The number of bytes per sample (bit_depth // 8) must evenly divide the chunk size. Provided chunk size: {chunk_size}. Provided bit depth: {bit_depth}."


def _extract_audio_patches(
    sample: bytes,
    chunk_size: int = constants.CHUNK_SIZE_BYTES,
) -> Iterator[bytes]:
  """Extracts audio patches from a sample."""
  patches = np.array_split(
      np.frombuffer(sample, dtype=np.uint8),
      range(
          chunk_size,
          len(sample),
          chunk_size,
      ),
  )
  if constants_audio.RANDOMIZE_CHUNKS: # shuffle patches randomly
    np.random.shuffle(patches)
  if len(patches[-1]) != chunk_size:
    patches.pop()
  return map(lambda x: x.tobytes(), patches)


def _convert_waveform_to_bytes(
    waveform: np.ndarray,
    bit_depth: int = constants_audio.BIT_DEPTH,
) -> bytes:
  """Converts a waveform to bytes."""
  assert bit_depth in constants_audio.VALID_BIT_DEPTHS, f"Invalid bit depth: {bit_depth}. Valid bit depths are {constants_audio.VALID_BIT_DEPTHS}."
  # determine if waveform is signed
  is_waveform_signed = np.issubdtype(waveform.dtype, np.signedinteger)

  # convert samples to specified bit depth
  current_width = waveform.dtype.itemsize # determine current width
  new_width = bit_depth // 8 # determine new width
  assert new_width in {1, 2, 3}
  waveform = audioop.lin2lin(waveform, current_width, new_width) # convert waveform to correct size

  # add bias if necessary to convert signed waveform to unsigned waveform 
  if is_waveform_signed:
    bias = 2 ** (bit_depth - 1)
    waveform = audioop.bias(waveform, new_width, bias)

  # return waveform as bytes
  return waveform


def get_musdb18mono_iterator(
    chunk_size: int = constants.CHUNK_SIZE_BYTES,
    num_chunks: int = constants.NUM_CHUNKS,
    bit_depth: int = constants_audio.BIT_DEPTH,
) -> Iterator[bytes]:
  """Returns an iterator for musdb18mono data."""
  _validate_arguments(chunk_size, num_chunks, bit_depth)
  musdb18mono_dataset = _get_musdb18mono_dataset()
  musdb18mono_dataset = map( # convert waveform to bytes
      functools.partial(_convert_waveform_to_bytes, bit_depth=bit_depth),
      musdb18mono_dataset,
  )
  idx = 0
  for data in musdb18mono_dataset:
    for patch in _extract_audio_patches(data, chunk_size=chunk_size):
      if idx == num_chunks:
        return
      yield patch
      idx += 1
      if idx % constants_audio.CHUNKS_PER_SAMPLE == 0:
        break


def _interleave_stereo_waveform(
    waveform: np.ndarray,
) -> np.ndarray:
  """Interleaves a stereo waveform."""
  assert waveform.ndim == 2 and waveform.shape[1] == 2, f"Waveform must be a 2D numpy array with 2 columns. Got shape {waveform.shape}."
  return waveform


def get_musdb18stereo_iterator(
    chunk_size: int = constants.CHUNK_SIZE_BYTES,
    num_chunks: int = constants.NUM_CHUNKS,
    bit_depth: int = constants_audio.BIT_DEPTH,
) -> Iterator[bytes]:
  """Returns an iterator for musdb18stereo data."""
  _validate_arguments(chunk_size, num_chunks, bit_depth)
  musdb18stereo_dataset = map( # convert stereo waveform to pseudo-mono interleaved waveform
      _interleave_stereo_waveform,
      _get_musdb18stereo_dataset(),
  )
  musdb18stereo_dataset = map( # convert waveform to bytes
      functools.partial(_convert_waveform_to_bytes, bit_depth=bit_depth),
      musdb18stereo_dataset,
  )
  idx = 0
  for data in musdb18stereo_dataset:
    for patch in _extract_audio_patches(data, chunk_size=chunk_size):
      if idx == num_chunks:
        return
      yield patch
      idx += 1
      if idx % constants_audio.CHUNKS_PER_SAMPLE == 0:
        break


GET_AUDIO_DATA_GENERATOR_FN_DICT = { # ensure none of the keys are the same as GET_DATA_GENERATOR_FN_DICT in data_loaders.py, since its values will overwrite values with shared key names in this dictionary
    'musdb18mono': get_musdb18mono_iterator,
    'musdb18stereo': get_musdb18stereo_iterator,
}
if constants_audio.MERGE_LMIC_DATA_GENERATOR_FN_DICT:
  from language_modeling_is_compression import data_loaders
  GET_AUDIO_DATA_GENERATOR_FN_DICT.update(data_loaders.GET_DATA_GENERATOR_FN_DICT) # add original data loaders for backwards compatibility
