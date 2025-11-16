"""Implements data loaders for new audio datasets (excluding LibriSpeech, which is already in data_loaders.py)."""

# To add a new audio data loader:
# 1. Create _get_<dataset name>_dataset() function (see _get_librispeech_dataset() for reference in data_loaders.py).
# 2. Create get_<dataset name>_iterator(num_chunks: int = constants.NUM_CHUNKS, bit_depth: int = constants_audio.BIT_DEPTH) -> Iterator[bytes] function (see get_librispeech_iterator() for reference in data_loaders.py).
# 3. Add key '<dataset name>' with value get_<dataset name>_iterator to this file's GET_AUDIO_DATA_GENERATOR_FN_DICT, which will be merged with GET_DATA_GENERATOR_FN_DICT in data_loaders.py.

import audioop
from collections.abc import Iterator
from typing import Callable, Dict
import pandas as pd
import numpy as np
import scipy.io.wavfile
import functools
import glob
from os.path import basename, dirname

from language_modeling_is_compression import constants
from language_modeling_is_compression import constants_audio
from language_modeling_is_compression import utils_audio

np.random.seed(42)


def _get_musdb18mono_dataset(
    partition: str = None,
    subset: str = None,
) -> Iterator[np.ndarray]:
  """Returns an iterator that yields numpy arrays, one per song."""
  assert partition is None or partition in ("train", "valid"), f"Invalid partition: {partition}. Valid partitions are None, 'train', and 'valid'."

  # Load MUSDB18 dataset
  musdb18mono = pd.read_csv(filepath_or_buffer=f"{constants_audio.MUSDB18MONO_DATA_DIR}/mixes.csv", sep=',', header=0, index_col=False)
  musdb18mono["path"] = musdb18mono["path"].apply(lambda path: f"{constants_audio.MUSDB18MONO_DATA_DIR}/{path}")

  # filter dataset
  if subset == "mixes": # include only mixes, instead of everything
    musdb18mono = musdb18mono[musdb18mono["is_mix"]]
  elif subset == "stems": # include only stems, instead of everything
    musdb18mono = musdb18mono[~musdb18mono["is_mix"]]
  if partition == "train": # include only the "train" partition
    musdb18mono = musdb18mono[musdb18mono["is_train"]]
  elif partition == "valid": # include only the "valid" partition
    musdb18mono = musdb18mono[~musdb18mono["is_train"]]

  # Return an iterator that yields one track at a time
  for path in musdb18mono["path"]:
    waveform, sample_rate = utils_audio.load_audio(path=path, bit_depth=16)
    yield waveform


def _get_musdb18stereo_dataset(
    partition: str = None,
    subset: str = None,
) -> Iterator[np.ndarray]:
  """Returns an iterator that yields numpy arrays, one per song."""
  assert partition is None or partition in ("train", "valid"), f"Invalid partition: {partition}. Valid partitions are None, 'train', and 'valid'."

  # Load MUSDB18 dataset
  musdb18stereo = pd.read_csv(filepath_or_buffer=f"{constants_audio.MUSDB18STEREO_DATA_DIR}/mixes.csv", sep=',', header=0, index_col=False)
  musdb18stereo["path"] = musdb18stereo["path"].apply(lambda path: f"{constants_audio.MUSDB18STEREO_DATA_DIR}/{path}")

  # filter dataset
  if subset == "mixes": # include only mixes, instead of everything
    musdb18stereo = musdb18stereo[musdb18stereo["is_mix"]]
  elif subset == "stems": # include only stems, instead of everything
    musdb18stereo = musdb18stereo[~musdb18stereo["is_mix"]]
  if partition == "train": # include only the "train" partition
    musdb18stereo = musdb18stereo[musdb18stereo["is_train"]]
  elif partition == "valid": # include only the "valid" partition
    musdb18stereo = musdb18stereo[~musdb18stereo["is_train"]]

  # Return an iterator that yields one track at a time
  for path in musdb18stereo["path"]:
    waveform, sample_rate = utils_audio.load_audio(path=path, bit_depth=16)
    yield waveform


def _get_librispeech_dataset() -> Iterator[np.ndarray]:
  """Returns an iterator that yields numpy arrays, one per song."""
  # Return an iterator that yields one track at a time
  for path in glob.iglob(f"{constants_audio.LIBRISPEECH_DATA_DIR}/**/*.flac", recursive=True):
    waveform, sample_rate = utils_audio.load_audio(path=path, bit_depth=16)
    yield waveform


def _get_ljspeech_dataset() -> Iterator[np.ndarray]:
  """Returns an iterator that yields numpy arrays, one per song."""
  # Return an iterator that yields one track at a time
  for path in glob.iglob(f"{constants_audio.LJSPEECH_DATA_DIR}/**/*.wav", recursive=True):
    waveform, sample_rate = utils_audio.load_audio(path=path, bit_depth=16)
    yield waveform


def _get_epidemic_dataset() -> Iterator[np.ndarray]:
  """Returns an iterator that yields numpy arrays, one per song."""
  # Return an iterator that yields one track at a time
  for path in glob.iglob(f"{constants_audio.EPIDEMIC_SOUND_DATA_DIR}/**/*.flac", recursive=True):
    waveform, sample_rate = utils_audio.load_audio(path=path, bit_depth=16)
    yield waveform


def _get_vctk_dataset() -> Iterator[np.ndarray]:
  """Returns an iterator that yields numpy arrays, one per song."""
  # Return an iterator that yields one track at a time
  for path in glob.iglob(f"{constants_audio.VCTK_DATA_DIR}/**/*.flac", recursive=True):
    waveform, sample_rate = utils_audio.load_audio(path=path, bit_depth=16)
    yield waveform


def _get_torrent_dataset(
    native_bit_depth: int,
    subset: str,
) -> Iterator[np.ndarray]:
  """Returns an iterator that yields numpy arrays, one per song."""
  assert native_bit_depth == 16 or native_bit_depth == 24, f"Invalid native bit depth: {native_bit_depth}. Valid native bit depths for Torrent data are 16 and 24."
  assert subset is None or subset in ("pro", "amateur", "freeload"), f"Invalid subset: {subset}. Valid subsets are None, 'pro', 'amateur', and 'freeload'."

  # Get paths
  if subset == "pro":
    paths = glob.iglob(f"{constants_audio.TORRENT_DATA_DATA_DIR}/Pro/{native_bit_depth}b/**/*.flac", recursive = True)
  elif subset == "amateur":
    paths = glob.iglob(f"{constants_audio.TORRENT_DATA_DATA_DIR}/train/Amateur/{native_bit_depth}b/**/*.flac", recursive = True)
  elif subset == "freeload":
    paths = glob.iglob(f"{constants_audio.TORRENT_DATA_DATA_DIR}/train/Freeload/{native_bit_depth}b/**/*.flac", recursive = True)
  else:
    paths = glob.iglob(f"{constants_audio.TORRENT_DATA_DATA_DIR}/**/{native_bit_depth}b/**/*.flac", recursive = True)

  # Return an iterator that yields one track at a time
  for path in paths:
    waveform, sample_rate = utils_audio.load_audio(path=path, bit_depth=native_bit_depth)
    yield waveform


def _get_birdvox_dataset() -> Iterator[np.ndarray]:
  """Returns an iterator that yields numpy arrays, one per song."""
  # Return an iterator that yields one track at a time
  for path in filter(lambda path: basename(dirname(path)) != "split_data", glob.iglob(f"{constants_audio.BIRDVOX_DATA_DIR}/**/*.flac", recursive=True)):
    waveform, sample_rate = utils_audio.load_audio(path=path, bit_depth=16)
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


def _interleave_stereo_waveform_if_necessary(
    waveform: np.ndarray,
) -> np.ndarray:
  """Interleaves a stereo waveform."""
  if waveform.ndim == 1 or (waveform.ndim == 2 and waveform.shape[1] == 1):
    return waveform
  elif waveform.ndim == 2 and waveform.shape[1] == 2:
    waveform = waveform.flatten()
    return waveform
  else:
    raise ValueError(f"Invalid waveform shape: {waveform.shape}. Valid shapes are 1D or 2D with 1 or 2 columns.")


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
  
  # determine properties of waveform
  is_waveform_signed = np.issubdtype(waveform.dtype, np.signedinteger)
  n_samples = np.prod(waveform.shape) # determine number of samples
  current_width = waveform.dtype.itemsize # determine current width
  assert current_width in {1, 2, 4}, f"Invalid current width: {current_width}. Valid current widths are 1, 2, and 4 bytes, representing np.int8/np.uint8, np.int16/np.uint16, and np.int32/np.uint32, respectively."
  
  # support differs for 32-bit samples, since this really means 24-bit samples, which are not properly represented because the 24-bit samples only comprise 3 of the 4 bytes
  if current_width == 4: # 32-bit samples really means 24-bit samples, which are not properly represented because the 24-bit samples only comprise 3 of the 4 bytes
    
    # if signed (np.int32), convert to unsigned (np.uint32), which will make later steps easier
    if is_waveform_signed: # so this means technically, these are 24-bit signed samples, but we'll convert them to 24-bit unsigned samples for easier processing
      min_val, max_val = waveform.min(), waveform.max()
      assert min_val >= -(2 ** 23) and max_val <= (2 ** 23) - 1, f"Waveform must be in the range [-(2 ** 23), (2 ** 23) - 1]. Got min {min_val} and max {max_val}."
      waveform = (waveform + (2 ** 23)).astype(np.uint32) # add 2 ** 23 to convert to unsigned (np.uint32)
      is_waveform_signed = False # waveform is now guaranteed to be unsigned, since if it didn't enter this block, it was already unsigned

    # convert into (n_samples, 4) array of bytes
    waveform = np.frombuffer(waveform.tobytes(), dtype=np.uint8)
    waveform = waveform.reshape(-1, 4) # reshape into (n_samples, 4) array of bytes
    assert waveform.shape[0] == n_samples, f"Number of samples does not match. Expected {n_samples}, got {waveform.shape[0]}."

    # convert to 24-bit samples, convert to bytes
    waveform = waveform[:, :3] # get first 3 bytes of each 4-byte sample
    waveform = waveform.flatten() # flatten, and because we converted to unsigned earlier, this is a 24-bit unsigned waveform once converted to bytes
    waveform = waveform.tobytes()
    current_width = 3 # 24-bit samples are represented in 3 bytes

  # otherwise, the waveform is 16-bit or 8-bit, which are properly represented in full width
  else:
    waveform = waveform.tobytes()  

  # convert waveform to correct size
  new_width = bit_depth // 8 # determine new width
  assert new_width in {1, 2, 3}, f"Invalid new width: {new_width}. Valid new widths are 1, 2, and 3 bytes, representing 8-bit, 16-bit, and 24-bit audio, respectively."
  waveform = audioop.lin2lin(waveform, current_width, new_width) # convert waveform to correct size

  # add bias if necessary to convert signed waveform to unsigned waveform 
  if is_waveform_signed:
    bias = 2 ** (bit_depth - 1)
    waveform = audioop.bias(waveform, new_width, bias)

  # return waveform as bytes, representing unsigned 8-bit, 16-bit, or 24-bit audio
  return waveform


def get_dataset_iterator(
    dataset: Iterator[np.ndarray],
    chunk_size: int = constants.CHUNK_SIZE_BYTES,
    num_chunks: int = constants.NUM_CHUNKS,
    bit_depth: int = constants_audio.BIT_DEPTH,
) -> Iterator[bytes]:
  """Returns an iterator for a dataset."""
  _validate_arguments(chunk_size, num_chunks, bit_depth)
  dataset = map( # convert stereo waveform to pseudo-mono interleaved waveform
      _interleave_stereo_waveform_if_necessary,
      dataset,
  )
  dataset = map( # convert waveform to bytes
      functools.partial(_convert_waveform_to_bytes, bit_depth=bit_depth),
      dataset,
  )
  idx = 0
  for data in dataset:
    for patch in _extract_audio_patches(data, chunk_size=chunk_size):
      yield patch
      idx += 1
      if idx == num_chunks:
        return
      elif idx % constants_audio.CHUNKS_PER_SAMPLE == 0:
        break


def get_musdb18mono_iterator(
    chunk_size: int = constants.CHUNK_SIZE_BYTES,
    num_chunks: int = constants.NUM_CHUNKS,
    bit_depth: int = constants_audio.BIT_DEPTH,
    partition: str = None,
    subset: str = None,
) -> Iterator[bytes]:
  """Returns an iterator for musdb18mono data."""
  musdb18mono_dataset = _get_musdb18mono_dataset(partition=partition, subset=subset)
  return get_dataset_iterator(musdb18mono_dataset, chunk_size=chunk_size, num_chunks=num_chunks, bit_depth=bit_depth)


def get_musdb18stereo_iterator(
    chunk_size: int = constants.CHUNK_SIZE_BYTES,
    num_chunks: int = constants.NUM_CHUNKS,
    bit_depth: int = constants_audio.BIT_DEPTH,
    partition: str = None,
    subset: str = None,
) -> Iterator[bytes]:
  """Returns an iterator for musdb18stereo data."""
  musdb18stereo_dataset = _get_musdb18stereo_dataset(partition=partition, subset=subset)
  return get_dataset_iterator(musdb18stereo_dataset, chunk_size=chunk_size, num_chunks=num_chunks, bit_depth=bit_depth)


def get_librispeech_iterator(
    chunk_size: int = constants.CHUNK_SIZE_BYTES,
    num_chunks: int = constants.NUM_CHUNKS,
    bit_depth: int = constants_audio.BIT_DEPTH,
) -> Iterator[bytes]:
  """Returns an iterator for librispeech data."""
  librispeech_dataset = _get_librispeech_dataset()
  return get_dataset_iterator(librispeech_dataset, chunk_size=chunk_size, num_chunks=num_chunks, bit_depth=bit_depth)


def get_ljspeech_iterator(
    chunk_size: int = constants.CHUNK_SIZE_BYTES,
    num_chunks: int = constants.NUM_CHUNKS,
    bit_depth: int = constants_audio.BIT_DEPTH,
) -> Iterator[bytes]:
  """Returns an iterator for ljspeech data."""
  ljspeech_dataset = _get_ljspeech_dataset()
  return get_dataset_iterator(ljspeech_dataset, chunk_size=chunk_size, num_chunks=num_chunks, bit_depth=bit_depth)


def get_epidemic_iterator(
    chunk_size: int = constants.CHUNK_SIZE_BYTES,
    num_chunks: int = constants.NUM_CHUNKS,
    bit_depth: int = constants_audio.BIT_DEPTH,
) -> Iterator[bytes]:
  """Returns an iterator for epidemic data."""
  epidemic_dataset = _get_epidemic_dataset()
  return get_dataset_iterator(epidemic_dataset, chunk_size=chunk_size, num_chunks=num_chunks, bit_depth=bit_depth)


def get_vctk_iterator(
    chunk_size: int = constants.CHUNK_SIZE_BYTES,
    num_chunks: int = constants.NUM_CHUNKS,
    bit_depth: int = constants_audio.BIT_DEPTH,
) -> Iterator[bytes]:
  """Returns an iterator for vctk data."""
  vctk_dataset = _get_vctk_dataset()
  return get_dataset_iterator(vctk_dataset, chunk_size=chunk_size, num_chunks=num_chunks, bit_depth=bit_depth)


def get_torrent_iterator(
    chunk_size: int = constants.CHUNK_SIZE_BYTES,
    num_chunks: int = constants.NUM_CHUNKS,
    bit_depth: int = constants_audio.BIT_DEPTH,
    native_bit_depth: int = constants_audio.TORRENT_DATA_NATIVE_BIT_DEPTH,
    subset: str = None,
) -> Iterator[bytes]:
  """Returns an iterator for torrent data."""
  torrent_dataset = _get_torrent_dataset(native_bit_depth=native_bit_depth, subset=subset)
  return get_dataset_iterator(torrent_dataset, chunk_size=chunk_size, num_chunks=num_chunks, bit_depth=bit_depth)


def get_birdvox_iterator(
    chunk_size: int = constants.CHUNK_SIZE_BYTES,
    num_chunks: int = constants.NUM_CHUNKS,
    bit_depth: int = constants_audio.BIT_DEPTH,
) -> Iterator[bytes]:
  """Returns an iterator for birdvox data."""
  birdvox_dataset = _get_birdvox_dataset()
  return get_dataset_iterator(birdvox_dataset, chunk_size=chunk_size, num_chunks=num_chunks, bit_depth=bit_depth)


# dictionary of audio data generator functions
def get_audio_data_generator_fn_dict() -> Dict[str, Callable[[], Iterator[bytes]]]:
  """Return the choices of datasets."""
  audio_data_generator_fn_dict = dict()
  for subset in (None, "mixes", "stems"): # None for all, "mixes" for mixes only, "stems" for stems only
    for partition in (None, "train", "valid"): # None for all, "train" for train, "valid" for valid
      audio_data_generator_fn_dict["musdb18mono" + (f"_{subset}" if subset is not None else "") + (f"_{partition}" if partition is not None else "")] = functools.partial(get_musdb18mono_iterator, partition=partition, subset=subset)
      audio_data_generator_fn_dict["musdb18stereo" + (f"_{subset}" if subset is not None else "") + (f"_{partition}" if partition is not None else "")] = functools.partial(get_musdb18stereo_iterator, partition=partition, subset=subset)
  audio_data_generator_fn_dict["librispeech"] = get_librispeech_iterator
  audio_data_generator_fn_dict["ljspeech"] = get_ljspeech_iterator
  audio_data_generator_fn_dict["epidemic"] = get_epidemic_iterator
  audio_data_generator_fn_dict["vctk"] = get_vctk_iterator
  for native_bit_depth in (16, 24):
    for subset in (None, "pro", "amateur", "freeload"):
      audio_data_generator_fn_dict[f"torrent{native_bit_depth}b" + (f"_{subset}" if subset is not None else "")] = functools.partial(get_torrent_iterator, native_bit_depth=native_bit_depth, subset=subset)
  audio_data_generator_fn_dict["birdvox"] = get_birdvox_iterator
  return audio_data_generator_fn_dict
GET_AUDIO_DATA_GENERATOR_FN_DICT = get_audio_data_generator_fn_dict()
