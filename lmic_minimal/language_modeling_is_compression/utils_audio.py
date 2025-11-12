"""Utility functions for audio data."""

from typing import Tuple
import soundfile as sf
import pydub
import io
import numpy as np

from language_modeling_is_compression import utils
from language_modeling_is_compression import constants_audio

def right_shift_bytes_by_one(data: bytes) -> Tuple[bytes, bytes, int]:
  """Returns right-shifted bytes, i.e., divided by 2, and the number of bytes.

  Our language models were trained on ASCII data. However, not all bytes can be
  decoded to ASCII, so we set the most significant bit (MSB) to 0, to ensure
  that we can decode the data to ASCII.

  However, for certain data types (e.g., images), masking the MSB and leaving
  the rest of the byte unchanged will destroy the structure of the data. Thus,
  we instead divide the number by two (i.e., we shift the bits to the right by
  one).

  Args:
    data: The bytes to be shifted.
  """
  n_discarded_bits = len(data)
  shifted_bytes = bytes([byte >> 1 for byte in data])
  discarded_lsbs_bits = ''.join(map(str, [byte & 1 for byte in data]))
  discarded_lsbs, _ = utils.bits_to_bytes(discarded_lsbs_bits)
  return shifted_bytes, discarded_lsbs, n_discarded_bits


def add_discarded_lsbs_back(shifted_bytes: bytes, discarded_lsbs: bytes) -> bytes:
  """Adds the discarded LSBs back to the data.

  Args:
    shifted_bytes: The shifted bytes to add the discarded LSBs back to.
    discarded_lsbs: The discarded LSBs to add back to the data.
  """
  n_discarded_bits = len(shifted_bytes)
  num_padded_bits = 8 - (n_discarded_bits % 8) if n_discarded_bits % 8 != 0 else 0 # number of extra bits to pad to make the discarded LSBs a multiple of 8
  discarded_lsbs_bits = utils.bytes_to_bits(discarded_lsbs, num_padded_bits=num_padded_bits)
  assert len(discarded_lsbs_bits) == n_discarded_bits, f"The discarded LSBs and the shifted bytes must have the same length (one discarded LSB per shifted byte), but got {len(discarded_lsbs_bits)=} and {n_discarded_bits=}"
  reconstructed_bytes = bytes([shifted_byte << 1 | lsb for shifted_byte, lsb in zip(shifted_bytes, map(int, discarded_lsbs_bits))])
  return reconstructed_bytes


def load_audio(path: str, bit_depth: int, expected_sample_rate: int = None) -> Tuple[np.ndarray, int]:
  """
  Load an audio file and convert it to the target bit depth.
  
  Args:
    path: The path to the audio file.
    bit_depth: The target bit depth. See constants_audio.VALID_BIT_DEPTHS for valid bit depths.
    expected_sample_rate: The expected sample rate. If None, the sample rate will not be checked.
  
  Returns:
    The waveform and sample rate. Note that waveform will always be signed integer data type.
  """

  # ensure bit depth is valid
  assert bit_depth in constants_audio.VALID_BIT_DEPTHS, f"Invalid bit depth: {bit_depth}. Valid bit depths are {constants_audio.VALID_BIT_DEPTHS}."

  # read audio file
  try:
    waveform, sample_rate = sf.read(file = path, dtype = np.float32) # get the audio as a numpy array
  except Exception as e: # if soundfile fails, use alternative method, which reads the file as a pydub AudioSegment and then converts to a numpy array
    # warnings.warn(f"Error reading audio file {path} with soundfile, using alternative method: {e}", category = RuntimeWarning)
    waveform = pydub.AudioSegment.from_file(file = path, format = path.split(".")[-1])
    stream = io.BytesIO()
    waveform.export(stream, format = "FLAC") # export as corrected FLAC file
    waveform, sample_rate = sf.read(file = stream, dtype = np.float32) # get the audio as a numpy array
    del stream
  waveform_dtype = np.int8 if bit_depth == 8 else np.int16 if bit_depth == 16 else np.int32
  waveform = (waveform * ((2 ** (bit_depth - 1)) - 1)).astype(waveform_dtype)

  # make assertions
  if expected_sample_rate is not None:
    assert sample_rate == expected_sample_rate, f"Sample rate mismatch: {sample_rate} != {expected_sample_rate}."
  waveform_min, waveform_max = waveform.min(), waveform.max()
  expected_waveform_min, expected_waveform_max = -(2 ** (bit_depth - 1)), (2 ** (bit_depth - 1)) - 1
  assert waveform_min >= expected_waveform_min and waveform_max <= expected_waveform_max, f"Waveform must be in the range [{expected_waveform_min}, {expected_waveform_max}]. Got min {waveform_min} and max {waveform_max}."
  
  # return waveform and sample rate
  return waveform, sample_rate