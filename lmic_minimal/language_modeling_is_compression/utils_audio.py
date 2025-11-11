"""Utility functions for audio data."""

from language_modeling_is_compression import utils
from typing import Tuple

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
