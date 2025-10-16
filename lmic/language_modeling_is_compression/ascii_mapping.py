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

"""ASCII mapping functions for different bit depths to enable Llama model compatibility."""

from typing import Callable, Tuple


def _pack_lsb_bits(lsb_bits_list: list) -> bytes:
  """Pack a list of LSB bits into bytes (8 bits per byte)."""
  dropped_bits_bytes = []
  for i in range(0, len(lsb_bits_list), 8):
    # Get next 8 bits
    bits = lsb_bits_list[i:i+8]
    # Pad with zeros if less than 8 bits
    while len(bits) < 8:
      bits.append(0)
    # Pack into byte
    packed_byte = 0
    for j, bit in enumerate(bits):
      packed_byte |= (bit << j)
    dropped_bits_bytes.append(packed_byte)
  return bytes(dropped_bits_bytes)


def ascii_map_8bit(data: bytes) -> Tuple[bytes, bytes]:
  """Map 8-bit data to ASCII range [0, 127] following the paper's approach.
  
  Paper: "for each byte, to map it into the range [0, 127], we simply divide it by 2, and lose the least significant bit"
  
  Args:
    data: Input bytes
    
  Returns:
    tuple: (ascii_bytes, dropped_lsb_bits)
  """
  ascii_bytes = []
  dropped_bits_list = []
  
  for byte in data:
    # Paper approach: divide by 2 (right-shift by 1), lose LSB
    ascii_byte = byte >> 1  # This maps [0, 255] to [0, 127]
    lsb = byte & 1  # Extract the LSB that was lost
    
    ascii_bytes.append(ascii_byte)
    dropped_bits_list.append(lsb)
  
  # Pack LSB bits into bytes (8 bits per byte)
  dropped_bits_bytes = _pack_lsb_bits(dropped_bits_list)
  
  return bytes(ascii_bytes), dropped_bits_bytes


def ascii_map_16bit(data: bytes) -> Tuple[bytes, bytes]:
  """Map 16-bit data to ASCII by splitting into 2 8-bit parts and applying ASCII mapping.
  
  A 16-bit sample becomes 2 ASCII characters, with dropped LSB bits stored separately.
  Paper approach: divide each 8-bit part by 2, lose LSB.
  This preserves full 16-bit precision by using 2 ASCII characters per sample.
  
  Args:
    data: Input bytes (must have even length)
    
  Returns:
    tuple: (ascii_bytes, dropped_lsb_bits)
  """
  if len(data) % 2 != 0:
    raise ValueError("16-bit data must have even number of bytes")
  
  ascii_bytes = []
  dropped_bits_list = []
  
  # Process each 16-bit sample
  for i in range(0, len(data), 2):
    # Get 16-bit sample (keep full 16-bit precision)
    sample_16bit = int.from_bytes(data[i:i+2], byteorder='little', signed=True)
    
    # Split into upper and lower 8-bit parts
    upper_8bit = (sample_16bit >> 8) & 0xFF
    lower_8bit = sample_16bit & 0xFF
    
    # Apply paper's ASCII mapping to each 8-bit part (divide by 2, lose LSB)
    for byte_val in [upper_8bit, lower_8bit]:
      ascii_byte = byte_val >> 1  # Divide by 2, maps [0, 255] to [0, 127]
      lsb = byte_val & 1  # Extract the LSB that was lost
      
      ascii_bytes.append(ascii_byte)
      dropped_bits_list.append(lsb)
  
  # Pack LSB bits into bytes (8 bits per byte)
  dropped_bits_bytes = _pack_lsb_bits(dropped_bits_list)
  
  return bytes(ascii_bytes), dropped_bits_bytes


def ascii_map_24bit(data: bytes) -> Tuple[bytes, bytes]:
  """Map 24-bit data to ASCII by splitting into 3 8-bit parts and applying ASCII mapping.
  
  A 24-bit sample becomes 3 ASCII characters, with dropped LSB bits stored separately.
  Paper approach: divide each 8-bit part by 2, lose LSB.
  This preserves full 24-bit precision by using 3 ASCII characters per sample.
  
  Args:
    data: Input bytes (must have length divisible by 3)
    
  Returns:
    tuple: (ascii_bytes, dropped_lsb_bits)
  """
  if len(data) % 3 != 0:
    raise ValueError("24-bit data must have length divisible by 3")
  
  ascii_bytes = []
  dropped_bits_list = []
  
  # Process each 24-bit sample
  for i in range(0, len(data), 3):
    # Get 24-bit sample (keep full 24-bit precision)
    sample_24bit = int.from_bytes(data[i:i+3], byteorder='little', signed=True)
    
    # Split into three 8-bit parts
    upper_8bit = (sample_24bit >> 16) & 0xFF
    middle_8bit = (sample_24bit >> 8) & 0xFF
    lower_8bit = sample_24bit & 0xFF
    
    # Apply paper's ASCII mapping to each 8-bit part (divide by 2, lose LSB)
    for byte_val in [upper_8bit, middle_8bit, lower_8bit]:
      ascii_byte = byte_val >> 1  # Divide by 2, maps [0, 255] to [0, 127]
      lsb = byte_val & 1  # Extract the LSB that was lost
      
      ascii_bytes.append(ascii_byte)
      dropped_bits_list.append(lsb)
  
  # Pack LSB bits into bytes (8 bits per byte)
  dropped_bits_bytes = _pack_lsb_bits(dropped_bits_list)
  
  return bytes(ascii_bytes), dropped_bits_bytes


def ascii_map_32bit(data: bytes) -> Tuple[bytes, bytes]:
  """Map 32-bit data to ASCII by splitting into 4 8-bit parts and applying ASCII mapping.
  
  A 32-bit sample becomes 4 ASCII characters, with dropped LSB bits stored separately.
  Paper approach: divide each 8-bit part by 2, lose LSB.
  This preserves full 32-bit precision by using 4 ASCII characters per sample.
  
  Args:
    data: Input bytes (must have length divisible by 4)
    
  Returns:
    tuple: (ascii_bytes, dropped_lsb_bits)
  """
  if len(data) % 4 != 0:
    raise ValueError("32-bit data must have length divisible by 4")
  
  ascii_bytes = []
  dropped_bits_list = []
  
  # Process each 32-bit sample
  for i in range(0, len(data), 4):
    # Get 32-bit sample (keep full 32-bit precision)
    sample_32bit = int.from_bytes(data[i:i+4], byteorder='little', signed=True)
    
    # Split into four 8-bit parts
    byte3 = (sample_32bit >> 24) & 0xFF
    byte2 = (sample_32bit >> 16) & 0xFF
    byte1 = (sample_32bit >> 8) & 0xFF
    byte0 = sample_32bit & 0xFF
    
    # Apply paper's ASCII mapping to each 8-bit part (divide by 2, lose LSB)
    for byte_val in [byte3, byte2, byte1, byte0]:
      ascii_byte = byte_val >> 1  # Divide by 2, maps [0, 255] to [0, 127]
      lsb = byte_val & 1  # Extract the LSB that was lost
      
      ascii_bytes.append(ascii_byte)
      dropped_bits_list.append(lsb)
  
  # Pack LSB bits into bytes (8 bits per byte)
  dropped_bits_bytes = _pack_lsb_bits(dropped_bits_list)
  
  return bytes(ascii_bytes), dropped_bits_bytes


def get_ascii_mapping_function_for_bit_depth(bit_depth: int) -> Callable[[bytes], Tuple[bytes, bytes]]:
  """Get the appropriate ASCII mapping function for a given bit depth.
  
  Args:
    bit_depth: Bit depth (8, 16, 24, or 32)
    
  Returns:
    ASCII mapping function for the specified bit depth
  """
  if bit_depth == 8:
    return ascii_map_8bit
  elif bit_depth == 16:
    return ascii_map_16bit
  elif bit_depth == 24:
    return ascii_map_24bit
  elif bit_depth == 32:
    return ascii_map_32bit
  else:
    raise ValueError(f"Unsupported bit depth: {bit_depth}. Supported: 8, 16, 24, 32")


def calculate_bits_per_sample(bit_depth: int) -> int:
  """Calculate bits per sample for a given bit depth."""
  return bit_depth


def calculate_bytes_per_sample(bit_depth: int) -> int:
  """Calculate bytes per sample for a given bit depth."""
  return bit_depth // 8


def calculate_ascii_chars_per_sample(bit_depth: int) -> int:
  """Calculate number of ASCII characters per sample for a given bit depth."""
  return bit_depth // 8


def reconstruct_original_bytes(ascii_bytes: bytes, dropped_lsb_bits: bytes, bit_depth: int) -> bytes:
  """Reconstruct original bytes from ASCII bytes and dropped LSB bits.
  
  Args:
    ascii_bytes: ASCII-mapped bytes
    dropped_lsb_bits: Dropped LSB bits
    bit_depth: Original bit depth
    
  Returns:
    Reconstructed original bytes
  """
  if bit_depth == 8:
    return reconstruct_8bit_bytes(ascii_bytes, dropped_lsb_bits)
  elif bit_depth == 16:
    return reconstruct_16bit_bytes(ascii_bytes, dropped_lsb_bits)
  elif bit_depth == 24:
    return reconstruct_24bit_bytes(ascii_bytes, dropped_lsb_bits)
  elif bit_depth == 32:
    return reconstruct_32bit_bytes(ascii_bytes, dropped_lsb_bits)
  else:
    raise ValueError(f"Unsupported bit depth: {bit_depth}")


def reconstruct_8bit_bytes(ascii_bytes: bytes, dropped_lsb_bits: bytes) -> bytes:
  """Reconstruct 8-bit bytes from ASCII bytes and dropped LSB bits."""
  original_bytes = []
  for i, ascii_byte in enumerate(ascii_bytes):
    lsb = dropped_lsb_bits[i] if i < len(dropped_lsb_bits) else 0
    # Reconstruct: original_byte = (ascii_byte << 1) | lsb
    original_byte = (ascii_byte << 1) | lsb
    original_bytes.append(original_byte)
  return bytes(original_bytes)


def reconstruct_16bit_bytes(ascii_bytes: bytes, dropped_lsb_bits: bytes) -> bytes:
  """Reconstruct 16-bit bytes from ASCII bytes and dropped LSB bits."""
  original_bytes = []
  for i in range(0, len(ascii_bytes), 2):
    # Reconstruct upper and lower 8-bit parts
    upper_ascii = ascii_bytes[i]
    lower_ascii = ascii_bytes[i + 1]
    upper_lsb = dropped_lsb_bits[i] if i < len(dropped_lsb_bits) else 0
    lower_lsb = dropped_lsb_bits[i + 1] if i + 1 < len(dropped_lsb_bits) else 0
    
    # Reconstruct: original_byte = (ascii_byte << 1) | lsb
    upper_8bit = (upper_ascii << 1) | upper_lsb
    lower_8bit = (lower_ascii << 1) | lower_lsb
    
    # Combine into 16-bit sample
    sample_16bit = (upper_8bit << 8) | lower_8bit
    original_bytes.extend(sample_16bit.to_bytes(2, byteorder='little', signed=True))
  
  return bytes(original_bytes)


def reconstruct_24bit_bytes(ascii_bytes: bytes, dropped_lsb_bits: bytes) -> bytes:
  """Reconstruct 24-bit bytes from ASCII bytes and dropped LSB bits."""
  original_bytes = []
  for i in range(0, len(ascii_bytes), 3):
    # Reconstruct three 8-bit parts
    upper_ascii = ascii_bytes[i]
    middle_ascii = ascii_bytes[i + 1]
    lower_ascii = ascii_bytes[i + 2]
    upper_lsb = dropped_lsb_bits[i] if i < len(dropped_lsb_bits) else 0
    middle_lsb = dropped_lsb_bits[i + 1] if i + 1 < len(dropped_lsb_bits) else 0
    lower_lsb = dropped_lsb_bits[i + 2] if i + 2 < len(dropped_lsb_bits) else 0
    
    # Reconstruct: original_byte = (ascii_byte << 1) | lsb
    upper_8bit = (upper_ascii << 1) | upper_lsb
    middle_8bit = (middle_ascii << 1) | middle_lsb
    lower_8bit = (lower_ascii << 1) | lower_lsb
    
    # Combine into 24-bit sample
    sample_24bit = (upper_8bit << 16) | (middle_8bit << 8) | lower_8bit
    original_bytes.extend(sample_24bit.to_bytes(3, byteorder='little', signed=True))
  
  return bytes(original_bytes)


def reconstruct_32bit_bytes(ascii_bytes: bytes, dropped_lsb_bits: bytes) -> bytes:
  """Reconstruct 32-bit bytes from ASCII bytes and dropped LSB bits."""
  original_bytes = []
  for i in range(0, len(ascii_bytes), 4):
    # Reconstruct four 8-bit parts
    byte3_ascii = ascii_bytes[i]
    byte2_ascii = ascii_bytes[i + 1]
    byte1_ascii = ascii_bytes[i + 2]
    byte0_ascii = ascii_bytes[i + 3]
    byte3_lsb = dropped_lsb_bits[i] if i < len(dropped_lsb_bits) else 0
    byte2_lsb = dropped_lsb_bits[i + 1] if i + 1 < len(dropped_lsb_bits) else 0
    byte1_lsb = dropped_lsb_bits[i + 2] if i + 2 < len(dropped_lsb_bits) else 0
    byte0_lsb = dropped_lsb_bits[i + 3] if i + 3 < len(dropped_lsb_bits) else 0
    
    # Reconstruct: original_byte = (ascii_byte << 1) | lsb
    byte3_8bit = (byte3_ascii << 1) | byte3_lsb
    byte2_8bit = (byte2_ascii << 1) | byte2_lsb
    byte1_8bit = (byte1_ascii << 1) | byte1_lsb
    byte0_8bit = (byte0_ascii << 1) | byte0_lsb
    
    # Combine into 32-bit sample
    sample_32bit = (byte3_8bit << 24) | (byte2_8bit << 16) | (byte1_8bit << 8) | byte0_8bit
    original_bytes.extend(sample_32bit.to_bytes(4, byteorder='little', signed=True))
  
  return bytes(original_bytes)
