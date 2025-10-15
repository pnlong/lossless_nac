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

"""Defines project-wide constants."""

NUM_CHUNKS = 488281
CHUNK_SIZE_BYTES = 2048
CHUNK_SHAPE_2D = (32, 64)
ALPHABET_SIZE = 256

# Audio-specific constants
ALPHABET_SIZE_16BIT = 65536  # For 16-bit audio
MIN_BLOCKING_SIZE = 512      # Minimum blocking size for stereo
DEFAULT_BLOCKING_SIZE = 1024 # Default blocking size

# Base 2 means that the coder writes bits.
ARITHMETIC_CODER_BASE = 2
# Precision 32 implies 32 bit arithmetic.
ARITHMETIC_CODER_PRECISION = 32

# Bit depth configurations for extended audio support
# Note: Alphabet size remains constant at 256 (ASCII characters) for all bit depths
# Higher bit depths are processed by splitting samples into multiple 8-bit parts
# Following the paper's approach: divide each byte by 2 (right-shift), lose LSB, store LSB bits

BIT_DEPTH_CONFIGS = {
    8: {
        'alphabet_size': ALPHABET_SIZE,  # 256 ASCII characters
        'bytes_per_sample': 1,
        'ascii_chars_per_sample': 1,  # 1 ASCII char per 8-bit sample
        'dropped_bits_per_sample': 1,  # 1 LSB bit dropped per 8-bit part
    },
    16: {
        'alphabet_size': ALPHABET_SIZE,  # 256 ASCII characters (same as 8-bit)
        'bytes_per_sample': 2,
        'ascii_chars_per_sample': 2,  # 2 ASCII chars per 16-bit sample
        'dropped_bits_per_sample': 2,  # 1 LSB bit dropped per 8-bit part
    },
    24: {
        'alphabet_size': ALPHABET_SIZE,  # 256 ASCII characters (same as 8-bit)
        'bytes_per_sample': 3,
        'ascii_chars_per_sample': 3,  # 3 ASCII chars per 24-bit sample
        'dropped_bits_per_sample': 3,  # 1 LSB bit dropped per 8-bit part
    },
    32: {
        'alphabet_size': ALPHABET_SIZE,  # 256 ASCII characters (same as 8-bit)
        'bytes_per_sample': 4,
        'ascii_chars_per_sample': 4,  # 4 ASCII chars per 32-bit sample
        'dropped_bits_per_sample': 4,  # 1 LSB bit dropped per 8-bit part
    }
}
