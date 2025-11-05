import sys
sys.path.insert(0, "/home/pnlong/lnac/lmic_minimal")
from language_modeling_is_compression.compressors_audio import llama
from language_modeling_is_compression.compressors_audio import flac

import scipy.io.wavfile

import numpy as np
import audioop

path = "/graft3/datasets/pnlong/lnac/sashimi/data/musdb18mono/train/Lushlife - Toynbee Suite.1.left.wav"
sr, data = scipy.io.wavfile.read(path)
i_start = 4000
chunk_size = 2048
data = data[i_start:i_start + chunk_size]
data = data.astype(np.int8)
data = data.tobytes()
data = audioop.bias(data, 1, 128) # convert to 8-bit unsigned integers
original_data_length = len(data)
print(f"original data length: {original_data_length}")

llama_model = 'llama-2-7b'

print(f"COMPRESSING DATA...")
compressed_data, num_padded_bits = llama.compress(data, llama_model = llama_model, return_num_padded_bits=True, use_slow_lossless_compression=True)
# compressed_data = flac.compress(data, bit_depth = 8, sample_rate = sr)
compressed_data_length = len(compressed_data)
# print(f"compressed data: {compressed_data}")
print(f"compressed data length: {compressed_data_length}")
print(f"number of padded bits: {num_padded_bits}")
print(f"compression ratio: {compressed_data_length / original_data_length:.2%}")

# compress data quickly
quick_compressed_data = llama.compress(data, llama_model = llama_model, return_num_padded_bits=False, use_slow_lossless_compression=False)
quick_compressed_data_length = len(quick_compressed_data)
print(f"quick compressed data length: {quick_compressed_data_length}")
print(f"compression ratio: {quick_compressed_data_length / original_data_length:.2%}")
print(f"quick compressed data matches compressed data: {quick_compressed_data == compressed_data}")

# try to decompress for real
decompressed_data = llama.decompress(compressed_data, llama_model = llama_model, num_padded_bits = num_padded_bits, uncompressed_length = original_data_length)
# decompressed_data = flac.decompress(compressed_data, bit_depth = 8, sample_rate = sr)
# print(f"decompressed data: {decompressed_data}")
decompressed_data_length = len(decompressed_data)
print(f"decompressed data length: {decompressed_data_length}")
assert decompressed_data == data, "Decompressed data does not match original data"
print(f"✓ Decompressed data matches original data")