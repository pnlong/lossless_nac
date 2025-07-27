# README
# Phillip Long
# July 21, 2025

# Speed test for comparing bitarray-based vs manual bit manipulation implementations.

# IMPORTS
##################################################

import numpy as np
import time
from typing import List, Tuple
import sys
from os.path import dirname, realpath

sys.path.insert(0, dirname(dirname(realpath(__file__))))
sys.path.insert(0, dirname(dirname(dirname(realpath(__file__)))))
import bitstream as bitarray_impl
import utils as manual_impl

##################################################


def generate_random_bits(n_bits: int) -> List[bool]:
    """Generate a random sequence of bits."""
    return np.random.randint(0, 2, size=n_bits, dtype=bool).tolist()

def time_write_sequence(bits: List[bool], implementation: str = "bitarray") -> Tuple[bytes, float]:
    """Time how long it takes to write a sequence of bits."""
    if implementation == "bitarray":
        stream = bitarray_impl.BitOutputStream(path=None, buffer_size=len(bits)//8 + 1)
    else:
        stream = manual_impl.BitOutputStream()
    
    start_time = time.perf_counter()
    
    for bit in bits:
        stream.write_bit(bit)
    
    encoded = stream.flush()
    end_time = time.perf_counter()
    
    return encoded, end_time - start_time

def time_read_sequence(encoded: bytes, n_bits: int, implementation: str = "bitarray") -> Tuple[List[bool], float]:
    """Time how long it takes to read a sequence of bits."""
    if implementation == "bitarray":
        stream = bitarray_impl.BitInputStream(stream=encoded)
    else:
        stream = manual_impl.BitInputStream(stream=encoded)
    
    start_time = time.perf_counter()
    
    decoded = []
    for _ in range(n_bits):
        decoded.append(stream.read_bit())
    
    end_time = time.perf_counter()
    
    return decoded, end_time - start_time

def run_speed_test(n_bits: int = 1_000_000, n_trials: int = 5):
    """Run speed test comparing both implementations."""
    print(f"Speed test for {n_bits:,} bits")
    print("-" * 50)
    
    # Generate random bits
    bits = generate_random_bits(n_bits)
    
    # Test results storage
    bitarray_write_times = []
    bitarray_read_times = []
    manual_write_times = []
    manual_read_times = []
    
    for trial in range(n_trials):
        print(f"\nTrial {trial + 1}/{n_trials}")
        
        # Test bitarray implementation
        encoded_bitarray, write_time = time_write_sequence(bits, "bitarray")
        decoded_bitarray, read_time = time_read_sequence(encoded_bitarray, n_bits, "bitarray")
        assert decoded_bitarray == bits, "Bitarray implementation failed to reproduce original sequence"
        bitarray_write_times.append(write_time)
        bitarray_read_times.append(read_time)
        print(f"Bitarray impl: Write: {write_time:.6f}s, Read: {read_time:.6f}s")
        
        # Test manual implementation
        encoded_manual, write_time = time_write_sequence(bits, "manual")
        decoded_manual, read_time = time_read_sequence(encoded_manual, n_bits, "manual")
        assert decoded_manual == bits, "Manual implementation failed to reproduce original sequence"
        manual_write_times.append(write_time)
        manual_read_times.append(read_time)
        print(f"Manual impl:   Write: {write_time:.6f}s, Read: {read_time:.6f}s")
    
    # Print summary statistics
    print("\nSummary Statistics (averaged over trials):")
    print("-" * 50)
    print("Bitarray Implementation:")
    print(f"  Write: {np.mean(bitarray_write_times):.6f}s ± {np.std(bitarray_write_times):.6f}s")
    print(f"  Read:  {np.mean(bitarray_read_times):.6f}s ± {np.std(bitarray_read_times):.6f}s")
    print(f"  Total: {np.mean(bitarray_write_times) + np.mean(bitarray_read_times):.6f}s")
    print("\nManual Implementation:")
    print(f"  Write: {np.mean(manual_write_times):.6f}s ± {np.std(manual_write_times):.6f}s")
    print(f"  Read:  {np.mean(manual_read_times):.6f}s ± {np.std(manual_read_times):.6f}s")
    print(f"  Total: {np.mean(manual_write_times) + np.mean(manual_read_times):.6f}s")
    
    # Calculate speedup
    total_bitarray = np.mean(bitarray_write_times) + np.mean(bitarray_read_times)
    total_manual = np.mean(manual_write_times) + np.mean(manual_read_times)
    speedup = total_manual / total_bitarray
    print(f"\nSpeedup: {speedup:.2f}x")

if __name__ == "__main__":
    import sys
    n_bits = int(sys.argv[1]) if len(sys.argv) > 1 else 1_000_000
    run_speed_test(n_bits=n_bits) 