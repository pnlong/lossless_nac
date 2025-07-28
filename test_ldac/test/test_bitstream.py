import os
import sys
import time
import random
import numpy as np
from typing import Tuple, List

# Add parent directory to Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import bitstream as cython_bs  # Cython implementation
import bitstream_py as python_bs  # Original Python implementation

def generate_test_data(size: int) -> Tuple[List[bool], bytes, int]:
    """Generate test data: random bits, bytes, and an integer."""
    # Generate random bits
    bits = [random.choice([True, False]) for _ in range(size)]
    
    # Generate random bytes
    byte_data = os.urandom(size // 8)
    
    # Generate random 32-bit integer
    test_int = random.randint(0, 2**32 - 1)
    
    return bits, byte_data, test_int

def test_correctness(implementation: str):
    """Test correctness of a bitstream implementation."""
    print(f"\nTesting {implementation} implementation for correctness...")
    
    # Choose implementation
    if implementation == "cython":
        BitInputStream = cython_bs.BitInputStream
        BitOutputStream = cython_bs.BitOutputStream
    else:
        BitInputStream = python_bs.BitInputStream
        BitOutputStream = python_bs.BitOutputStream
    
    # Test parameters
    test_size = 1000
    bits, byte_data, test_int = generate_test_data(test_size)
    
    # Create a temporary file for testing in the test directory
    temp_file = os.path.join(os.path.dirname(__file__), "test_stream.bin")
    
    try:
        # Test writing
        output_stream = BitOutputStream(temp_file)
        
        # Write and verify individual bits
        for bit in bits:
            output_stream.write_bit(bit)
        
        # Write and verify bytes
        output_stream.align_to_byte()
        output_stream.write_bytes(byte_data)
        
        # Write and verify integer
        output_stream.write_uint(test_int)
        
        output_stream.close()
        
        # Test reading
        input_stream = BitInputStream(temp_file)
        
        # Read and verify bits
        read_bits = []
        for _ in range(len(bits)):
            read_bits.append(input_stream.read_bit())
        
        if read_bits != bits:
            print("Bit mismatch details:")
            for i, (expected, actual) in enumerate(zip(bits, read_bits)):
                if expected != actual:
                    print(f"Position {i}: Expected {expected}, Got {actual}")
            raise AssertionError(f"Bit mismatch in {implementation}")
        
        # Read and verify bytes
        input_stream.align_to_byte()
        read_bytes = input_stream.read_bytes(len(byte_data))
        if read_bytes != byte_data:
            print("Byte mismatch details:")
            print(f"Expected: {byte_data.hex()}")
            print(f"Got:      {read_bytes.hex()}")
            raise AssertionError(f"Byte mismatch in {implementation}")
        
        # Read and verify integer
        read_int = input_stream.read_uint()
        if read_int != test_int:
            print(f"Integer mismatch: Expected {test_int}, Got {read_int}")
            raise AssertionError(f"Integer mismatch in {implementation}")
        
        print(f"{implementation} implementation passed all correctness tests!")
        return True
        
    except Exception as e:
        print(f"Error in {implementation} implementation: {str(e)}")
        return False
        
    finally:
        # Cleanup
        if os.path.exists(temp_file):
            os.remove(temp_file)

def benchmark_performance(implementation: str, sizes: List[int], iterations: int = 5):
    """Benchmark performance of a bitstream implementation."""
    print(f"\nBenchmarking {implementation} implementation...")
    
    # Choose implementation
    if implementation == "cython":
        BitInputStream = cython_bs.BitInputStream
        BitOutputStream = cython_bs.BitOutputStream
    else:
        BitInputStream = python_bs.BitInputStream
        BitOutputStream = python_bs.BitOutputStream
    
    results = {}
    bench_file = os.path.join(os.path.dirname(__file__), "bench_stream.bin")
    
    for size in sizes:
        print(f"\nTesting with {size} bits...")
        bits, byte_data, test_int = generate_test_data(size)
        
        # Measure write performance
        write_times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            
            output_stream = BitOutputStream(bench_file)
            for bit in bits:
                output_stream.write_bit(bit)
            output_stream.align_to_byte()
            output_stream.write_bytes(byte_data)
            output_stream.write_uint(test_int)
            output_stream.close()
            
            write_times.append(time.perf_counter() - start_time)
        
        # Measure read performance
        read_times = []
        for _ in range(iterations):
            start_time = time.perf_counter()
            
            input_stream = BitInputStream(bench_file)
            for _ in range(len(bits)):
                input_stream.read_bit()
            input_stream.align_to_byte()
            input_stream.read_bytes(len(byte_data))
            input_stream.read_uint()
            
            read_times.append(time.perf_counter() - start_time)
        
        # Store results
        results[size] = {
            'write': np.mean(write_times),
            'read': np.mean(read_times)
        }
        
        # Cleanup
        if os.path.exists(bench_file):
            os.remove(bench_file)
    
    return results

def main():
    """Main test function."""
    # Test correctness
    python_correct = test_correctness("python")
    cython_correct = test_correctness("cython")
    
    if not (python_correct and cython_correct):
        print("\nSkipping benchmarks due to correctness failures.")
        return
    
    # Benchmark performance
    sizes = [1000, 10000, 100000, 1000000]
    python_results = benchmark_performance("python", sizes)
    cython_results = benchmark_performance("cython", sizes)
    
    # Print comparison
    print("\nPerformance Comparison:")
    print("\nSize (bits) | Implementation | Write Time (s) | Read Time (s) | Speedup (Write) | Speedup (Read)")
    print("-" * 85)
    
    for size in sizes:
        py_write = python_results[size]['write']
        py_read = python_results[size]['read']
        cy_write = cython_results[size]['write']
        cy_read = cython_results[size]['read']
        
        write_speedup = py_write / cy_write
        read_speedup = py_read / cy_read
        
        print(f"{size:10d} | Python        | {py_write:13.6f} | {py_read:12.6f} | {'N/A':14s} | {'N/A':13s}")
        print(f"{size:10d} | Cython        | {cy_write:13.6f} | {cy_read:12.6f} | {write_speedup:14.2f} | {read_speedup:13.2f}")
        print("-" * 85)

if __name__ == "__main__":
    main() 