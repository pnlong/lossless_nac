#!/usr/bin/env python3
"""
Naive Rice decoding wrapper script that calls the C helper program.

This script:
1. Reads a binary file containing encoded Rice data
2. Calls the C naive rice decoder  
3. Outputs decoded 32-bit integers to stdout
"""

import sys
import os
import subprocess
import tempfile

def main():
    if len(sys.argv) != 3 and len(sys.argv) != 4:
        print("Usage: python3 naive_rice_decode.py <input_file> <num_samples> [k_parameter]", file=sys.stderr)
        print("  input_file: binary file containing encoded Rice data", file=sys.stderr)
        print("  num_samples: number of samples to decode", file=sys.stderr)
        print("  k_parameter: Rice parameter (default: 1)", file=sys.stderr)
        sys.exit(1)
    
    input_filename = sys.argv[1]
    num_samples = sys.argv[2]
    k_param = sys.argv[3] if len(sys.argv) >= 4 else "1"
    
    # Validate num_samples
    try:
        num_samples_int = int(num_samples)
        if num_samples_int <= 0:
            raise ValueError("num_samples must be positive")
    except ValueError:
        print(f"Error: Invalid num_samples: {num_samples}", file=sys.stderr)
        sys.exit(1)
    
    # Path to C helper program
    script_dir = os.path.dirname(os.path.abspath(__file__))
    decoder_path = os.path.join(script_dir, "naive_rice_decode_helper")
    
    # Check if C helper exists
    if not os.path.exists(decoder_path):
        print(f"Error: C helper not found at {decoder_path}", file=sys.stderr)
        print("Please run 'make' in the naive_rice_helpers directory to compile the helpers", file=sys.stderr)
        sys.exit(1)
    
    # Check if input file exists
    if not os.path.exists(input_filename):
        print(f"Error: Input file not found: {input_filename}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Call C decoder helper
        result = subprocess.run(
            [decoder_path, input_filename, num_samples, k_param],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        
        # Output decoded data to stdout
        sys.stdout.buffer.write(result.stdout)
        
    except subprocess.CalledProcessError as e:
        stderr_text = e.stderr.decode("utf-8", errors="ignore") if e.stderr else "No error message"
        print(f"C decoder failed (exit {e.returncode}): {stderr_text}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error calling C decoder: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 