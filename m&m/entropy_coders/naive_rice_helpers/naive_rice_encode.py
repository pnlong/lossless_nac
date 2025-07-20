#!/usr/bin/env python3
"""
Naive Rice encoding wrapper script that calls the C helper program.

This script:
1. Reads a binary file containing 32-bit integers
2. Calls the C naive rice encoder
3. Outputs encoded bitstream to stdout
"""

import sys
import os
import subprocess
import tempfile

def main():
    if len(sys.argv) != 2 and len(sys.argv) != 3:
        print("Usage: python3 naive_rice_encode.py <input_file> [k_parameter]", file=sys.stderr)
        print("  input_file: binary file containing 32-bit signed integers", file=sys.stderr)
        print("  k_parameter: Rice parameter (default: 1)", file=sys.stderr)
        sys.exit(1)
    
    input_filename = sys.argv[1]
    k_param = sys.argv[2] if len(sys.argv) >= 3 else "1"
    
    # Path to C helper program
    script_dir = os.path.dirname(os.path.abspath(__file__))
    encoder_path = os.path.join(script_dir, "naive_rice_encode_helper")
    
    # Check if C helper exists
    if not os.path.exists(encoder_path):
        print(f"Error: C helper not found at {encoder_path}", file=sys.stderr)
        print("Please run 'make' in the naive_rice_helpers directory to compile the helpers", file=sys.stderr)
        sys.exit(1)
    
    # Check if input file exists
    if not os.path.exists(input_filename):
        print(f"Error: Input file not found: {input_filename}", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Call C encoder helper
        result = subprocess.run(
            [encoder_path, input_filename, k_param],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        
        # Output encoded data to stdout
        sys.stdout.buffer.write(result.stdout)
        
    except subprocess.CalledProcessError as e:
        stderr_text = e.stderr.decode("utf-8", errors="ignore") if e.stderr else "No error message"
        print(f"C encoder failed (exit {e.returncode}): {stderr_text}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error calling C encoder: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main() 