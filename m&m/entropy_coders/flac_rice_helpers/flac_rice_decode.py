#!/usr/bin/env python3
"""
FLAC entropy decoding script for LPC order 0 implementation.

This script:
1. Reads an entropy-coded binary file (output from flac_entropy_encode2.py)
2. Extracts method type and partition order from the embedded metadata
3. Decodes Rice-coded residuals using modified FLAC decoder
4. Outputs decoded int32 residuals as binary to stdout
"""

import sys
import os
import struct
import subprocess
import tempfile
import argparse

# Magic marker constants (must match the encoder)
ENTROPY_CODING_START_MAGIC = 0xDEADBEEF
ENTROPY_CODING_END_MAGIC = 0xCAFEBABE

def validate_entropy_file(filename):
    """Validate that the file contains entropy-coded data."""
    try:
        with open(filename, 'rb') as f:
            data = f.read()
        
        if len(data) < 1:  # At least some data
            raise ValueError(f"File is empty: {len(data)} bytes")
        
        # The file should contain pure entropy-coded data without magic markers
        # Basic validation: check that we have some data
        if len(data) > 10000:  # Sanity check - entropy data shouldn't be too large
            print(f"Warning: Entropy file is quite large: {len(data)} bytes", file=sys.stderr)
        
        print(f"Validated entropy file: {len(data)} bytes (pure entropy-coded data)", file=sys.stderr)
        return True
        
    except Exception as e:
        print(f"Error validating entropy file: {e}", file=sys.stderr)
        return False

def compile_helper_if_needed(helper_source_path, helper_executable_path, flac_lib_path, include_path):
    """Compile the C helper if it doesn't exist or if source is newer."""
    
    # Check if executable exists and is newer than source
    if (os.path.exists(helper_executable_path) and 
        os.path.exists(helper_source_path) and
        os.path.getmtime(helper_executable_path) > os.path.getmtime(helper_source_path)):
        return True  # Already compiled and up to date
    
    print("Compiling FLAC entropy decoder helper...", file=sys.stderr)
    
    # Compile the helper
    compile_cmd = [
        'gcc', '-O2', '-I' + include_path, helper_source_path, 
        flac_lib_path, '-lm', '-o', helper_executable_path
    ]
    
    try:
        result = subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
        print("Decoder helper compiled successfully", file=sys.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed: {e}", file=sys.stderr)
        print(f"Command: {' '.join(compile_cmd)}", file=sys.stderr)
        print(f"Error output: {e.stderr}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description='FLAC entropy decoding script')
    parser.add_argument('entropy_file', help='Entropy-coded binary file (from flac_entropy_encode2.py)')
    parser.add_argument('num_samples', type=int, help='Number of samples to decode')
    parser.add_argument('--flac-lib', default='/home/pnlong/lnac/flac_entropy_coding/src/libFLAC/.libs/libFLAC-static.a', 
                        help='Path to FLAC static library')
    parser.add_argument('--include-path', default='/home/pnlong/lnac/flac_entropy_coding/include', 
                        help='Path to FLAC include directory')
    
    args = parser.parse_args()
    
    # Get paths for the helper program
    script_dir = os.path.dirname(os.path.abspath(__file__))
    helper_source_path = os.path.join(script_dir, 'flac_rice_decode_helper.c')
    helper_executable_path = os.path.join(script_dir, 'flac_rice_decode_helper')
    
    # Validate inputs
    if args.num_samples <= 0:
        print(f"Error: Invalid number of samples: {args.num_samples}", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(args.entropy_file):
        print(f"Error: Entropy file {args.entropy_file} does not exist", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(args.flac_lib):
        print(f"Error: FLAC library {args.flac_lib} does not exist", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.exists(helper_source_path):
        print(f"Error: Helper source file {helper_source_path} does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Validate entropy file format
    if not validate_entropy_file(args.entropy_file):
        sys.exit(1)
    
    # Compile helper if needed
    if not compile_helper_if_needed(helper_source_path, helper_executable_path, args.flac_lib, args.include_path):
        sys.exit(1)
    
    print(f"Decoding {args.num_samples} samples from {args.entropy_file}...", file=sys.stderr)
    
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create temporary output file
        temp_output = os.path.join(temp_dir, 'decoded.bin')
        
        # Run the helper program
        print("Running FLAC entropy decoder...", file=sys.stderr)
        try:
            result = subprocess.run([
                helper_executable_path, 
                args.entropy_file, 
                temp_output, 
                str(args.num_samples)
            ], check=True, capture_output=True, text=True)
            
            if result.stderr:
                print(result.stderr, file=sys.stderr)
                
        except subprocess.CalledProcessError as e:
            print(f"Decoding failed: {e}", file=sys.stderr)
            if e.stderr:
                print(f"Error output: {e.stderr}", file=sys.stderr)
            sys.exit(1)
        
        # Read the decoded output
        try:
            with open(temp_output, 'rb') as f:
                decoded_data = f.read()
            
            # Verify we got the expected amount of data
            expected_bytes = args.num_samples * 4  # 4 bytes per int32
            if len(decoded_data) != expected_bytes:
                print(f"Warning: Expected {expected_bytes} bytes, got {len(decoded_data)} bytes", file=sys.stderr)
            
            print(f"Successfully decoded {len(decoded_data)} bytes ({len(decoded_data)//4} samples)", file=sys.stderr)
            
            # Output decoded residuals to stdout
            sys.stdout.buffer.write(decoded_data)
            sys.stdout.buffer.flush()
            
        except Exception as e:
            print(f"Error reading decoded output: {e}", file=sys.stderr)
            sys.exit(1)

if __name__ == '__main__':
    main() 