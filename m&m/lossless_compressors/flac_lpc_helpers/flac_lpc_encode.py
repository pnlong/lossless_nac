#!/usr/bin/env python3
"""
FLAC LPC Encoder - Encode raw binary audio samples using modified FLAC with magic markers.

Usage: python3 flac_lpc_encode.py <input.bin>

Input: input.bin - raw binary file containing 32-bit signed integers (mono samples)
Output: estimator bits (without magic markers) to stdout
"""

import sys
import os
import subprocess
import tempfile

def compile_helper_if_needed(helper_source_path, helper_executable_path, flac_lib_path, include_path):
    """Compile the C helper if it doesn't exist or if source is newer."""
    
    # Check if executable exists and is newer than source
    if (os.path.exists(helper_executable_path) and 
        os.path.exists(helper_source_path) and
        os.path.getmtime(helper_executable_path) > os.path.getmtime(helper_source_path)):
        return True  # Already compiled and up to date
    
    print("Compiling FLAC LPC encoder helper...", file=sys.stderr)
    
    # Compile the helper
    compile_cmd = [
        'gcc', '-O2', '-I' + include_path, helper_source_path, 
        flac_lib_path, '-lm', '-o', helper_executable_path
    ]
    
    try:
        result = subprocess.run(compile_cmd, check=True, capture_output=True, text=True)
        print("Helper compiled successfully", file=sys.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Compilation failed: {e}", file=sys.stderr)
        print(f"Command: {' '.join(compile_cmd)}", file=sys.stderr)
        print(f"Error output: {e.stderr}", file=sys.stderr)
        return False

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <input.bin>", file=sys.stderr)
        print("Input: raw binary file containing 32-bit signed integers", file=sys.stderr)
        print("Output: estimator bits (without magic markers) to stdout", file=sys.stderr)
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Default paths
    flac_lib_path = '/home/pnlong/lnac/flac_lossy_estimating/src/libFLAC/.libs/libFLAC-static.a'
    include_path = '/home/pnlong/lnac/flac_lossy_estimating/include'
    
    # Get paths for the helper program
    script_dir = os.path.dirname(os.path.abspath(__file__))
    helper_source_path = os.path.join(script_dir, 'flac_lpc_encode_helper.c')
    helper_executable_path = os.path.join(script_dir, 'flac_lpc_encode_helper')
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Check if FLAC library exists
    if not os.path.exists(flac_lib_path):
        print(f"Error: FLAC library {flac_lib_path} does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Check if helper source exists
    if not os.path.exists(helper_source_path):
        print(f"Error: Helper source file {helper_source_path} does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Check if helper executable exists
    if not os.path.exists(helper_executable_path):
        print(f"Error: Helper executable {helper_executable_path} does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Create temporary file for estimator bits
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_filepath = temp_file.name
    
    try:
        # Run the helper program
        print("Running FLAC LPC encoder...", file=sys.stderr)
        result = subprocess.run([helper_executable_path, input_file, temp_filepath], 
                              check=True, capture_output=True, text=True)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        # Read the estimator bits from temporary file
        with open(temp_filepath, 'rb') as f:
            estimator_bits = f.read()
        
        print(f"Extracted {len(estimator_bits)} estimator bits", file=sys.stderr)
        
        # Output estimator bits to stdout
        sys.stdout.buffer.write(estimator_bits)
        sys.stdout.buffer.flush()
        
        print("Encoding completed successfully", file=sys.stderr)
        
    except subprocess.CalledProcessError as e:
        print(f"Encoding failed: {e}", file=sys.stderr)
        if e.stderr:
            print(f"Error output: {e.stderr}", file=sys.stderr)
        sys.exit(1)
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_filepath):
            try:
                os.remove(temp_filepath)
            except OSError:
                pass

if __name__ == '__main__':
    main()
