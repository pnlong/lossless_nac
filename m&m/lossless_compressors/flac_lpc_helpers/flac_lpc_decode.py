#!/usr/bin/env python3
"""
FLAC LPC Decoder - Decode estimator bits to approximate waveform using direct prediction algorithms.

Usage: python3 flac_lpc_decode.py <encoded.bin>

Input: encoded.bin - estimator bits (without magic markers) 
Output: raw binary samples (32-bit signed integers) representing estimated waveform to stdout
"""

import sys
import os
import subprocess
import tempfile

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <encoded.bin>", file=sys.stderr)
        print("Input: estimator bits (without magic markers)", file=sys.stderr)
        print("Output: raw binary samples (32-bit signed integers) to stdout", file=sys.stderr)
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Get paths for the helper program
    script_dir = os.path.dirname(os.path.abspath(__file__))
    helper_executable_path = os.path.join(script_dir, 'flac_lpc_decode_helper')
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Check if helper executable exists
    if not os.path.exists(helper_executable_path):
        print(f"Error: Helper executable {helper_executable_path} does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Create temporary file for decoded samples
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_filepath = temp_file.name
    
    try:
        # Run the helper program
        print("Running FLAC LPC decoder...", file=sys.stderr)
        result = subprocess.run([helper_executable_path, input_file, temp_filepath], 
                              check=True, capture_output=True, text=True)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        # Read the decoded samples from temporary file
        with open(temp_filepath, 'rb') as f:
            decoded_samples = f.read()
        
        print(f"Decoded {len(decoded_samples) // 4} samples", file=sys.stderr)
        
        # Output samples to stdout
        sys.stdout.buffer.write(decoded_samples)
        sys.stdout.buffer.flush()
        
        print("Decoding completed successfully", file=sys.stderr)
        
    except subprocess.CalledProcessError as e:
        print(f"Decoding failed: {e}", file=sys.stderr)
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
