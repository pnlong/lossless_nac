#!/usr/bin/env python3
"""
FLAC LPC Decoder - Decode estimator bits to approximate waveform using direct prediction algorithms.

Usage: python3 flac_lpc_decode.py <encoded.bin> <num_samples>

Input: 
  - encoded.bin: estimator bits (without magic markers) with simplified format:
    * Constant: type(01) + constant_value
    * Verbatim: type(00) + bits_per_sample(5) + verbatim_samples  
    * Fixed: type(10) + order + warmup_samples + residuals
    * LPC: type(11) + order + qlp_coeff_precision + quantization_level + qlp_coeffs + warmup_samples + residuals
  - num_samples: number of samples to decode
Output: raw binary samples (32-bit signed integers) representing estimated waveform to stdout
"""

import sys
import os
import subprocess
import tempfile

def main():
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <encoded.bin> <num_samples>", file=sys.stderr)
        print("Input: estimator bits (without magic markers) + number of samples", file=sys.stderr)
        print("Simplified format (num_samples provided separately):", file=sys.stderr)
        print("  Constant: type(01) + constant_value", file=sys.stderr)
        print("  Verbatim: type(00) + bits_per_sample(5) + verbatim_samples", file=sys.stderr)
        print("  Fixed: type(10) + order + warmup_samples + residuals", file=sys.stderr)
        print("  LPC: type(11) + order + qlp_coeff_precision + quantization_level + qlp_coeffs + warmup_samples + residuals", file=sys.stderr)
        print("Output: raw binary samples (32-bit signed integers) to stdout", file=sys.stderr)
        sys.exit(1)
    
    input_file = sys.argv[1]
    try:
        num_samples = int(sys.argv[2])
        if num_samples <= 0:
            raise ValueError("Number of samples must be positive")
    except ValueError as e:
        print(f"Error: Invalid number of samples '{sys.argv[2]}': {e}", file=sys.stderr)
        sys.exit(1)
    
    # Get paths for the helper program
    script_dir = os.path.dirname(os.path.abspath(__file__))
    helper_executable_path = os.path.join(script_dir, 'flac_lpc_decode_helper_fixed')
    
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
        # Run the helper program with num_samples parameter
        print(f"Running FLAC LPC decoder for {num_samples} samples...", file=sys.stderr)
        result = subprocess.run([helper_executable_path, input_file, temp_filepath, str(num_samples)], 
                              check=True, capture_output=True, text=True)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        # Read the decoded samples from temporary file
        with open(temp_filepath, 'rb') as f:
            decoded_samples = f.read()
        
        expected_bytes = num_samples * 4  # 32-bit samples
        if len(decoded_samples) != expected_bytes:
            print(f"Warning: Expected {expected_bytes} bytes but got {len(decoded_samples)} bytes", file=sys.stderr)
        
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
