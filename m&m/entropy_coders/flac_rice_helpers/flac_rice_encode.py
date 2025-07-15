#!/usr/bin/env python3
"""
Entropy coding test script for FLAC LPC order 0 implementation.

This script:
1. Reads a binary file containing 32-bit integers (1D residuals)
2. Converts to mono waveform format
3. Runs through modified FLAC encoder with LPC order 0 
4. Extracts entropy-coded section using magic markers
5. Outputs entropy-coded bitstream to stdout
"""

import sys
import os
import struct
import subprocess
import tempfile
import argparse

# Magic marker constants (must match the C code)
ENTROPY_CODING_START_MAGIC = 0xDEADBEEF
ENTROPY_CODING_END_MAGIC = 0xCAFEBABE

def read_binary_residuals(filename):
    """Read 32-bit integers from binary file."""
    try:
        with open(filename, 'rb') as f:
            data = f.read()
        
        if len(data) % 4 != 0:
            raise ValueError(f"File size {len(data)} is not a multiple of 4 bytes")
        
        # Unpack as little-endian 32-bit signed integers
        residuals = list(struct.unpack(f'<{len(data)//4}i', data))
        return residuals
    except Exception as e:
        print(f"Error reading binary file: {e}", file=sys.stderr)
        sys.exit(1)

def compile_helper_if_needed(helper_source_path, helper_executable_path, flac_lib_path, include_path):
    """Compile the C helper if it doesn't exist or if source is newer."""
    
    # Check if executable exists and is newer than source
    if (os.path.exists(helper_executable_path) and 
        os.path.exists(helper_source_path) and
        os.path.getmtime(helper_executable_path) > os.path.getmtime(helper_source_path)):
        return True  # Already compiled and up to date
    
    print("Compiling FLAC entropy helper...", file=sys.stderr)
    
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

def find_magic_markers(data):
    """Find magic markers in the binary data and extract entropy-coded section."""
    start_magic = struct.pack('>I', ENTROPY_CODING_START_MAGIC)  # Big-endian
    end_magic = struct.pack('>I', ENTROPY_CODING_END_MAGIC)      # Big-endian
    
    start_pos = data.find(start_magic)
    end_pos = data.find(end_magic)
    
    if start_pos == -1:
        print(f"Error: Start magic marker (0x{ENTROPY_CODING_START_MAGIC:08X}) not found", file=sys.stderr)
        return None
    
    if end_pos == -1:
        print(f"Error: End magic marker (0x{ENTROPY_CODING_END_MAGIC:08X}) not found", file=sys.stderr)
        return None
    
    if start_pos >= end_pos:
        print(f"Error: Start marker at position {start_pos} is not before end marker at position {end_pos}", file=sys.stderr)
        return None
    
    # Extract the section between start and end markers (EXCLUDING the markers themselves)
    entropy_start = start_pos + 4  # Skip the 4-byte start marker
    entropy_end = end_pos          # Stop before the end marker
    entropy_section = data[entropy_start:entropy_end]
    
    print(f"Found entropy-coded section: {len(entropy_section)} bytes", file=sys.stderr)
    print(f"Start marker at position: {start_pos}", file=sys.stderr)
    print(f"End marker at position: {end_pos}", file=sys.stderr)
    print(f"Extracted entropy data from position {entropy_start} to {entropy_end}", file=sys.stderr)
    
    return entropy_section

def main():
    parser = argparse.ArgumentParser(description='FLAC entropy coding test script')
    parser.add_argument('input_file', help='Binary file containing 32-bit integers')
    parser.add_argument('--flac-lib', default='/home/pnlong/lnac/flac_entropy_coding/src/libFLAC/.libs/libFLAC-static.a', 
                        help='Path to FLAC static library')
    parser.add_argument('--include-path', default='/home/pnlong/lnac/flac_entropy_coding/include', 
                        help='Path to FLAC include directory')
    
    args = parser.parse_args()
    
    # Get paths for the helper program
    script_dir = os.path.dirname(os.path.abspath(__file__))
    helper_source_path = os.path.join(script_dir, 'flac_rice_encode_helper.c')
    helper_executable_path = os.path.join(script_dir, 'flac_rice_encode_helper')
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Check if FLAC library exists
    if not os.path.exists(args.flac_lib):
        print(f"Error: FLAC library {args.flac_lib} does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Check if helper source exists
    if not os.path.exists(helper_source_path):
        print(f"Error: Helper source file {helper_source_path} does not exist", file=sys.stderr)
        sys.exit(1)
    
    # Compile helper if needed
    if not compile_helper_if_needed(helper_source_path, helper_executable_path, args.flac_lib, args.include_path):
        sys.exit(1)
    
    # Read residuals from binary file
    print(f"Reading residuals from {args.input_file}...", file=sys.stderr)
    residuals = read_binary_residuals(args.input_file)
    print(f"Read {len(residuals)} samples", file=sys.stderr)
    
    # Warn if too many samples
    if len(residuals) > 10000:
        print(f"Warning: Number of samples ({len(residuals)}) is larger than typical FLAC block size (max 10,000)", file=sys.stderr)
    
    # Create temporary directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create temporary input file (already in binary format)
        temp_input = os.path.join(temp_dir, 'input.bin')
        with open(temp_input, 'wb') as f:
            f.write(open(args.input_file, 'rb').read())
        
        # Create temporary output file
        temp_output = os.path.join(temp_dir, 'output.flac')
        
        # Run the helper program
        print("Running FLAC entropy encoder...", file=sys.stderr)
        try:
            result = subprocess.run([helper_executable_path, temp_input, temp_output], 
                                  check=True, capture_output=True, text=True)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
        except subprocess.CalledProcessError as e:
            print(f"Encoding failed: {e}", file=sys.stderr)
            if e.stderr:
                print(f"Error output: {e.stderr}", file=sys.stderr)
            sys.exit(1)
        
        # Read the output FLAC file
        with open(temp_output, 'rb') as f:
            flac_data = f.read()
        
        print(f"Generated FLAC file: {len(flac_data)} bytes", file=sys.stderr)
        
        # Extract entropy-coded section
        entropy_section = find_magic_markers(flac_data)
        if entropy_section is None:
            sys.exit(1)
        
        # Output entropy-coded bitstream to stdout
        sys.stdout.buffer.write(entropy_section)
        sys.stdout.buffer.flush()

if __name__ == '__main__':
    main() 