#!/usr/bin/env python3
"""
Entropy coding test script for FLAC LPC order 0 implementation.

This script:
1. Reads a binary file containing 32-bit integers (1D residuals)
2. Converts to mono waveform format
3. Runs through modified FLAC encoder with LPC order 0 
4. Extracts entropy-coded section using magic markers
5. Outputs entropy-coded bitstream to stdout

Can be used as a script or imported as a module.
"""

import sys
import os
import struct
import subprocess
import tempfile
import argparse
from typing import List, Optional

# Magic marker constants (must match the C code)
ENTROPY_CODING_START_MAGIC = 0xDEADBEEF
ENTROPY_CODING_END_MAGIC = 0xCAFEBABE

# Default paths
DEFAULT_FLAC_LIB = '/home/pnlong/lnac/flac_entropy_coding/src/libFLAC/.libs/libFLAC-static.a'
DEFAULT_INCLUDE_PATH = '/home/pnlong/lnac/flac_entropy_coding/include'

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
        raise

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

def encode_residuals(residuals: List[int], 
                    flac_lib_path: str = DEFAULT_FLAC_LIB,
                    include_path: str = DEFAULT_INCLUDE_PATH) -> bytes:
    """
    Encode residuals using FLAC Rice entropy coding.
    
    This is the main function interface for use as an imported module.
    Handles large arrays by chunking them into smaller pieces.
    
    Parameters
    ----------
    residuals : List[int]
        List of 32-bit integer residuals to encode
    flac_lib_path : str
        Path to FLAC static library (unused with old encoder)
    include_path : str  
        Path to FLAC include directory (unused with old encoder)
        
    Returns
    -------
    bytes
        Entropy-coded bitstream
    """
    
    if not residuals:
        return bytes()
    
    # The old encoder has limitations with large arrays (fails around 1000+ samples)
    # Split large arrays into smaller chunks that it can handle reliably
    MAX_CHUNK_SIZE = 512  # Conservative size that works reliably
    
    if len(residuals) <= MAX_CHUNK_SIZE:
        # Small array - encode directly
        return _encode_chunk(residuals)
    else:
        # Large array - split into chunks and encode with metadata
        return _encode_chunked(residuals, MAX_CHUNK_SIZE)

def _encode_chunk(residuals: List[int]) -> bytes:
    """
    Encode a single chunk of residuals using the old FLAC encoder.
    
    Parameters
    ----------
    residuals : List[int]
        List of residuals (should be reasonably small)
        
    Returns
    -------
    bytes
        Entropy-coded bitstream for this chunk
    """
    
    # Get path for the old working encoder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    helper_executable_path = os.path.join(script_dir, 'flac_rice_encode_helper')
    
    # Check if helper exists
    if not os.path.exists(helper_executable_path):
        raise RuntimeError(f"FLAC encoder helper not found: {helper_executable_path}")
    
    # Create temporary input file with residuals
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as temp_input:
        temp_input_path = temp_input.name
        # Pack residuals as little-endian 32-bit signed integers
        for residual in residuals:
            temp_input.write(struct.pack('<i', residual))
    
    try:
        # Run the old encoder: ./old_flac_rice_encode input.bin > output.bin
        # Use predictor order 0 (default Rice coding)
        result = subprocess.run([helper_executable_path, temp_input_path, '0'], 
                              check=True, capture_output=True)
        
        # The old encoder outputs entropy data directly to stdout
        entropy_data = result.stdout
        
        if not entropy_data:
            raise RuntimeError("FLAC encoder produced no output")
        
        return entropy_data
        
    except subprocess.CalledProcessError as e:
        stderr_text = e.stderr.decode("utf-8", errors="ignore") if e.stderr else "No error message"
        raise RuntimeError(f"FLAC encoding failed: {stderr_text}")
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_input_path)
        except OSError:
            pass

def _encode_chunked(residuals: List[int], chunk_size: int) -> bytes:
    """
    Encode a large array by splitting it into chunks.
    
    Format: [MAGIC][num_chunks][chunk1_size][chunk1_data_size][chunk1_data][chunk2_size][chunk2_data_size][chunk2_data]...
    
    Parameters
    ----------
    residuals : List[int]
        Large list of residuals to encode
    chunk_size : int
        Maximum size of each chunk
        
    Returns
    -------
    bytes
        Encoded data with chunking metadata
    """
    
    # Magic marker to identify chunked format
    CHUNKED_MAGIC = 0xFEEDFACE
    
    # Split into chunks
    chunks = []
    for i in range(0, len(residuals), chunk_size):
        chunk = residuals[i:i + chunk_size]
        chunks.append(chunk)
    
    # Encode each chunk
    encoded_chunks = []
    for chunk in chunks:
        encoded_chunk = _encode_chunk(chunk)
        encoded_chunks.append(encoded_chunk)
    
    # Build the chunked format
    result = bytearray()
    
    # Write magic marker (4 bytes)
    result.extend(struct.pack('<I', CHUNKED_MAGIC))
    
    # Write number of chunks (4 bytes)
    result.extend(struct.pack('<I', len(chunks)))
    
    # Write each chunk with metadata
    for chunk, encoded_chunk in zip(chunks, encoded_chunks):
        # Write chunk size (number of samples) (4 bytes)
        result.extend(struct.pack('<I', len(chunk)))
        # Write encoded data size (4 bytes)
        result.extend(struct.pack('<I', len(encoded_chunk)))
        # Write encoded data
        result.extend(encoded_chunk)
    
    return bytes(result)

def decode_residuals(encoded_data: bytes, num_samples: int) -> List[int]:
    """
    Decode FLAC Rice entropy-coded data back to residuals.
    
    Handles both single-chunk and multi-chunk formats automatically.
    
    Parameters
    ----------
    encoded_data : bytes
        The entropy-coded data
    num_samples : int
        Expected number of samples to decode
        
    Returns
    -------
    List[int]
        List of decoded residuals
    """
    
    if not encoded_data:
        return []
    
    # Check if this is chunked format (starts with magic marker)
    CHUNKED_MAGIC = 0xFEEDFACE
    
    if len(encoded_data) >= 4:
        magic = struct.unpack('<I', encoded_data[:4])[0]
        if magic == CHUNKED_MAGIC:
            return _decode_chunked(encoded_data, num_samples)
    
    # Single chunk format
    return _decode_single_chunk(encoded_data, num_samples)

def _decode_chunked(encoded_data: bytes, expected_samples: int) -> List[int]:
    """Decode multi-chunk format."""
    
    CHUNKED_MAGIC = 0xFEEDFACE
    offset = 0
    
    # Read and verify magic marker
    magic = struct.unpack('<I', encoded_data[offset:offset+4])[0]
    if magic != CHUNKED_MAGIC:
        raise RuntimeError(f"Invalid chunked magic marker: 0x{magic:08X}")
    offset += 4
    
    # Read number of chunks
    num_chunks = struct.unpack('<I', encoded_data[offset:offset+4])[0]
    offset += 4
    
    all_residuals = []
    
    for chunk_idx in range(num_chunks):
        # Read chunk metadata
        chunk_samples = struct.unpack('<I', encoded_data[offset:offset+4])[0]
        offset += 4
        
        encoded_size = struct.unpack('<I', encoded_data[offset:offset+4])[0] 
        offset += 4
        
        # Read chunk data
        chunk_data = encoded_data[offset:offset+encoded_size]
        offset += encoded_size
        
        # Decode this chunk
        chunk_residuals = _decode_single_chunk(chunk_data, chunk_samples)
        all_residuals.extend(chunk_residuals)
    
    # Verify we got the expected number of samples
    if len(all_residuals) != expected_samples:
        raise RuntimeError(f"Decoded {len(all_residuals)} samples, expected {expected_samples}")
    
    return all_residuals

def _decode_single_chunk(encoded_data: bytes, num_samples: int) -> List[int]:
    """Decode single chunk using the old FLAC decoder."""
    
    # Get path for the old working decoder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    helper_executable_path = os.path.join(script_dir, 'flac_rice_decode_helper')
    
    # Check if helper exists
    if not os.path.exists(helper_executable_path):
        raise RuntimeError(f"FLAC decoder helper not found: {helper_executable_path}")
    
    # Create temporary input file with encoded data
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as temp_input:
        temp_input_path = temp_input.name
        temp_input.write(encoded_data)
    
    try:
        # Run the old decoder: ./old_flac_rice_decode input.bin num_samples > output.bin
        result = subprocess.run([
            helper_executable_path, 
            temp_input_path, 
            str(num_samples),
            '0'  # predictor order 0 for Rice coding
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # The old decoder outputs decoded data to stdout
        decoded_data = result.stdout
        
        if len(decoded_data) != num_samples * 4:
            raise RuntimeError(f"Expected {num_samples * 4} bytes, got {len(decoded_data)} bytes")
        
        # Unpack as little-endian 32-bit signed integers
        residuals = []
        for i in range(0, len(decoded_data), 4):
            residual = struct.unpack('<i', decoded_data[i:i+4])[0]
            residuals.append(residual)
        
        return residuals
        
    except subprocess.CalledProcessError as e:
        stderr_text = e.stderr.decode("utf-8", errors="ignore") if e.stderr else "No error message"
        raise RuntimeError(f"FLAC decoding failed: {stderr_text}")
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_input_path)
        except OSError:
            pass

def main():
    """Script interface for command-line usage."""
    parser = argparse.ArgumentParser(description='FLAC entropy coding test script')
    parser.add_argument('input_file', help='Binary file containing 32-bit integers')
    parser.add_argument('--flac-lib', default=DEFAULT_FLAC_LIB, 
                        help='Path to FLAC static library')
    parser.add_argument('--include-path', default=DEFAULT_INCLUDE_PATH, 
                        help='Path to FLAC include directory')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        print(f"Error: Input file {args.input_file} does not exist", file=sys.stderr)
        sys.exit(1)
    
    try:
        # Read residuals from binary file
        print(f"Reading residuals from {args.input_file}...", file=sys.stderr)
        residuals = read_binary_residuals(args.input_file)
        print(f"Read {len(residuals)} samples", file=sys.stderr)
        
        # Warn if too many samples
        if len(residuals) > 10000:
            print(f"Warning: Number of samples ({len(residuals)}) is larger than typical FLAC block size (max 10,000)", file=sys.stderr)
        
        # Encode residuals
        entropy_section = encode_residuals(residuals, args.flac_lib, args.include_path)
        
        # Output entropy-coded bitstream to stdout
        sys.stdout.buffer.write(entropy_section)
        sys.stdout.buffer.flush()
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main() 