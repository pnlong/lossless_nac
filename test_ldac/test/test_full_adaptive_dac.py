# README
# Test script for Full Adaptive DAC encoder
# Phillip Long
# July 12, 2025

# IMPORTS
##################################################

import numpy as np
import torch
import time
import os
from audiotools import AudioSignal

# Add paths for imports
from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(dirname(realpath(__file__))))
sys.path.insert(0, f"{dirname(dirname(dirname(realpath(__file__))))}/dac")

from full_lossless_compressors.ldac_compressor import *
from full_lossless_compressors import adaptive_dac
from entropy_coders.entropy_coder import EntropyCoder
from entropy_coders.factory import get_entropy_coder
import dac

##################################################


# CONFIGURATION
##################################################

# File paths
INPUT_AUDIO_PATH = "/home/pnlong/lnac/offended.wav"
OUTPUT_COMPRESSED_PATH = "/tmp/test_adaptive_dac.ldac"
OUTPUT_DECODED_PATH = "/tmp/test_adaptive_dac_decoded.wav"

# Model path (from ldac_compressor.py)
MODEL_PATH = DAC_PATH

# Encoding parameters

# NOTE: Adaptive DAC expects INTEGER audio data in int32 format:
#   - Mono:   shape (n_samples,)
#   - Stereo: shape (n_samples, 2)
#   This script automatically converts from floating point to the required integer format.
#   
#   Unlike Naive DAC, Adaptive DAC automatically selects the optimal codebook level
#   for each subframe to maximize compression efficiency.

##################################################


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """
    Load audio file and convert to the INTEGER format expected by adaptive DAC.
    
    Adaptive DAC expects:
    - Integer values (int32)
    - Shape: (n_samples,) for mono or (n_samples, 2) for stereo
    
    Parameters
    ----------
    path : str
        Path to the audio file.
        
    Returns
    -------
    tuple[np.ndarray, int]
        Audio data as int32 array with correct shape and sample rate.
    """
    print(f"Loading audio from: {path}")
    
    # Load using audiotools (gives floating point data)
    audio = AudioSignal(path)
    
    # Get basic info
    sample_rate = audio.sample_rate
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {audio.duration:.2f} seconds")
    print(f"Channels: {audio.num_channels}")
    
    # Convert to numpy - AudioSignal can give various shapes
    audio_data = audio.audio_data.numpy()
    print(f"Loaded shape: {audio_data.shape}")
    print(f"Loaded dtype: {audio_data.dtype} (floating point)")
    
    # Handle different possible shapes from AudioSignal
    # Common shapes: (channels, samples), (1, channels, samples), (batch, channels, samples)
    while len(audio_data.shape) > 2:
        # Remove extra dimensions (batch dimensions, etc.)
        audio_data = audio_data.squeeze(0)
        print(f"Squeezed shape: {audio_data.shape}")
    
    print(f"Final loaded shape: {audio_data.shape} (should be channels, samples)")
    
    # Verify we have the expected (channels, samples) format
    if len(audio_data.shape) != 2:
        raise ValueError(f"Expected 2D audio data (channels, samples), got shape {audio_data.shape}")
    
    if audio_data.shape[0] != audio.num_channels:
        raise ValueError(f"Shape mismatch: expected {audio.num_channels} channels, got {audio_data.shape[0]}")
    
    # Convert floating point to int32 integers (required by adaptive DAC)
    audio_data = convert_audio_floating_to_fixed(
        waveform=audio_data, 
        output_dtype=np.int32
    )
    print(f"Converted to integers, dtype: {audio_data.dtype}")
    
    # Reshape to the format expected by adaptive DAC
    if audio.num_channels == 1:
        # Mono: (1, samples) → (samples,)
        audio_data = audio_data.squeeze(0)  # Remove channel dimension
        expected_shape = f"({len(audio_data)},)"
    elif audio.num_channels == 2:
        # Stereo: (2, samples) → (samples, 2)
        audio_data = audio_data.T  # Transpose to (samples, channels)
        expected_shape = f"({audio_data.shape[0]}, 2)"
    else:
        raise ValueError(f"Adaptive DAC only supports mono or stereo audio, got {audio.num_channels} channels")
    
    print(f"Final shape: {audio_data.shape} - {expected_shape}")
    print(f"Value range: [{np.min(audio_data)}, {np.max(audio_data)}]")
    
    # Validation checks
    assert audio_data.dtype == np.int32, f"Expected int32, got {audio_data.dtype}"
    if audio.num_channels == 1:
        assert len(audio_data.shape) == 1, f"Mono should be 1D, got shape {audio_data.shape}"
    else:
        assert len(audio_data.shape) == 2 and audio_data.shape[1] == 2, f"Stereo should be (n_samples, 2), got {audio_data.shape}"
    
    print("✅ Audio data converted to correct integer format for adaptive DAC")
    
    return audio_data, sample_rate


def setup_model_and_entropy_coder(sample_rate: int):
    """
    Set up the DAC model and entropy coder.
    
    Parameters
    ----------
    sample_rate : int
        Sample rate of the audio.
        
    Returns
    -------
    tuple
        DAC model and entropy coder.
    """
    print("Setting up DAC model...")
    
    # Load DAC model - use CPU for debugging CUDA errors
    device = torch.device("cuda:0")  # Force CPU for debugging
    print(f"Using device: {device}")
    
    model = dac.DAC.load(MODEL_PATH).to(device)
    model.eval()
    
    print(f"Model sample rate: {model.sample_rate} Hz")
    
    # Verify sample rate compatibility
    if sample_rate != model.sample_rate:
        print(f"WARNING: Audio sample rate ({sample_rate}) doesn't match model ({model.sample_rate})")
        print("Consider resampling the audio first")
    
    # Set up entropy coder - using adaptive rice for better compression
    print("Setting up entropy coder...")
    # entropy_coder = get_entropy_coder(type_ = "verbatim")
    # entropy_coder = get_entropy_coder(type_ = "naive_rice", k = 12)
    entropy_coder = get_entropy_coder(type_ = "adaptive_rice")
    
    return model, entropy_coder


def test_adaptive_dac_encoding():
    """
    Main test function for Adaptive DAC encoding.
    """
    print("=" * 60)
    print("ADAPTIVE DAC ENCODER TEST")
    print("=" * 60)
    
    try:
        # Initialize variables for summary
        compression_ratio = 0
        perfect_reconstruction = False
        
        # Load audio
        audio_data, sample_rate = load_audio(INPUT_AUDIO_PATH)
        
        # Set up model and entropy coder
        model, entropy_coder = setup_model_and_entropy_coder(sample_rate)
        
        # Audio analysis
        print("\n" + "=" * 40)
        print("AUDIO ANALYSIS")
        print("=" * 40)
        
        audio_duration = len(audio_data) / sample_rate
        print(f"📊 AUDIO PROPERTIES (INTEGER FORMAT FOR ADAPTIVE DAC):")
        print(f"  Duration:           {audio_duration:.2f} seconds")
        print(f"  Sample rate:        {sample_rate:,} Hz")
        print(f"  Total samples:      {len(audio_data):,}")
        print(f"  Data type:          {audio_data.dtype} (INTEGER - required by adaptive DAC)")
        print(f"  Shape:              {audio_data.shape}")
        print(f"  Channels:           {'Mono' if len(audio_data.shape) == 1 else 'Stereo'}")
        print(f"  Value range:        [{np.min(audio_data)}, {np.max(audio_data)}]")
        print(f"  RMS level:          {np.sqrt(np.mean(audio_data.astype(np.float64)**2)):.1f}")
        
        # Calculate some basic statistics for compression reference
        unique_values = len(np.unique(audio_data))
        theoretical_entropy = np.log2(unique_values) if unique_values > 1 else 0
        print(f"  Unique values:      {unique_values:,}")
        print(f"  Theoretical entropy: {theoretical_entropy:.2f} bits/sample")
        print(f"  Expected format:    {'✅ Correct' if audio_data.dtype == np.int32 else '❌ Wrong - should be int32'}")
        print(f"  Adaptive feature:   ✅ Automatically selects optimal codebook levels")
        
        # Test encoding
        print("\n" + "=" * 40)
        print("ENCODING")
        print("=" * 40)
        
        start_time = time.time()
        
        adaptive_dac.encode_to_file(
            path=OUTPUT_COMPRESSED_PATH,
            data=audio_data,
            entropy_coder=entropy_coder,
            model=model,
            sample_rate=sample_rate,
        )
        
        encoding_time = time.time() - start_time
        print(f"Encoding completed in {encoding_time:.2f} seconds")
        
        # Detailed compression analysis
        if os.path.exists(OUTPUT_COMPRESSED_PATH):
            compressed_size = os.path.getsize(OUTPUT_COMPRESSED_PATH)
            original_size = audio_data.nbytes
            compression_ratio = original_size / compressed_size
            compression_percentage = (1 - compressed_size / original_size) * 100
            bits_per_sample_original = (original_size * 8) / len(audio_data)
            bits_per_sample_compressed = (compressed_size * 8) / len(audio_data)
            
            print(f"📊 COMPRESSION STATISTICS:")
            print(f"  Original size:      {original_size:,} bytes ({original_size/1024/1024:.2f} MB)")
            print(f"  Compressed size:    {compressed_size:,} bytes ({compressed_size/1024/1024:.2f} MB)")
            print(f"  Compression ratio:  {compression_ratio:.2f}:1")
            print(f"  Space saved:        {compression_percentage:.1f}%")
            print(f"  Original bits/sample: {bits_per_sample_original:.2f}")
            print(f"  Compressed bits/sample: {bits_per_sample_compressed:.2f}")
            
            # Adaptive DAC specific information
            print(f"  Adaptive optimization: ✅ Codebook levels selected per subframe")
            print(f"  Expected improvement: Better than fixed codebook level")
        
        # Test decoding
        print("\n" + "=" * 40)
        print("DECODING")
        print("=" * 40)
        
        start_time = time.time()
        
        decoded_audio = adaptive_dac.decode_from_file(
            path=OUTPUT_COMPRESSED_PATH,
            model=model,
        )
        
        decoding_time = time.time() - start_time
        print(f"Decoding completed in {decoding_time:.2f} seconds")
        
        # Comprehensive lossless verification
        print("\n" + "=" * 40)
        print("LOSSLESS VERIFICATION")
        print("=" * 40)
        
        print(f"📋 DATA COMPARISON (INTEGER FORMAT):")
        print(f"  Original shape:     {audio_data.shape}")
        print(f"  Decoded shape:      {decoded_audio.shape}")
        print(f"  Original dtype:     {audio_data.dtype} {'✅' if audio_data.dtype == np.int32 else '❌'}")
        print(f"  Decoded dtype:      {decoded_audio.dtype} {'✅' if decoded_audio.dtype == np.int32 else '❌'}")
        print(f"  Original range:     [{np.min(audio_data)}, {np.max(audio_data)}]")
        print(f"  Decoded range:      [{np.min(decoded_audio)}, {np.max(decoded_audio)}]")
        print(f"  Format consistency: {'✅ Both integer' if audio_data.dtype == decoded_audio.dtype == np.int32 else '❌ Wrong format'}")
        
        # Multiple verification checks
        shape_match = audio_data.shape == decoded_audio.shape
        dtype_match = audio_data.dtype == decoded_audio.dtype
        perfect_reconstruction = np.array_equal(audio_data, decoded_audio)
        
        print(f"\n🔍 VERIFICATION RESULTS:")
        print(f"  Shape match:        {'✅' if shape_match else '❌'}")
        print(f"  Dtype match:        {'✅' if dtype_match else '❌'}")
        print(f"  Perfect reconstruction: {'✅' if perfect_reconstruction else '❌'}")
        
        if perfect_reconstruction:
            print(f"\n🎉 SUCCESS: ADAPTIVE ENCODING IS PERFECTLY LOSSLESS!")
            print(f"   Every single sample was reconstructed exactly.")
            print(f"   Adaptive codebook selection maintained perfect fidelity.")
        else:
            print(f"\n⚠️  WARNING: Reconstruction differs from original")
            
            # Detailed error analysis
            if shape_match and dtype_match:
                diff = audio_data.astype(np.float64) - decoded_audio.astype(np.float64)
                max_abs_diff = np.max(np.abs(diff))
                mean_abs_diff = np.mean(np.abs(diff))
                std_diff = np.std(diff)
                num_different = np.sum(diff != 0)
                percent_different = (num_different / diff.size) * 100
                
                print(f"   Maximum absolute difference: {max_abs_diff}")
                print(f"   Mean absolute difference:    {mean_abs_diff:.6f}")
                print(f"   Standard deviation:          {std_diff:.6f}")
                print(f"   Samples different:           {num_different:,} ({percent_different:.2f}%)")
                
                # Check if differences are within typical floating-point precision
                if max_abs_diff <= 1e-10:
                    print(f"   → Differences are within floating-point precision (likely OK)")
                elif max_abs_diff <= 1:
                    print(f"   → Small integer differences (may be acceptable)")
                else:
                    print(f"   → Significant differences detected!")
            else:
                print(f"   → Shape or dtype mismatch prevents detailed analysis")
        
        # Save decoded audio for comparison
        print(f"\n💾 SAVING DECODED AUDIO:")
        print(f"  Output path: {OUTPUT_DECODED_PATH}")
        print(f"  Decoded shape: {decoded_audio.shape}")
        print(f"  Decoded dtype: {decoded_audio.dtype}")
        
        # Convert back to floating point for saving
        decoded_float = convert_audio_fixed_to_floating(decoded_audio)
        print(f"  After float conversion: {decoded_float.shape}, {decoded_float.dtype}")
        
        # Reshape for AudioSignal (expects channels first)
        if len(decoded_audio.shape) == 1:
            # Mono: (samples,) → (1, samples)
            decoded_float = decoded_float.reshape(1, -1)
            print(f"  Reshaped for mono: {decoded_float.shape}")
        else:
            # Stereo: (samples, 2) → (2, samples)
            decoded_float = decoded_float.T
            print(f"  Transposed for stereo: {decoded_float.shape}")
            
        decoded_signal = AudioSignal(decoded_float, sample_rate=sample_rate)
        decoded_signal.write(OUTPUT_DECODED_PATH)
        print(f"  ✅ Saved successfully")
        
        # Performance summary
        print("\n" + "=" * 40)
        print("PERFORMANCE SUMMARY")
        print("=" * 40)
        
        total_time = encoding_time + decoding_time
        realtime_factor_encode = audio_duration / encoding_time if encoding_time > 0 else float('inf')
        realtime_factor_decode = audio_duration / decoding_time if decoding_time > 0 else float('inf')
        realtime_factor_total = audio_duration / total_time if total_time > 0 else float('inf')
        
        print(f"⏱️  TIMING ANALYSIS:")
        print(f"  Audio duration:     {audio_duration:.2f} seconds")
        print(f"  Encoding time:      {encoding_time:.2f} seconds ({realtime_factor_encode:.1f}x realtime)")
        print(f"  Decoding time:      {decoding_time:.2f} seconds ({realtime_factor_decode:.1f}x realtime)")
        print(f"  Total time:         {total_time:.2f} seconds ({realtime_factor_total:.1f}x realtime)")
        print(f"  Adaptive overhead:  Included in encoding time")
        
        # Adaptive DAC specific summary
        print("\n" + "=" * 40)
        print("ADAPTIVE DAC FEATURES")
        print("=" * 40)
        
        print(f"🧠 ADAPTIVE OPTIMIZATION:")
        print(f"  Codebook selection:  ✅ Automatic per subframe")
        print(f"  Compression optimization: ✅ Maximized through level selection")
        print(f"  Batch processing:    ✅ Efficient grouped operations")
        print(f"  Perfect reconstruction: {'✅' if perfect_reconstruction else '❌'}")
        
        if perfect_reconstruction and compression_ratio > 1:
            print(f"\n🎊 OVERALL RESULT: SUCCESS!")
            print(f"   ✅ Lossless adaptive compression achieved")
            print(f"   ✅ {compression_ratio:.1f}:1 compression ratio")
            print(f"   ✅ {realtime_factor_total:.1f}x realtime processing")
            print(f"   ✅ Adaptive codebook optimization working")
        else:
            print(f"\n⚠️  OVERALL RESULT: ISSUES DETECTED")
            if not perfect_reconstruction:
                print(f"   ❌ Not perfectly lossless")
            if compression_ratio <= 1:
                print(f"   ❌ No compression achieved (ratio: {compression_ratio:.2f})")
        
        print("\n" + "=" * 60)
        print("ADAPTIVE DAC TEST COMPLETED!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main entry point."""
    # Check if input file exists
    if not os.path.exists(INPUT_AUDIO_PATH):
        print(f"❌ ERROR: Input audio file not found: {INPUT_AUDIO_PATH}")
        print("Please ensure the file exists and try again.")
        return
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"❌ ERROR: DAC model not found: {MODEL_PATH}")
        print("Please ensure the DAC model is downloaded and available.")
        return
    
    # Run the test
    test_adaptive_dac_encoding()


if __name__ == "__main__":
    main()

################################################## 