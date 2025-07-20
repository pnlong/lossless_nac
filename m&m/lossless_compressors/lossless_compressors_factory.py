# README
# Phillip Long
# July 6, 2025

# Lossless Compressor factory. Lossless compressors are named by the type of lossy estimator they use.

# IMPORTS
##################################################

import numpy as np
import sys
import traceback
from scipy.io import wavfile
import logging

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, f"{dirname(dirname(realpath(__file__)))}/entropy_coders")

from lossless_compressors import LosslessCompressor, BLOCK_SIZE_DEFAULT, INTERCHANNEL_DECORRELATION_DEFAULT
from entropy_coders import EntropyCoder
from naive_lpc import NaiveLPC
from adaptive_lpc import AdaptiveLPC
from flac_lpc import FlacLPC
from naive_dac import NaiveDAC
from adaptive_dac import AdaptiveDAC

##################################################


# LOSSLESS COMPRESSOR FACTORY
##################################################

def factory(type_: str, entropy_coder: EntropyCoder, block_size: int = BLOCK_SIZE_DEFAULT, interchannel_decorrelation: bool = INTERCHANNEL_DECORRELATION_DEFAULT, **kwargs) -> LosslessCompressor:
    """
    Factory method for creating lossless compressors.

    Parameters
    ----------
    type_ : str
        The type of lossless compressor to create.
    entropy_coder : EntropyCoder
        The entropy coder to use.
    block_size : int, default = BLOCK_SIZE_DEFAULT
        The block size to use for encoding.
    interchannel_decorrelation : bool, default = INTERCHANNEL_DECORRELATION_DEFAULT
        Whether to try different interchannel decorrelation schemes.
    **kwargs : dict
        Additional keyword arguments to pass to the lossless compressor constructor.
    """
    if type_ == "naive_lpc":
        return NaiveLPC(entropy_coder = entropy_coder, block_size = block_size, interchannel_decorrelation = interchannel_decorrelation, **kwargs)
    elif type_ == "adaptive_lpc":
        return AdaptiveLPC(entropy_coder = entropy_coder, block_size = block_size, interchannel_decorrelation = interchannel_decorrelation, **kwargs)
    elif type_ == "flac_lpc":
        return FlacLPC(entropy_coder = entropy_coder, block_size = block_size, interchannel_decorrelation = interchannel_decorrelation, **kwargs)
    elif type_ == "naive_dac":
        return NaiveDAC(entropy_coder = entropy_coder, block_size = block_size, interchannel_decorrelation = interchannel_decorrelation, **kwargs)
    elif type_ == "adaptive_dac":
        return AdaptiveDAC(entropy_coder = entropy_coder, block_size = block_size, interchannel_decorrelation = interchannel_decorrelation, **kwargs)
    else:
        raise ValueError(f"Invalid lossless compressor type: {type_}")

# lossless compressor types
TYPES = [
    "naive_lpc",
    "adaptive_lpc",
    "flac_lpc",
    "naive_dac",
    "adaptive_dac",
]

##################################################


# TEST FUNCTIONS
##################################################

def test_compressor(compressor_type, compressor_name, entropy_coder, test_data, interchannel_decorrelation: bool = INTERCHANNEL_DECORRELATION_DEFAULT, **kwargs):
    """Test a compressor with given test data."""
    print(f"\n{'='*60}")
    print(f"Testing {compressor_name}")
    print(f"{'='*60}")
    
    try:
        # Initialize compressor using factory
        print(f"Initializing {compressor_name}...")
        if 'dac' in compressor_type:
            # DAC compressors need special initialization
            compressor = factory(
                compressor_type, 
                entropy_coder, 
                block_size=1024,
                device="cpu",  # Use CPU to avoid CUDA issues
                interchannel_decorrelation = interchannel_decorrelation,
                **kwargs
            )
        else:
            compressor = factory(
                compressor_type, 
                entropy_coder, 
                block_size=1024,
                interchannel_decorrelation = interchannel_decorrelation,
                **kwargs
            )
        
        print(f"✓ {compressor_name} initialized successfully")
        
        # Test encoding
        print(f"Encoding test data with {compressor_name}...")
        bottleneck = compressor.encode(test_data)
        print(f"✓ Encoding successful. Bottleneck type: {type(bottleneck)}")
        
        # Test compressed size calculation
        print(f"Calculating compressed size...")
        compressed_size = compressor.get_compressed_size(bottleneck)
        original_size = test_data.nbytes
        compression_ratio = compressed_size / original_size
        print(f"✓ Original size: {original_size} bytes")
        print(f"✓ Compressed size: {compressed_size} bytes")
        print(f"✓ Compression ratio: {compression_ratio:.4f}")
        
        # Test decoding
        print(f"Decoding with {compressor_name}...")
        reconstructed = compressor.decode(bottleneck)
        print(f"✓ Decoding successful. Reconstructed shape: {reconstructed.shape}, dtype: {reconstructed.dtype}")
        
        # Test losslessness
        print(f"Verifying losslessness...")
        if np.array_equal(test_data, reconstructed):
            print(f"✓ {compressor_name} is LOSSLESS!")
        else:
            print(f"✗ {compressor_name} is NOT lossless!")
            print(f"  Max absolute difference: {np.max(np.abs(test_data - reconstructed))}")
            print(f"  Mean absolute difference: {np.mean(np.abs(test_data - reconstructed))}")
            
        return True, compression_ratio
        
    except Exception as e:
        print(f"✗ {compressor_name} FAILED with error:")
        print(f"  Error type: {type(e).__name__}")
        print(f"  Error message: {str(e)}")
        print(f"  Traceback:")
        traceback.print_exc()
        return False, None

def test(interchannel_decorrelation: bool = False) -> dict:
    """
    Test compression rates of different lossless compressors using real audio data.
    """
    print("Testing Lossless Compressors with Real Audio Data")
    print("=================================================")
    
    # Read actual audio file
    print("Reading test.wav file...")
    try:
        sample_rate, audio_data = wavfile.read('/home/pnlong/lnac/test.wav')
        print(f"✓ Successfully read test.wav")
        print(f"  Sample rate: {sample_rate} Hz")
        print(f"  Data shape: {audio_data.shape}")
        print(f"  Data type: {audio_data.dtype}")
        print(f"  Data range: [{np.min(audio_data)}, {np.max(audio_data)}]")
        
        # Convert to int32 for consistency
        if audio_data.dtype != np.int32:
            audio_data = audio_data.astype(np.int32)
            print(f"✓ Converted to int32")
        
        # Create test cases
        test_cases = {}
        if len(audio_data.shape) == 1:
            # Mono audio
            test_cases["Real_Audio_Mono"] = audio_data
        else:
            # Stereo audio - test both mono (left channel) and stereo
            test_cases["Real_Audio_Mono"] = audio_data[:, 0]  # Left channel only
            test_cases["Real_Audio_Stereo"] = audio_data      # Both channels
        
        print(f"✓ Created test cases: {list(test_cases.keys())}")
        
    except Exception as e:
        print(f"✗ Failed to read test.wav: {e}")
        traceback.print_exc()
        return {}
    
    # Import entropy coder
    try:
        print("Importing adaptive rice entropy coder...")
        from adaptive_rice import AdaptiveRiceCoder
        entropy_coder = AdaptiveRiceCoder()
        print("✓ Adaptive rice entropy coder imported successfully")
    except Exception as e:
        print(f"✗ Failed to import entropy coder: {e}")
        traceback.print_exc()
        return {}
    
    # Define compressors to test
    compressors_to_test = [
        ("naive_lpc", "NaiveLPC", {"order": 9}),
        ("adaptive_lpc", "AdaptiveLPC", {}),
        ("flac_lpc", "FlacLPC", {}),
        ("naive_dac", "NaiveDAC", {"codebook_level": 3, "batch_size": 8}),
        ("adaptive_dac", "AdaptiveDAC", {"batch_size": 8}),
    ]
    
    # Test each compressor with each test case
    results = {}
    
    for test_name, test_data in test_cases.items():
        print(f"\n{'#'*80}")
        print(f"TESTING WITH {test_name.upper()} DATA")
        print(f"Data shape: {test_data.shape}, dtype: {test_data.dtype}")
        print(f"Data range: [{np.min(test_data)}, {np.max(test_data)}]")
        print(f"{'#'*80}")
        
        results[test_name] = {}
        
        for compressor_type, compressor_name, kwargs in compressors_to_test:
            success, compression_ratio = test_compressor(
                compressor_type, compressor_name, entropy_coder, test_data, interchannel_decorrelation = interchannel_decorrelation, **kwargs
            )
            results[test_name][compressor_name] = {
                'success': success,
                'compression_ratio': compression_ratio
            }
    
    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*80}")
    
    for test_name, test_results in results.items():
        print(f"\n{test_name} Data Results:")
        print("-" * 40)
        for compressor_name, result in test_results.items():
            status = "✓ PASS" if result['success'] else "✗ FAIL"
            ratio = f" (ratio: {result['compression_ratio']:.4f})" if result['compression_ratio'] else ""
            print(f"  {compressor_name:<15}: {status}{ratio}")
    
    print(f"\n{'='*80}")
    print("Test completed!")
    
    return results

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":
    # Enable info logging (disable debug for cleaner output)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("Running lossless compressor tests with real audio data...")
    print("Interchannel decorrelation: OFF")
    print()
    test(interchannel_decorrelation=False)

##################################################