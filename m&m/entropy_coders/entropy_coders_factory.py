# README
# Phillip Long
# July 6, 2025

# Entropy Coder factory.

# IMPORTS
##################################################

import numpy as np

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))

from entropy_coders import EntropyCoder
from verbatim import VerbatimCoder
from naive_rice import NaiveRiceCoder
from adaptive_rice import AdaptiveRiceCoder
from flac_rice import FlacRiceCoder

##################################################


# ENTROPY CODER FACTORY
##################################################

def factory(type_: str, **kwargs) -> EntropyCoder:
    """
    Factory method for creating entropy coders.

    Parameters
    ----------
    type_ : str
        The type of entropy coder to create.
    **kwargs : dict
        Additional keyword arguments to pass to the entropy coder constructor.
    """
    if type_ == "verbatim":
        return VerbatimCoder(**kwargs)
    elif type_ == "naive_rice":
        return NaiveRiceCoder(**kwargs)
    elif type_ == "adaptive_rice":
        return AdaptiveRiceCoder(**kwargs)
    elif type_ == "flac_rice":
        return FlacRiceCoder(**kwargs)
    else:
        raise ValueError(f"Invalid entropy coder type: {type_}")

# entropy coder types
TYPES = [
    "verbatim",
    "naive_rice",
    "adaptive_rice",
    "flac_rice",
]

##################################################


# TEST FUNCTIONS
##################################################

def test() -> dict:
    """
    Test compression rates of different entropy coders on LPC residuals from real audio data.
    
    Loads audio data, generates LPC residuals, and tests the compression performance 
    of all available entropy coders.
    
    Returns
    -------
    dict
        Dictionary containing compression results for each coder type.
        Keys are coder names, values are dicts with:
        - "original_size": size of uncompressed data in bytes
        - "compressed_size": size of compressed data in bytes  
        - "compression_ratio": original_size / compressed_size
        - "space_savings": (1 - compressed_size/original_size) * 100 (%)
    """
    
    print("Testing entropy coders on LPC residuals from real audio data...")
    print("=" * 60)
        
    # use geometric distribution
    n = 4000
    np.random.seed(42)
    data = np.random.geometric(p = 0.2, size = n).astype(np.int32) - 1
    
    print(f"Generated {len(data)} samples")
    print(f"Data range: [{np.min(data)}, {np.max(data)}]")
    print(f"Mean: {np.mean(data):.2f}, Std: {np.std(data):.2f}")
    print()
    
    # calculate original size
    original_size = data.nbytes
    
    # test each entropy coder
    coder_types = ["verbatim", "naive_rice", "adaptive_rice", "flac_rice"]
    results = dict()
    
    for coder_type in coder_types:
        try:
            print(f"Testing {coder_type}...")
            
            # create coder
            coder = factory(coder_type)
            
            # encode data
            compressed_data = coder.encode(data)
            compressed_size = len(compressed_data)
            
            # test decode to verify correctness
            decoded_data = coder.decode(compressed_data, len(data))
            
            # verify data integrity
            if not np.array_equal(data, decoded_data):
                print(f"  ERROR: {coder_type} failed verification!")
                results[coder_type] = {"error": "Decode verification failed"}
                continue
            
            # calculate metrics
            compression_ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
            space_savings = (1 - (compressed_size / original_size)) * 100
            
            # store results
            results[coder_type] = {
                "original_size": original_size,
                "compressed_size": compressed_size,
                "compression_ratio": compression_ratio,
                "space_savings": space_savings
            }
            
            print(f"  Original size: {original_size:,} bytes")
            print(f"  Compressed size: {compressed_size:,} bytes")
            print(f"  Compression ratio: {compression_ratio:.2f}x")
            print(f"  Space savings: {space_savings:.1f}%")
            print()
            
        # catch errors
        except Exception as e:
            print(f"  ERROR testing {coder_type}: {str(e)}")
            results[coder_type] = {"error": str(e)}
            print()
    
    # print summary comparison
    print("COMPRESSION COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Coder':<15} {'Size (bytes)':<12} {'Ratio':<8} {'Savings':<8}")
    print("-" * 60)
    
    valid_results = {k: v for k, v in results.items() if "error" not in v}
    
    # sort by compression ratio (best first)
    sorted_results = sorted(valid_results.items(), 
                          key=lambda x: x[1]['compression_ratio'], 
                          reverse=True)
    
    for coder_type, metrics in sorted_results:
        print(f"{coder_type:<15} {metrics['compressed_size']:<12,} "
              f"{metrics['compression_ratio']:<8.2f} {metrics['space_savings']:<8.1f}%")
    
    # print any errors
    error_results = {k: v for k, v in results.items() if "error" in v}
    if error_results:
        print("\nErrors encountered:")
        for coder_type, error_info in error_results.items():
            print(f"  {coder_type}: {error_info['error']}")
    
    return results

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":
    test()

##################################################