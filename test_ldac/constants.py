# README
# Phillip Long
# July 26, 2025

# Constants for LDAC.

# IMPORTS
##################################################

import numpy as np
from math import ceil, log2

##################################################


# CONSTANTS
##################################################

DAC_PATH = "/home/pnlong/.cache/descript/dac/weights_44khz_8kbps_0.0.1.pth" # path to descript audio codec pretrained model
ZAC_PATH = "/data3/pnlong/zachdac/latest/dac/weights.pth" # path to zachdac pretrained model
MAXIMUM_CODEBOOK_LEVEL = 9 # maximum codebook level for the pretrained descript audio codec model
MAXIMUM_CODEBOOK_LEVEL_BITS = ceil(log2(MAXIMUM_CODEBOOK_LEVEL + 1)) # convert into number of bits
CODEBOOK_LEVEL_DEFAULT = MAXIMUM_CODEBOOK_LEVEL # codebook level for descript audio codec model, use the upper bound as the default
MAXIMUM_AUDIO_SCALE = 2 ** (32 - 1) # maximum audio scale for DAC processing
MAXIMUM_AUDIO_SCALE_BITS = ceil(log2((log2(MAXIMUM_AUDIO_SCALE) + 1) / 8)) # convert into number of bits
AUDIO_SCALE_DEFAULT = 2 ** (16 - 1)  # audio-appropriate scaling factor for DAC processing
BITS_PER_SAMPLE_BITS = 8 # number of bits per sample for dtype

##################################################


# HELPER FUNCTIONS
##################################################

def get_optimal_audio_scale(waveform: np.ndarray) -> float:
    """
    Get the optimal audio scale for a waveform.
    
    Parameters
    ----------
    waveform : np.ndarray
        The waveform to get the optimal audio scale for.
        
    Returns
    -------
    float
        The optimal audio scale.
    """
    max_abs = np.max(np.abs(waveform))
    optimal_audio_scale = ceil((log2(max_abs) + 1) / 8)
    optimal_audio_scale = 2 ** ((8 * optimal_audio_scale) - 1)
    optimal_audio_scale = float(optimal_audio_scale)
    return optimal_audio_scale

def get_minimum_number_of_bits_for_sample(sample: int) -> int:
    """
    Get the minimum number of bits required to represent a sample.
    
    Parameters
    ----------
    sample : int
        The sample to get the minimum number of bits for.

    Returns
    -------
    int
        The minimum number of bits required to represent the sample.
    """
    return ceil(log2(sample + 1))

def convert_audio_fixed_to_floating(waveform: np.ndarray, audio_scale: float = AUDIO_SCALE_DEFAULT) -> np.ndarray:
    """
    Convert fixed-point audio to floating-point using appropriate audio scaling.

    Parameters
    ----------
    waveform : np.ndarray
        The waveform to convert.
    audio_scale : float, default = AUDIO_SCALE_DEFAULT
        The audio scale to use.
    
    Returns
    -------
    np.ndarray
        The converted waveform.
    """
    return waveform.astype(np.float32) / audio_scale

def convert_audio_floating_to_fixed(waveform: np.ndarray, output_dtype: type = np.int32, audio_scale: float = AUDIO_SCALE_DEFAULT) -> np.ndarray:
    """
    Convert floating-point audio to fixed-point using appropriate audio scaling.
    
    Parameters
    ----------
    waveform : np.ndarray
        The waveform to convert.
    output_dtype : type, default = np.int32
        The output dtype to use.
    audio_scale : float, default = AUDIO_SCALE_DEFAULT
        The audio scale to use.

    Returns
    -------
    np.ndarray
        The converted waveform
    """
    return np.round(waveform * audio_scale).astype(output_dtype)

def get_numpy_dtype_bit_size(dtype: np.dtype) -> int:
    """
    Get the number of bits per sample for a numpy dtype.
    
    Parameters
    ----------
    dtype : np.dtype
        The numpy dtype.
        
    Returns
    -------
    int
        Number of bits per sample.
    """
    return dtype.itemsize * 8

def get_numpy_dtype_from_bit_size(bit_size: int) -> np.dtype:
    """
    Get a numpy dtype from the number of bits per sample.
    
    Parameters
    ----------
    bit_size : int
        Number of bits per sample.
        
    Returns
    -------
    np.dtype
        Appropriate numpy dtype.
    """
    return np.dtype(f"int{bit_size}")

def encode_audio_scale_bits(audio_scale: float = AUDIO_SCALE_DEFAULT) -> int:
    """
    Encode audio scale as 2 bits (0-3) where actual audio scale = 2^((8*(x+1))-1).
    
    Parameters
    ----------
    audio_scale : float
        The actual audio scale to encode.
        
    Returns
    -------
    int
        Encoded audio scale bits (0-3).
    """
    audio_scale_bits = int((log2(audio_scale) + 1) / 8) - 1
    return audio_scale_bits

def decode_audio_scale_bits(audio_scale_bits: int = int(log2(AUDIO_SCALE_DEFAULT) / 8) - 1) -> float:
    """
    Decode audio scale from 2 bits (0-3) where actual audio scale = 2^((8*(x+1))-1).
    
    Parameters
    ----------
    audio_scale_bits : int, default = int(log2(AUDIO_SCALE_DEFAULT) / 8) - 1
        Encoded audio scale bits (0-3).
        
    Returns
    -------
    float
        Actual audio scale.
    """
    return 2 ** ((8 * (audio_scale_bits + 1)) - 1)

##################################################


# DATA VALIDATION FUNCTIONS
##################################################

def validate_input_data(data: np.ndarray) -> None:
    """
    Validate input data format and type.
    
    Parameters
    ----------
    data : np.ndarray
        The data to validate.
        
    Raises
    ------
    AssertionError
        If data format is invalid.
    """
    
    assert len(data.shape) == 1 or len(data.shape) == 2, "Data must be 1D or 2D."
    if len(data.shape) == 2:
        assert data.shape[1] == 2, "Data must be 2D with 2 channels."
    assert data.dtype == np.int32, "Data must be int32."

def validate_output_data(reconstructed_data: np.ndarray) -> None:
    """
    Validate output data format and type.
    
    Parameters
    ----------
    reconstructed_data : np.ndarray
        The reconstructed data to validate.
        
    Raises
    ------
    AssertionError
        If data format is invalid.
    """
    
    assert len(reconstructed_data.shape) == 1 or len(reconstructed_data.shape) == 2, "Reconstructed data must be 1D or 2D."
    if len(reconstructed_data.shape) == 2:
        assert reconstructed_data.shape[1] == 2, "Reconstructed data must be 2D with 2 channels."
    assert reconstructed_data.dtype == np.int32, "Reconstructed data must be int32."

##################################################


# TESTING CONSTANTS
##################################################

# filepaths
INPUT_FILEPATH = "/deepfreeze/pnlong/lnac/test_data/musdb18_preprocessed-44100/data.csv"
OUTPUT_DIR = "/deepfreeze/pnlong/lnac/eval/ldac_new2"

# output file
NA_STRING = "NA"

##################################################