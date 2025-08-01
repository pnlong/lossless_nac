# README
# Phillip Long
# July 12, 2025

# DAC Compressor.

# IMPORTS
##################################################

import numpy as np
from typing import Tuple, List
import warnings
import torch
from audiotools import AudioSignal
from math import log2, ceil
import struct

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(dirname(realpath(__file__))))
sys.path.insert(0, f"{dirname(dirname(dirname(realpath(__file__))))}/dac") # for dac import

from entropy_coders.entropy_coder import EntropyCoder
import constants
import dac

# ignore deprecation warning from pytorch
warnings.filterwarnings(action = "ignore", message = "torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm")

##################################################


# CONSTANTS
##################################################

DAC_PATH = constants.DAC_PATH # path to descript audio codec pretrained model
MAXIMUM_CODEBOOK_LEVEL = constants.MAXIMUM_CODEBOOK_LEVEL # maximum codebook level for the pretrained descript audio codec model
MAXIMUM_CODEBOOK_LEVEL_BITS = constants.MAXIMUM_CODEBOOK_LEVEL_BITS # convert into number of bits
CODEBOOK_LEVEL_DEFAULT = constants.CODEBOOK_LEVEL_DEFAULT # codebook level for descript audio codec model, use the upper bound as the default
MAXIMUM_AUDIO_SCALE = constants.MAXIMUM_AUDIO_SCALE # maximum audio scale for DAC processing
MAXIMUM_AUDIO_SCALE_BITS = constants.MAXIMUM_AUDIO_SCALE_BITS # convert into number of bits
AUDIO_SCALE_DEFAULT = constants.AUDIO_SCALE_DEFAULT # audio-appropriate scaling factor for DAC processing
BITS_PER_SAMPLE_BITS = constants.BITS_PER_SAMPLE_BITS # number of bits per sample for dtype

N_SAMPLES_BITS = 64 # number of bits for number of samples
DAC_TIME_DIMENSION_BITS = 32 # number of bits for DAC time dimension
ENCODED_RESIDUALS_SIZE_BITS = 32 # number of bits for encoded residuals size
WINDOW_DURATION_DEFAULT = 5.0
CHUNK_LENGTH_BITS = 16 # number of bits for chunk length
INPUT_DB_BITS = 32 # number of bits for input dB, has to be 32 because float32 is 32 bits

##################################################


# HELPER FUNCTIONS
##################################################

get_optimal_audio_scale = constants.get_optimal_audio_scale

get_minimum_number_of_bits_for_sample = constants.get_minimum_number_of_bits_for_sample

##################################################


# CONVERSION FUNCTIONS
##################################################

convert_audio_fixed_to_floating = constants.convert_audio_fixed_to_floating

convert_audio_floating_to_fixed = constants.convert_audio_floating_to_fixed

get_numpy_dtype_bit_size = constants.get_numpy_dtype_bit_size

get_numpy_dtype_from_bit_size = constants.get_numpy_dtype_from_bit_size

encode_audio_scale_bits = constants.encode_audio_scale_bits

decode_audio_scale_bits = constants.decode_audio_scale_bits

def encode_input_db_bits(
    input_db: float,
) -> int:
    """
    Encode input dB in bits.

    Parameters
    ----------
    input_db : float
        The input dB to encode.

    Returns
    -------
    int
        The input dB represented in bits.
    """
    return int.from_bytes(struct.pack("f", input_db), byteorder = "little")

def decode_input_db_bits(
    input_db_bits: int,
) -> float:
    """
    Decode input dB in bits.

    Parameters
    ----------
    input_db_bits : int
        The input dB represented in bits.

    Returns
    -------
    float
        The input dB.
    """
    return struct.unpack("f", input_db_bits.to_bytes(4, byteorder = "little"))[0]

##################################################


# DAC CODES FUNCTIONS
##################################################

def dac_encode_codes_only(
    data: np.ndarray,
    model: dac.model.dac.DAC,
    sample_rate: int,
    audio_scale: float,
    window_duration: float,
) -> dac.DACFile:
    """
    Encode data through DAC and return codes at maximum codebook level.
    
    Parameters
    ----------
    data : np.ndarray
        The data to encode in shape (n_samples,) if mono, (n_samples, 2) if stereo.
    model : dac.model.dac.DAC
        The DAC model to use for encoding.
    sample_rate : int
        The sample rate of the data.
    audio_scale : float
        The audio scale to use for DAC encoding.
    window_duration : float
        The window duration to use for DAC encoding.

    Returns
    -------
    dac.DACFile
        DAC file containing codes at maximum codebook level.
    """

    # convert to AudioSignal format
    if len(data.shape) == 1: # mono
        data = np.expand_dims(np.expand_dims(data, axis = 0), axis = 0) # add batch and channel dimensions
    else: # stereo
        data = np.expand_dims(data.T, axis = 0) # add batch dimension so of size (1, 2, n_samples)
    data = convert_audio_fixed_to_floating(waveform = data, audio_scale = audio_scale) # convert to float with correct audio scaling
    data = AudioSignal(audio_path_or_array = data, sample_rate = sample_rate) # convert to AudioSignal format
    
    # encode
    dac_file = model.compress(audio_path_or_signal = data, win_duration = window_duration)
    
    return dac_file
    
def dac_encode_codes_and_residuals(
    data: np.ndarray,
    model: dac.model.dac.DAC,
    sample_rate: int,
    entropy_coder: EntropyCoder,
    audio_scale: float,
    window_duration: float,
    codebook_level: int,
    dac_file: dac.DACFile = None,
) -> Tuple[dac.DACFile, List[bytes]]:
    """
    Encode data through DAC and return codes and entropy-encoded residuals.

    Parameters
    ----------
    data : np.ndarray
        The data to encode in shape (n_samples,) if mono, (n_samples, 2) if stereo.
    model : dac.model.dac.DAC
        The DAC model to use for encoding.
    sample_rate : int
        The sample rate of the data.
    entropy_coder : EntropyCoder
        The entropy coder to use.
    audio_scale : float
        The audio scale to use for DAC encoding.
    window_duration : float
        The window duration to use for DAC encoding.
    codebook_level : int
        The number of codebooks to use for DAC encoding.
    dac_file : dac.DACFile, default = None
        DAC file to use for encoding. If None, will be computed.

    Returns
    -------
    Tuple[dac.DACFile, List[bytes]]
        DAC file containing codes and entropy-encoded residuals for each channel in the DAC file.
    """

    # determine if mono
    is_mono = len(data.shape) == 1
    
    # if codes are not provided, compute them
    if dac_file is None:
        dac_file = dac_encode_codes_only(
            data = data,
            model = model,
            sample_rate = sample_rate,
            audio_scale = audio_scale,
            window_duration = window_duration,
        )
    else:
        assert codebook_level <= dac_file.codes.shape[1], f"Codebook level must be less than or equal to the codebook level of codes. Codebook level of codes: {dac_file.codes.shape[1]}, provided codebook level: {codebook_level}."
    
    # truncate codes to desired codebook level
    dac_file.codes = dac_file.codes[:, :codebook_level, :]
    
    # batch decode for approximate reconstruction
    approximate_data = model.decompress(obj = dac_file)

    # convert back to numpy
    approximate_data = approximate_data.audio_data.squeeze(dim = 0).detach().cpu().numpy() # squeeze out batch dimension since it is 1
    approximate_data = convert_audio_floating_to_fixed(
        waveform = approximate_data, 
        output_dtype = data.dtype,
        audio_scale = audio_scale,
    )
    
    # compute residuals
    residuals = (np.expand_dims(data, axis = 0) if is_mono else data.T) - approximate_data # ensure residuals is of shape (n_channels, n_samples)
    
    # get output
    encoded_residuals = [entropy_coder.encode(nums = residuals_for_channel) for residuals_for_channel in residuals]
    
    return (dac_file, encoded_residuals)

def dac_decode(
    codes: np.ndarray,
    encoded_residuals: List[bytes],
    model: dac.model.dac.DAC,
    entropy_coder: EntropyCoder,
    n_samples: int,
    audio_scale: float,
    chunk_length: int,
    input_db: float,
    padding: bool,
) -> np.ndarray:
    """
    Decode DAC codes and entropy-encoded residuals.

    Parameters
    ----------
    codes : np.ndarray
        DAC codes to decode. Shape: (n_channels, codebook_level, dac_time_dimension).
    encoded_residuals : List[bytes]
        Entropy-encoded residuals to decode.
    model : dac.model.dac.DAC
        The DAC model to use for decoding.
    entropy_coder : EntropyCoder
        The entropy coder to use.
    n_samples : int
        The number of samples to decode (the same across all channels).
    audio_scale: float
        The audio scale that was used for encoding (the same across all channels).
    chunk_length : int
        The chunk length that was used for encoding (the same across all channels).
    input_db : float
        The input dB that was used for encoding (the same across all channels).
    padding : bool
        Whether the data was padded for encoding.

    Returns
    -------
    np.ndarray
        Decoded data. Shape: (n_samples, n_channels) if len(encoded_residuals) == 1, otherwise (n_samples,).
    """

    # determine if mono
    is_mono = len(codes) == 1

    # convert to tensor
    codes_tensor = torch.from_numpy(codes) # stack codes into tensor of shape (n_channels, codebook_level, dac_time_dimension)

    # recreate DAC file
    dac_file = dac.DACFile(
        codes = codes_tensor,
        chunk_length = chunk_length,
        original_length = n_samples,
        input_db = torch.tensor([input_db], dtype = torch.float32, device = model.device),
        channels = len(codes),
        sample_rate = model.sample_rate,
        padding = padding,
        dac_version = "1.0.0",
    )

    # decode
    approximate_data = model.decompress(obj = dac_file)
    
    # wrangle approximate data
    approximate_data = approximate_data.audio_data.squeeze(dim = 0).detach().cpu().numpy() # remove batch dimension, which is 1
    approximate_data = convert_audio_floating_to_fixed(
        waveform = approximate_data, 
        output_dtype = np.int32,
        audio_scale = audio_scale,
    )
    
    # decode residuals
    residuals = [entropy_coder.decode(stream = encoded_residuals_channel, num_samples = n_samples).astype(approximate_data.dtype) for encoded_residuals_channel in encoded_residuals]

    # reconstruct data from approximate data and residuals
    assert len(approximate_data) == len(residuals), f"Approximate data and residuals must have the same number of channels. Approximate data channels: {len(approximate_data)}, residuals channels: {len(residuals)}."
    reconstructed_data = [approximate_data_channel + residuals_channel for approximate_data_channel, residuals_channel in zip(approximate_data, residuals)]

    # correct shape
    if is_mono: # if mono
        reconstructed_data = reconstructed_data[0]
    else: # if stereo
        reconstructed_data = np.stack(reconstructed_data, axis = -1) # stack channels into single array
    
    return reconstructed_data

##################################################


# DATA VALIDATION FUNCTIONS
##################################################

validate_input_data = constants.validate_input_data

validate_output_data = constants.validate_output_data

def validate_input_args(
    sample_rate: int,
    model: dac.model.dac.DAC,
    codebook_level: int = None,
    audio_scale: float = AUDIO_SCALE_DEFAULT,
    window_duration: float = WINDOW_DURATION_DEFAULT,
) -> None:
    """
    Validate input arguments.
    
    Parameters
    ----------
    sample_rate : int
        The sample rate of the data.
    model : dac.model.dac.DAC
        The DAC model to use.
    codebook_level : int, default = None
        The number of codebooks to use for DAC encoding.
    audio_scale : float, default = AUDIO_SCALE_DEFAULT
        The audio scale to use for encoding.
    window_duration : float, default = WINDOW_DURATION_DEFAULT
        The window duration to use for encoding.
    """
    assert sample_rate == model.sample_rate, f"Sample rate must match the sample rate of the model. Model sample rate: {model.sample_rate}, provided sample rate: {sample_rate}."
    assert audio_scale > 0 and audio_scale <= MAXIMUM_AUDIO_SCALE and ((log2(audio_scale) + 1) / 8) % 1 == 0, f"Audio scale must be less than or equal to {MAXIMUM_AUDIO_SCALE} and satisfy 2^((8*x)-1) = audio_scale for some integer x."
    if codebook_level is not None: # if codebook level is provided, validate it
        assert codebook_level > 0 and codebook_level <= MAXIMUM_CODEBOOK_LEVEL, f"Codebook level must be between 1 and {MAXIMUM_CODEBOOK_LEVEL}."
    assert window_duration > 0, f"Window duration must be greater than 0. Provided window duration: {window_duration}."

##################################################