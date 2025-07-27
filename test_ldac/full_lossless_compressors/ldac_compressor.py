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
from math import log2

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

##################################################


# DAC CODES FUNCTIONS
##################################################

def dac_encode_codes_only(
    data: np.ndarray,
    model: dac.model.dac.DAC,
    sample_rate: int,
    audio_scale: float,
) -> torch.Tensor:
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

    Returns
    -------
    torch.Tensor
        DAC codes at maximum codebook level.
        If data is mono, shape: (1, MAXIMUM_CODEBOOK_LEVEL, dac_time_dimension).
        If data is stereo, shape: (2, MAXIMUM_CODEBOOK_LEVEL, dac_time_dimension).
    """

    # convert to AudioSignal format
    if len(data.shape) == 1: # mono
        data = np.expand_dims(np.expand_dims(data, axis = 0), axis = 0) # add batch and channel dimensions
    else: # stereo
        data = np.expand_dims(data.T, axis = 1) # add batch dimension so of size (2, 1, n_samples)
    data = AudioSignal(audio_path_or_array = data, sample_rate = sample_rate)
    
    # preprocess batch
    x = model.preprocess(
        audio_data = torch.from_numpy(convert_audio_fixed_to_floating(waveform = data.audio_data.numpy(), audio_scale = audio_scale)).to(model.device), # convert to float with correct audio scaling
        sample_rate = sample_rate,
    ).float()
    
    # encode
    _, codes, _, _, _ = model.encode(x)
    
    return codes
    
def dac_encode_codes_and_residuals(
    data: np.ndarray,
    model: dac.model.dac.DAC,
    sample_rate: int,
    entropy_coder: EntropyCoder,
    audio_scale: float,
    codebook_level: int,
    codes: torch.Tensor = None,
) -> List[Tuple[np.ndarray, bytes]]:
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
    codebook_level : int
        The number of codebooks to use for DAC encoding.
    codes : torch.Tensor, default = None
        DAC codes to use for encoding. If None, will be computed. Expected shape: (1, MAXIMUM_CODEBOOK_LEVEL, dac_time_dimension).

    Returns
    -------
    List[Tuple[np.ndarray, bytes]]
        List of tuples of DAC codes and entropy-encoded residuals.
        If mono, list is of length 1. If stereo, list is of length 2.
    """

    # determine if mono
    is_mono = len(data.shape) == 1
    
    # if codes are not provided, compute them
    if codes is None:
        codes = dac_encode_codes_only(
            data = data,
            model = model,
            sample_rate = sample_rate,
            audio_scale = audio_scale,
        )
    else:
        assert codebook_level <= codes.shape[1], f"Codebook level must be less than or equal to the codebook level of codes. Codebook level of codes: {codes.shape[1]}, provided codebook level: {codebook_level}."
    
    # batch decode for approximate reconstruction
    codes = codes[:, :codebook_level, :] # truncate codes to desired codebook level
    
    # batch decode for approximate reconstruction
    z = model.quantizer.from_codes(codes = codes)[0].detach() # get z from codes
    approximate_data = model.decode(z) # decode z to approximate reconstruction

    # convert back to numpy
    codes = codes.detach().cpu().numpy()
    approximate_data = approximate_data.squeeze(dim = 1).detach().cpu().numpy() # squeeze out channel dimension since it is 1
    approximate_data = approximate_data[:, :len(data)] # truncate to number of samples
    approximate_data = convert_audio_floating_to_fixed(
        waveform = approximate_data, 
        output_dtype = data.dtype,
        audio_scale = audio_scale,
    )
    

    # compute residuals
    residuals = (np.expand_dims(data, axis = 0) if is_mono else data.T) - approximate_data # ensure residuals is of shape (n_channels, n_samples)
    
    # get output
    output = [(
        codes_for_channel, # get sample from codes
        entropy_coder.encode(nums = residuals_for_channel), # entropy encode residuals
    ) for codes_for_channel, residuals_for_channel in zip(codes, residuals)]
    
    return output

def dac_decode(
    codes: List[np.ndarray],
    encoded_residuals: List[bytes],
    model: dac.model.dac.DAC,
    entropy_coder: EntropyCoder,
    n_samples: int,
    audio_scale: float,
    is_mono: bool,
) -> List[np.ndarray]:
    """
    Decode DAC codes and entropy-encoded residuals.

    Parameters
    ----------
    codes : List[np.ndarray]
        DAC codes to decode. Shape: (codebook_level, dac_time_dimension).
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
    is_mono : bool
        Whether the data was mono.

    Returns
    -------
    List[np.ndarray]
        Decoded data. Same length as list of provided codes.
    """

    # convert to tensor and add batch dimension if needed
    is_stereo_but_only_one_channel_provided = not is_mono and len(codes) == 1 # ensure consistent batch sizes across encoding and decoding
    codes_tensor = torch.stack(tensors = [torch.from_numpy(codes_for_channel) for codes_for_channel in codes], dim = 0).to(model.device) # stack codes into tensor of shape (batch, codebook_level, dac_time_dimension)
    if is_stereo_but_only_one_channel_provided:
        codes_tensor = codes_tensor.repeat(2, 1, 1) # repeat batch dimension if needed to ensure batch size of 2 for stereo

    # batch decode for approximate reconstruction
    z = model.quantizer.from_codes(codes = codes_tensor)[0].detach() # get z from codes
    approximate_data = model.decode(z) # decode z to approximate reconstruction
    
    # wrangle approximate data
    approximate_data = approximate_data.squeeze(dim = 1).detach().cpu().numpy() # remove channel dimension, which is 1
    approximate_data = approximate_data[:, :n_samples] # truncate to number of samples
    if is_stereo_but_only_one_channel_provided:
        approximate_data = approximate_data[0].unsqueeze(dim = 0) # take just first channel from batching for stereo but we only want one channel, but ensure batch dimension is still there
    approximate_data = convert_audio_floating_to_fixed(
        waveform = approximate_data, 
        output_dtype = np.int32,
        audio_scale = audio_scale,
    )
    
    # decode residuals
    residuals = [entropy_coder.decode(stream = encoded_residuals_channel, num_samples = n_samples).astype(approximate_data.dtype) for encoded_residuals_channel in encoded_residuals]

    # reconstruct data from approximate data and residuals
    reconstructed_data = [approximate_data_channel + residuals_channel for approximate_data_channel, residuals_channel in zip(approximate_data, residuals)]
    
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
    """
    assert sample_rate == model.sample_rate, f"Sample rate must match the sample rate of the model. Model sample rate: {model.sample_rate}, provided sample rate: {sample_rate}."
    assert audio_scale > 0 and audio_scale <= MAXIMUM_AUDIO_SCALE and ((log2(audio_scale) + 1) / 8) % 1 == 0, f"Audio scale must be less than or equal to {MAXIMUM_AUDIO_SCALE} and satisfy 2^((8*x)-1) = audio_scale for some integer x."
    if codebook_level is not None: # if codebook level is provided, validate it
        assert codebook_level > 0 and codebook_level <= MAXIMUM_CODEBOOK_LEVEL, f"Codebook level must be between 1 and {MAXIMUM_CODEBOOK_LEVEL}."

##################################################