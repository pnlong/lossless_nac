# README
# Phillip Long
# July 21, 2025

# LDAC lossless compressor.

# IMPORTS
##################################################

import numpy as np
from typing import Union

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))
sys.path.insert(0, f"{dirname(dirname(realpath(__file__)))}/entropy_coders")
sys.path.insert(0, f"{dirname(dirname(dirname(realpath(__file__))))}/dac") # for dac import

from ldac_compressor import *
import naive_dac
import adaptive_dac
import bitstream
from entropy_coder import EntropyCoder
import dac

##################################################


# MAIN ENCODE AND DECODE FUNCTIONS
##################################################

def encode(
    data: np.ndarray,
    entropy_coder: EntropyCoder,
    model: dac.model.dac.DAC,
    sample_rate: int,
    codebook_level: int = 0, # default to 0 for adaptive dac
    audio_scale: float = None,
    block_size: int = BLOCK_SIZE_DEFAULT,
    batch_size: int = BATCH_SIZE_DEFAULT,
) -> Union[naive_dac.BOTTLENECK_TYPE, adaptive_dac.BOTTLENECK_TYPE]:
    """
    Encode the original data into the bottleneck.

    Parameters
    ----------
    data : np.ndarray
        The data to encode. Shape: (n_samples,) for mono, (n_samples, 2) for stereo.
    entropy_coder : EntropyCoder
        The entropy coder to use.
    model : dac.model.dac.DAC
        The DAC model to use.
    sample_rate : int
        The sample rate of the data.
    codebook_level : int, default = 0
        The number of codebooks to use for DAC encoding. Default to 0 for Adaptive DAC.
    audio_scale : float, default = None
        The audio scale to use for encoding. If None, will be calculated using the data. If provided, will be used as is.
    block_size : int, default = BLOCK_SIZE_DEFAULT
        The block size to use for encoding.
    batch_size : int, default = BATCH_SIZE_DEFAULT
        The batch size to use for encoding.

    Returns
    -------
    Union[naive_dac.BOTTLENECK_TYPE, adaptive_dac.BOTTLENECK_TYPE]
        The bottleneck.
    """

    # encode using adaptive dac if codebook level is 0
    if codebook_level == 0:
        return adaptive_dac.encode(
            data = data,
            entropy_coder = entropy_coder,
            model = model,
            sample_rate = sample_rate,
            audio_scale = audio_scale,
            block_size = block_size,
            batch_size = batch_size,
        )

    # if valid codebook level, encode using naive dac
    else:
        return naive_dac.encode(
            data = data,
            entropy_coder = entropy_coder,
            model = model,
            sample_rate = sample_rate,
            codebook_level = codebook_level,
            audio_scale = audio_scale,
            block_size = block_size,
            batch_size = batch_size,
        )

def decode(
    bottleneck: Union[naive_dac.BOTTLENECK_TYPE, adaptive_dac.BOTTLENECK_TYPE],
    model: dac.model.dac.DAC,
) -> np.ndarray:
    """
    Decode the bottleneck into the original data.

    Parameters
    ----------
    bottleneck : Union[naive_dac.BOTTLENECK_TYPE, adaptive_dac.BOTTLENECK_TYPE]
        The bottleneck to decode.
    model : dac.model.dac.DAC
        The DAC model to use.

    Returns
    -------
    np.ndarray
        The decoded original data.
    """

    # unpack bottleneck
    codebook_level = bottleneck[0]
    
    # decode using adaptive dac if codebook level is 0
    if codebook_level == 0:
        return adaptive_dac.decode(
            bottleneck = bottleneck,
            model = model,
        )

    # if valid codebook level, decode using naive dac
    else:
        return naive_dac.decode(
            bottleneck = bottleneck,
            model = model,
        )

##################################################


# ENCODE AND DECODE TO FILE FUNCTIONS
##################################################

def encode_to_file(
    path: str,
    data: np.ndarray,
    entropy_coder: EntropyCoder,
    model: dac.model.dac.DAC,
    sample_rate: int,
    codebook_level: int = 0, # default to 0 for adaptive dac
    audio_scale: float = None,
    block_size: int = BLOCK_SIZE_DEFAULT,
    batch_size: int = BATCH_SIZE_DEFAULT,
) -> None:
    """
    Encode the data to a file.

    Parameters
    ----------
    path : str
        The path to the file to write.
    data : np.ndarray
        The data to encode.
    entropy_coder : EntropyCoder
        The entropy coder to use.
    model : dac.model.dac.DAC
        The DAC model to use.
    sample_rate : int
        The sample rate of the data.
    codebook_level : int, default = 0
        The number of codebooks to use for DAC encoding. Default to 0 for Adaptive DAC.
    audio_scale : float, default = None
        The audio scale to use for encoding. If None, will be calculated using the data. If provided, will be used as is.
    block_size : int, default = BLOCK_SIZE_DEFAULT
        The block size to use for encoding.
    batch_size : int, default = BATCH_SIZE_DEFAULT
        The batch size to use for encoding.

    Returns
    -------
    None
    """

     # encode using adaptive dac if codebook level is 0
    if codebook_level == 0:
        return adaptive_dac.encode_to_file(
            path = path,
            data = data,
            entropy_coder = entropy_coder,
            model = model,
            sample_rate = sample_rate,
            audio_scale = audio_scale,
            block_size = block_size,
            batch_size = batch_size,
        )

    # if valid codebook level, encode using naive dac
    else:
        return naive_dac.encode_to_file(
            path = path,
            data = data,
            entropy_coder = entropy_coder,
            model = model,
            sample_rate = sample_rate,
            codebook_level = codebook_level,
            audio_scale = audio_scale,
            block_size = block_size,
            batch_size = batch_size,
        )

def decode_from_file(
    path: str,
    model: dac.model.dac.DAC,
) -> np.ndarray:
    """
    Decode the data from a file.

    Parameters
    ----------
    path : str
        The path to the file to read.
    model : dac.model.dac.DAC
        The DAC model to use.

    Returns
    -------
    np.ndarray
        The decoded data.
    """

    # parse codebook level from file
    with open(path, mode = "rb") as f:
        first_byte = f.read(1) # read first byte
    input_stream = bitstream.BitInputStream(stream = first_byte)
    codebook_level = input_stream.read_bits(n = MAXIMUM_CODEBOOK_LEVEL_BITS)
    del first_byte, input_stream

    # decode using adaptive dac if codebook level is 0
    if codebook_level == 0:
        return adaptive_dac.decode_from_file(
            path = path,
            model = model,
        )

    # if valid codebook level, decode using naive dac
    else:
        return naive_dac.decode_from_file(
            path = path,
            model = model,
        )

##################################################