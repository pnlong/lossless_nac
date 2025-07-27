# README
# Phillip Long
# July 12, 2025

# Naive DAC Compressor.

# IMPORTS
##################################################

import numpy as np
from typing import List, Tuple
import torch

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))
sys.path.insert(0, f"{dirname(dirname(dirname(realpath(__file__))))}/dac") # for dac import

from ldac_compressor import *
from entropy_coders.entropy_coder import EntropyCoder
from entropy_coders.serialize import serialize, deserialize, SERIALIZED_ENTROPY_CODER_BITS
import bitstream
import dac

##################################################


# BOTTLENECK TYPE
##################################################

# type of bottleneck frames
BOTTLENECK_FRAME_TYPE = Tuple[np.ndarray, bytes] # bottleneck frame type is a tuple of the DAC codes and encoded residuals

# type of bottleneck
BOTTLENECK_TYPE = Tuple[int, int, int, int, int, List[BOTTLENECK_FRAME_TYPE]] # bottleneck type is a tuple of the codebook level, number of samples, DAC time dimension, audio scale bits, serialized entropy coder, and list of frames

##################################################


# SIZE CALCULATION FUNCTIONS
##################################################

def get_compressed_frame_size(
    bottleneck_frame: BOTTLENECK_FRAME_TYPE,
) -> float:
    """
    Get the size of a compressed frame in bytes.
    
    Parameters
    ----------
    bottleneck_frame : BOTTLENECK_FRAME_TYPE
        The compressed frame as (DAC codes, encoded residuals).
        
    Returns
    -------
    float
        The size of the compressed frame in bytes
    """

    # unpack bottleneck frame
    codes, encoded_residuals = bottleneck_frame

    # add size for DAC codes
    total_size = BITS_PER_SAMPLE_BITS # get_numpy_dtype_bit_size(dtype = codes.dtype) # number of bits per sample for dtype
    bits_per_sample = get_minimum_number_of_bits_for_sample(sample = codes.max())
    total_size += bits_per_sample * codes.size

    # add size for encoded residuals
    total_size += ENCODED_RESIDUALS_SIZE_BITS # 4 bytes for the number of bytes for encoded residuals as 32 bit unsigned integer
    total_size += len(encoded_residuals) * 8

    # convert total_size to bytes
    total_size /= 8
    
    return total_size

def get_compressed_bottleneck_size(
    bottleneck: BOTTLENECK_TYPE,
) -> float:
    """
    Get the size of a compressed bottleneck in bytes.

    Parameters
    ----------
    bottleneck : BOTTLENECK_TYPE
        The compressed bottleneck.
        
    Returns
    -------
    float
        The size of the compressed bottleneck in bytes
    """

    # unpack bottleneck
    codebook_level, n_samples, dac_time_dimension, audio_scale_bits, serialized_entropy_coder, frames = bottleneck

    # add size for codebook level
    total_size = MAXIMUM_CODEBOOK_LEVEL_BITS / 8

    # add size for number of samples
    total_size += N_SAMPLES_BITS / 8
    
    # add size for DAC time dimension
    total_size += DAC_TIME_DIMENSION_BITS / 8
    
    # add size for audio scale bits
    total_size += MAXIMUM_AUDIO_SCALE_BITS / 8
    
    # add size for serialized entropy coder
    total_size += SERIALIZED_ENTROPY_CODER_BITS / 8 # 1 byte for serialized entropy coder

    # add size for bit for number of subframes in frame
    total_size += 1 / 8 # boolean for whether there are two frames (stereo) or one (mono)
    
    # add size for each frame
    for frame in frames:
        total_size += get_compressed_frame_size(bottleneck_frame = frame)

    return total_size

##################################################


# MAIN ENCODE AND DECODE FUNCTIONS
##################################################

def encode(
    data: np.ndarray,
    entropy_coder: EntropyCoder,
    model: dac.model.dac.DAC,
    sample_rate: int,
    codebook_level: int = CODEBOOK_LEVEL_DEFAULT,
    audio_scale: float = None,
) -> BOTTLENECK_TYPE:
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
    codebook_level : int, default = CODEBOOK_LEVEL_DEFAULT
        The number of codebooks to use for DAC encoding.
    audio_scale : float, default = None
        The audio scale to use for encoding. If None, will be calculated using the data. If provided, will be used as is.

    Returns
    -------
    BOTTLENECK_TYPE
        The bottleneck.
    """

    # calculate audio scale if not provided
    if audio_scale is None:
        audio_scale = get_optimal_audio_scale(waveform = data)
    
    # ensure input is valid
    validate_input_data(data = data)
    validate_input_args(
        sample_rate = sample_rate,
        model = model,
        codebook_level = codebook_level,
        audio_scale = audio_scale,
    )
    assert codebook_level > 0, "Codebook level must be positive for Naive DAC."
    
    # batch encode frames
    with torch.no_grad():
        frames = dac_encode_codes_and_residuals(
            data = data,
            model = model,
            sample_rate = sample_rate,
            entropy_coder = entropy_coder,
            audio_scale = audio_scale,
            codebook_level = codebook_level,
        )

    # pack bottleneck
    n_samples = len(data)
    audio_scale_bits = encode_audio_scale_bits(audio_scale = audio_scale)
    dac_time_dimension = frames[0][0].shape[-1] # get dac time dimension from first frame
    serialized_entropy_coder = serialize(entropy_coder = entropy_coder)
    bottleneck = (
        codebook_level,
        n_samples,
        dac_time_dimension,
        audio_scale_bits,
        serialized_entropy_coder,
        frames,
    )
    
    return bottleneck

def decode(
    bottleneck: BOTTLENECK_TYPE,
    model: dac.model.dac.DAC,
) -> np.ndarray:
    """
    Decode the bottleneck into the original data.

    Parameters
    ----------
    bottleneck : BOTTLENECK_TYPE
        The bottleneck to decode.
    model : dac.model.dac.DAC
        The DAC model to use.

    Returns
    -------
    np.ndarray
        The decoded original data.
    """

    # unpack bottleneck
    codebook_level, n_samples, dac_time_dimension, audio_scale_bits, serialized_entropy_coder, frames = bottleneck
    audio_scale = decode_audio_scale_bits(audio_scale_bits = audio_scale_bits)
    entropy_coder = deserialize(header = serialized_entropy_coder)

    # prepare data for batch decode frames
    codes = [frame[0] for frame in frames] # get codes from frames
    encoded_residuals = [frame[1] for frame in frames] # get encoded residuals from frames
    is_mono = len(frames) == 1 # check if there is only one frame (mono)

    # batch decode frames
    with torch.no_grad():
        reconstructed_data = dac_decode(
            codes = codes,
            encoded_residuals = encoded_residuals,
            model = model,
            entropy_coder = entropy_coder,
            n_samples = n_samples,
            audio_scale = audio_scale,
            is_mono = is_mono,
        )
    
    # convert into a single numpy array
    if is_mono:
        reconstructed_data = reconstructed_data[0]
    else:
        reconstructed_data = np.stack(reconstructed_data, axis = 1) # stack frames for stereo

    # ensure output is valid
    validate_output_data(reconstructed_data = reconstructed_data)
    
    return reconstructed_data

##################################################


# WRITE BOTTLENECK FUNCTIONS
##################################################

def write_frame(
    frame: BOTTLENECK_FRAME_TYPE,
    bitstream: bitstream.BitOutputStream,
) -> None:
    """
    Write a frame to a bitstream.

    Parameters
    ----------
    frame : BOTTLENECK_FRAME_TYPE
        The frame to write.
    bitstream : bitstream.BitOutputStream
        The bitstream to write to.

    Returns
    -------
    None
    """

    # unpack frame
    codes, encoded_residuals = frame
    
    # write codes array
    codes_bits_per_sample = get_minimum_number_of_bits_for_sample(sample = codes.max()) # get_numpy_dtype_bit_size(dtype = codes.dtype)
    bitstream.write_bits(bits = codes_bits_per_sample, n = BITS_PER_SAMPLE_BITS) # number of bits per sample for dtype
    for code in codes.flatten():
        bitstream.write_bits(bits = int(code), n = codes_bits_per_sample)
    
    # write encoded residuals
    bitstream.write_bits(bits = len(encoded_residuals), n = ENCODED_RESIDUALS_SIZE_BITS) # number of bytes for encoded residuals as 32 bit unsigned integer
    for byte in encoded_residuals:
        bitstream.write_bits(bits = byte, n = 8)

    return

def write_bottleneck(
    bottleneck: BOTTLENECK_TYPE,
    path: str,
) -> None:
    """
    Write the bottleneck to a file.

    Parameters
    ----------
    bottleneck : BOTTLENECK_TYPE
        The bottleneck to write.
    path : str
        The path to the file to write.

    Returns
    -------
    None
    """

    # unpack bottleneck
    expected_size = get_compressed_bottleneck_size(bottleneck = bottleneck)
    codebook_level, n_samples, dac_time_dimension, audio_scale_bits, serialized_entropy_coder, frames = bottleneck

    # create bitstream
    bit_output = bitstream.BitOutputStream(path = path, buffer_size = int(expected_size * 1.2)) # buffer size is 1.2x the size of the bottleneck to avoid reallocating memory

    # write codebook level
    bit_output.write_bits(bits = codebook_level, n = MAXIMUM_CODEBOOK_LEVEL_BITS)

    # write number of samples
    bit_output.write_bits(bits = n_samples, n = N_SAMPLES_BITS)

    # write dac time dimension
    bit_output.write_bits(bits = dac_time_dimension, n = DAC_TIME_DIMENSION_BITS)
    
    # write audio scale bits
    bit_output.write_bits(bits = audio_scale_bits, n = MAXIMUM_AUDIO_SCALE_BITS)

    # write serialized entropy coder
    bit_output.write_bits(bits = serialized_entropy_coder, n = SERIALIZED_ENTROPY_CODER_BITS)
    
    # write bit for whether there are two frames (stereo) or one (mono)
    bit_output.write_bit(bit = len(frames) == 2)

    # align to byte
    bit_output.align_to_byte()  
    
    # write each frame (each frame is a tuple of the DAC codes and encoded residuals)
    for frame in frames:
        write_frame(frame = frame, bitstream = bit_output)
    
    # close bitstream (writes to file)
    bit_output.flush()
    bit_output.close()

    return
        
##################################################


# READ BOTTLENECK FUNCTIONS
##################################################

def read_frame(
    bitstream: bitstream.BitInputStream,
    codebook_level: int,
    dac_time_dimension: int,
) -> BOTTLENECK_FRAME_TYPE:
    """
    Read a frame from a bitstream.

    Parameters
    ----------
    bitstream : bitstream.BitInputStream
        The bitstream to read from.
    codebook_level : int
        The number of codebooks to use for DAC encoding.
    dac_time_dimension : int
        The DAC time dimension.

    Returns
    -------
    BOTTLENECK_FRAME_TYPE
        The frame.
    """

    # read codes array
    codes_bits_per_sample = bitstream.read_bits(n = BITS_PER_SAMPLE_BITS)
    codes = [bitstream.read_bits(n = codes_bits_per_sample) for _ in range(codebook_level * dac_time_dimension)]
    codes = np.array(codes).astype(np.int64) # np.array(codes).astype(get_numpy_dtype_from_bit_size(bit_size = codes_bits_per_sample))
    codes = codes.reshape(codebook_level, dac_time_dimension)

    # read encoded residuals
    encoded_residuals_len = bitstream.read_bits(n = ENCODED_RESIDUALS_SIZE_BITS)
    encoded_residuals = bytes([bitstream.read_bits(n = 8) for _ in range(encoded_residuals_len)])
    
    # compile frame
    frame = (codes, encoded_residuals)
    
    return frame

def read_bottleneck(
    path: str,
) -> BOTTLENECK_TYPE:
    """
    Read the bottleneck from a file.

    Parameters
    ----------
    path : str
        The path to the file to read.

    Returns
    -------
    BOTTLENECK_TYPE
        The bottleneck.
    """

    # create bitstream
    bit_input = bitstream.BitInputStream(path = path)

    # read codebook level
    codebook_level = bit_input.read_bits(n = MAXIMUM_CODEBOOK_LEVEL_BITS)

    # read number of samples
    n_samples = bit_input.read_bits(n = N_SAMPLES_BITS)

    # read dac time dimension
    dac_time_dimension = bit_input.read_bits(n = DAC_TIME_DIMENSION_BITS)

    # read audio scale bits
    audio_scale_bits = bit_input.read_bits(n = MAXIMUM_AUDIO_SCALE_BITS)

    # read serialized entropy coder
    serialized_entropy_coder = bit_input.read_bits(n = SERIALIZED_ENTROPY_CODER_BITS)
    
    # read bit for whether there are two frames (stereo) or one (mono)
    is_stereo = bit_input.read_bit()

    # align to byte
    bit_input.align_to_byte()
    
    # read each frame
    frames = [None] * (2 if is_stereo else 1)
    for i in range(len(frames)):
        frames[i] = read_frame(
            bitstream = bit_input,
            codebook_level = codebook_level,
            dac_time_dimension = dac_time_dimension,
        )

    # pack bottleneck
    bottleneck = (codebook_level, n_samples, dac_time_dimension, audio_scale_bits, serialized_entropy_coder, frames)
    
    return bottleneck

##################################################


# ENCODE AND DECODE TO FILE FUNCTIONS
##################################################

def encode_to_file(
    path: str,
    data: np.ndarray,
    entropy_coder: EntropyCoder,
    model: dac.model.dac.DAC,
    sample_rate: int,
    codebook_level: int = CODEBOOK_LEVEL_DEFAULT,
    audio_scale: float = None,
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
    codebook_level : int, default = CODEBOOK_LEVEL_DEFAULT
        The number of codebooks to use for DAC encoding.
    audio_scale : float, default = None
        The audio scale to use for encoding. If None, will be calculated using the data. If provided, will be used as is.

    Returns
    -------
    None
    """

    # get bottleneck
    bottleneck = encode(
        data = data,
        entropy_coder = entropy_coder,
        model = model,
        sample_rate = sample_rate,
        codebook_level = codebook_level,
        audio_scale = audio_scale,
    )

    # write bottleneck to file
    write_bottleneck(
        bottleneck = bottleneck,
        path = path,
    )

    return

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

    # read bottleneck from file
    bottleneck = read_bottleneck(
        path = path,
    )

    # decode bottleneck
    data = decode(
        bottleneck = bottleneck,
        model = model,
    )

    return data

##################################################