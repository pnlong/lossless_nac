# README
# Phillip Long
# July 12, 2025

# Adaptive DAC Compressor.

# IMPORTS
##################################################

import numpy as np
from typing import List, Tuple, Union
import torch
from copy import deepcopy

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
BOTTLENECK_FRAME_TYPE = Tuple[int, np.ndarray, bytes] # bottleneck frame type is a tuple of the codebook level, DAC codes, and encoded residuals

# type of bottleneck
BOTTLENECK_TYPE = Tuple[int, int, int, int, int, int, int, bool, List[BOTTLENECK_FRAME_TYPE]] # bottleneck type is a tuple of the codebook level, number of samples, DAC time dimension, audio scale bits, serialized entropy coder, chunk length, input dB bits, padding, and list of frames

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
        The compressed frame as (codebook level, DAC codes, encoded residuals).
        
    Returns
    -------
    float
        The size of the compressed frame in bytes
    """

    # unpack bottleneck frame
    codebook_level, codes, encoded_residuals = bottleneck_frame

    # add size for codebook level
    total_size = MAXIMUM_CODEBOOK_LEVEL_BITS

    # add size for DAC codes
    total_size += BITS_PER_SAMPLE_BITS # get_numpy_dtype_bit_size(dtype = codes.dtype) # number of bits per sample for dtype
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
    codebook_level, n_samples, dac_time_dimension, audio_scale_bits, serialized_entropy_coder, chunk_length, input_db_bits, padding, frames = bottleneck

    # add size for global codebook level, which is redundant for Adaptive DAC, but we include it to make it easily compatible with Naive DAC (this value is 0)
    total_size = MAXIMUM_CODEBOOK_LEVEL_BITS

    # add size for number of samples
    total_size += N_SAMPLES_BITS

    # add size for DAC time dimension
    total_size += DAC_TIME_DIMENSION_BITS
    
    # add size for audio scale bits
    total_size += MAXIMUM_AUDIO_SCALE_BITS

    # add size for serialized entropy coder
    total_size += SERIALIZED_ENTROPY_CODER_BITS # 1 byte for serialized entropy coder

    # add size for chunk length
    total_size += CHUNK_LENGTH_BITS

    # add size for input dB bits
    total_size += INPUT_DB_BITS

    # add size for padding
    total_size += 1 # boolean for whether there is padding

    # add size for bit for number of subframes in frame
    total_size += 1 # boolean for whether there are two frames (stereo) or one (mono)
    
    # add size for each frame
    for frame in frames:
        total_size += get_compressed_frame_size(bottleneck_frame = frame) * 8

    # convert total_size to bytes
    total_size /= 8

    return total_size

##################################################


# MAIN ENCODE AND DECODE FUNCTIONS
##################################################

def encode(
    data: np.ndarray,
    entropy_coder: EntropyCoder,
    model: dac.model.dac.DAC,
    sample_rate: int,
    audio_scale: float = None,
    window_duration: float = WINDOW_DURATION_DEFAULT,
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
    audio_scale : float, default = None
        The audio scale to use for encoding. If None, will be calculated using the data. If provided, will be used as is.
    window_duration : float, default = WINDOW_DURATION_DEFAULT
        The window duration to use for encoding.

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
        audio_scale = audio_scale,
        window_duration = window_duration,
    )
    
    # batch encode frames
    with torch.no_grad():

        # get codes at maximum codebook level
        maximum_codebook_level_dac_file = dac_encode_codes_only(
            data = data,
            model = model,
            sample_rate = sample_rate,
            audio_scale = audio_scale,
            window_duration = window_duration,
        )

        # get information to pack bottleneck
        n_samples = len(data)
        dac_time_dimension = maximum_codebook_level_dac_file.codes.shape[-1] # get dac time dimension from first frame
        audio_scale_bits = encode_audio_scale_bits(audio_scale = audio_scale)
        serialized_entropy_coder = serialize(entropy_coder = entropy_coder)
        chunk_length = maximum_codebook_level_dac_file.chunk_length
        input_db_bits = encode_input_db_bits(input_db = maximum_codebook_level_dac_file.input_db.item())
        padding = maximum_codebook_level_dac_file.padding

        # encode each channel independently
        frames = [None] * maximum_codebook_level_dac_file.channels
        for channel_idx in range(len(frames)):

            # modify dac file to include only this channel
            channel_dac_file = deepcopy(maximum_codebook_level_dac_file)
            channel_dac_file.codes = channel_dac_file.codes[channel_idx, :, :].unsqueeze(dim = 0)
            channel_dac_file.channels = 1

            # try all codebook levels
            best_frame = None
            best_compressed_frame_size = float("inf")
            for candidate_codebook_level in range(MAXIMUM_CODEBOOK_LEVEL, 0, -1):
                candidate_dac_file = deepcopy(channel_dac_file)
                candidate_dac_file, candidate_encoded_residuals = dac_encode_codes_and_residuals(
                    data = data[:, channel_idx] if len(data.shape) == 2 else data,
                    model = model,
                    sample_rate = sample_rate,
                    entropy_coder = entropy_coder,
                    audio_scale = audio_scale,
                    window_duration = window_duration,
                    codebook_level = candidate_codebook_level,
                    dac_file = candidate_dac_file,
                )
                candidate_frame = (candidate_codebook_level, candidate_dac_file.codes.detach().cpu().numpy(), candidate_encoded_residuals[0]) # pack candidate frame with codebook level, codes, and encoded residuals
                candidate_compressed_frame_size = get_compressed_frame_size(bottleneck_frame = candidate_frame)
                if candidate_compressed_frame_size < best_compressed_frame_size:
                    best_compressed_frame_size = candidate_compressed_frame_size
                    best_frame = deepcopy(candidate_frame)
                del candidate_dac_file, candidate_encoded_residuals, candidate_frame, candidate_compressed_frame_size # delete to free memory

            # add best frame to frames
            frames[channel_idx] = deepcopy(best_frame)

            # free up memory
            del channel_dac_file, best_frame, best_compressed_frame_size

        # free up memory
        del maximum_codebook_level_dac_file

    # pack bottleneck
    bottleneck = (
        0, # 0 global codebook level for Adaptive DAC
        n_samples,
        dac_time_dimension,
        audio_scale_bits,
        serialized_entropy_coder,
        chunk_length,
        input_db_bits,
        padding,
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
    _, n_samples, dac_time_dimension, audio_scale_bits, serialized_entropy_coder, chunk_length, input_db_bits, padding, frames = bottleneck
    audio_scale = decode_audio_scale_bits(audio_scale_bits = audio_scale_bits)
    entropy_coder = deserialize(header = serialized_entropy_coder)
    input_db = decode_input_db_bits(input_db_bits = input_db_bits)
    is_mono = len(frames) == 1

    # batch decode frames
    with torch.no_grad():
        reconstructed_data = [None] * len(frames)
        for i, (codebook_level, codes, encoded_residuals) in enumerate(frames):
            reconstructed_data[i] = dac_decode(
                codes = codes,
                encoded_residuals = [encoded_residuals],
                model = model,
                entropy_coder = entropy_coder,
                n_samples = n_samples,
                audio_scale = audio_scale,
                chunk_length = chunk_length,
                input_db = input_db,
                padding = padding,
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
) -> Tuple[int, int, int]:
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
    Tuple[int, int, int]
        A tuple containing the number of metadata bits, estimator bits, and entropy bits.
    """

    # unpack frame
    codebook_level, codes, encoded_residuals = frame
    
    # write codebook level
    metadata_bits_start = bitstream.get_position()
    bitstream.write_bits(bits = codebook_level, n = MAXIMUM_CODEBOOK_LEVEL_BITS)
    metadata_bits_end = bitstream.get_position()
    metadata_bits = metadata_bits_end - metadata_bits_start

    # write codes array
    estimator_bits_start = bitstream.get_position()
    codes_bits_per_sample = get_minimum_number_of_bits_for_sample(sample = codes.max()) # get_numpy_dtype_bit_size(dtype = codes.dtype)
    bitstream.write_bits(bits = codes_bits_per_sample, n = BITS_PER_SAMPLE_BITS) # number of bits per sample for dtype
    for code in codes.flatten():
        bitstream.write_bits(bits = int(code), n = codes_bits_per_sample)
    estimator_bits_end = bitstream.get_position()
    estimator_bits = estimator_bits_end - estimator_bits_start

    # write encoded residuals
    entropy_bits_start = bitstream.get_position()
    bitstream.write_bits(bits = len(encoded_residuals), n = ENCODED_RESIDUALS_SIZE_BITS) # number of bytes for encoded residuals as 32 bit unsigned integer
    for byte in encoded_residuals:
        bitstream.write_bits(bits = byte, n = 8)
    entropy_bits_end = bitstream.get_position()
    entropy_bits = entropy_bits_end - entropy_bits_start

    return (metadata_bits, estimator_bits, entropy_bits)

def write_bottleneck(
    bottleneck: BOTTLENECK_TYPE,
    path: str,
) -> dict:
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
    dict
        A dictionary containing statistics about the bottleneck.
    """

    # unpack bottleneck
    expected_size = get_compressed_bottleneck_size(bottleneck = bottleneck)
    _, n_samples, dac_time_dimension, audio_scale_bits, serialized_entropy_coder, chunk_length, input_db_bits, padding, frames = bottleneck

    # create bitstream
    bit_output = bitstream.BitOutputStream(path = path, buffer_size = int(expected_size * 1.2)) # buffer size is 1.2x the size of the bottleneck to avoid reallocating memory
    metadata_bits, estimator_bits, entropy_bits = 0, 0, 0

    # write codebook level
    bit_output.write_bits(bits = 0, n = MAXIMUM_CODEBOOK_LEVEL_BITS) # global codebook level is redundant for Adaptive DAC, but we include it to make it easily compatible with Naive DAC (this value is 0)

    # write number of samples (assumes N_SAMPLES_BITS is no larger than 64)
    if N_SAMPLES_BITS > 32:
        bit_output.write_bits(bits = (n_samples) & 0xFFFFFFFF, n = 32) # lower 32 bits
        bit_output.write_bits(bits = (n_samples >> 32) & 0xFFFFFFFF, n = N_SAMPLES_BITS - 32) # upper 32 bits
    else:
        bit_output.write_bits(bits = n_samples, n = N_SAMPLES_BITS)

    # write dac time dimension
    bit_output.write_bits(bits = dac_time_dimension, n = DAC_TIME_DIMENSION_BITS)
    
    # write audio scale bits
    bit_output.write_bits(bits = audio_scale_bits, n = MAXIMUM_AUDIO_SCALE_BITS)

    # write serialized entropy coder
    bit_output.write_bits(bits = serialized_entropy_coder, n = SERIALIZED_ENTROPY_CODER_BITS)

    # write chunk length
    bit_output.write_bits(bits = chunk_length, n = CHUNK_LENGTH_BITS)

    # write input dB bits
    bit_output.write_bits(bits = input_db_bits, n = INPUT_DB_BITS)

    # write padding
    bit_output.write_bit(bit = padding)
    
    # write bit for whether there are two frames (stereo) or one (mono)
    bit_output.write_bit(bit = len(frames) == 2)

    # align to byte
    bit_output.align_to_byte()  
    metadata_bits += bit_output.get_position()
    
    # write each frame (each frame is a tuple of the DAC codes and encoded residuals)
    for frame in frames:
        metadata_bits_frame, estimator_bits_frame, entropy_bits_frame = write_frame(frame = frame, bitstream = bit_output)
        metadata_bits += metadata_bits_frame
        estimator_bits += estimator_bits_frame
        entropy_bits += entropy_bits_frame
    
    # close bitstream (writes to file)
    bit_output.flush()
    bit_output.close()

    return {
        "total_bits": bit_output.get_position(),
        "metadata_bits": metadata_bits,
        "estimator_bits": estimator_bits,
        "entropy_bits": entropy_bits,
    }
        
##################################################


# READ BOTTLENECK FUNCTIONS
##################################################

def read_frame(
    bitstream: bitstream.BitInputStream,
    dac_time_dimension: int,
) -> BOTTLENECK_FRAME_TYPE:
    """
    Read a frame from a bitstream.

    Parameters
    ----------
    bitstream : bitstream.BitInputStream
        The bitstream to read from.
    dac_time_dimension : int
        The DAC time dimension.

    Returns
    -------
    BOTTLENECK_FRAME_TYPE
        The frame.
    """

    # read codebook level
    codebook_level = bitstream.read_bits(n = MAXIMUM_CODEBOOK_LEVEL_BITS)

    # read codes array
    codes_bits_per_sample = bitstream.read_bits(n = BITS_PER_SAMPLE_BITS)
    codes = [bitstream.read_bits(n = codes_bits_per_sample) for _ in range(codebook_level * dac_time_dimension)]
    codes = np.array(codes).astype(np.int64) # np.array(codes).astype(get_numpy_dtype_from_bit_size(bit_size = codes_bits_per_sample))
    codes = codes.reshape(1, codebook_level, dac_time_dimension)

    # read encoded residuals
    encoded_residuals_len = bitstream.read_bits(n = ENCODED_RESIDUALS_SIZE_BITS)
    encoded_residuals = bytes([bitstream.read_bits(n = 8) for _ in range(encoded_residuals_len)])
    
    # compile frame
    frame = (codebook_level, codes, encoded_residuals)
    
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

    # read redundant codebook level
    _ = bit_input.read_bits(n = MAXIMUM_CODEBOOK_LEVEL_BITS) # global codebook level is redundant for Adaptive DAC, but we include it to make it easily compatible with Naive DAC (this value is 0)
    
    # read number of samples (assumes N_SAMPLES_BITS is no larger than 64)
    if N_SAMPLES_BITS > 32:
        n_samples_lower = bit_input.read_bits(n = 32) # lower 32 bits
        n_samples_upper = bit_input.read_bits(n = N_SAMPLES_BITS - 32) # upper 32 bits
        n_samples = (n_samples_upper << 32) | n_samples_lower
    else:
        n_samples = bit_input.read_bits(n = N_SAMPLES_BITS)

    # read dac time dimension
    dac_time_dimension = bit_input.read_bits(n = DAC_TIME_DIMENSION_BITS)

    # read audio scale bits
    audio_scale_bits = bit_input.read_bits(n = MAXIMUM_AUDIO_SCALE_BITS)

    # read serialized entropy coder
    serialized_entropy_coder = bit_input.read_bits(n = SERIALIZED_ENTROPY_CODER_BITS)

    # read chunk length
    chunk_length = bit_input.read_bits(n = CHUNK_LENGTH_BITS)

    # read input dB bits
    input_db_bits = bit_input.read_bits(n = INPUT_DB_BITS)

    # read padding
    padding = bit_input.read_bit()
    
    # read bit for whether there are two frames (stereo) or one (mono)
    is_stereo = bit_input.read_bit()
    n_channels = 2 if is_stereo else 1

    # align to byte
    bit_input.align_to_byte()
    
    # read each frame
    frames = [None] * n_channels
    for i in range(len(frames)):
        frames[i] = read_frame(
            bitstream = bit_input,
            dac_time_dimension = dac_time_dimension,
        )

    # pack bottleneck
    bottleneck = (0, n_samples, dac_time_dimension, audio_scale_bits, serialized_entropy_coder, chunk_length, input_db_bits, padding, frames)
    
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
    audio_scale: float = None,
    window_duration: float = WINDOW_DURATION_DEFAULT,
    return_statistics: bool = False,
) -> Union[None, dict]:
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
    audio_scale : float, default = None
        The audio scale to use for encoding. If None, will be calculated using the data. If provided, will be used as is.
    window_duration : float, default = WINDOW_DURATION_DEFAULT
        The window duration to use for encoding.
    return_statistics : bool, default = False
        Whether to return statistics about the bottleneck.
    
    Returns
    -------
    Union[None, dict]
        None if return_statistics is False, otherwise a dictionary containing statistics about the bottleneck.
    """

    # get bottleneck
    bottleneck = encode(
        data = data,
        entropy_coder = entropy_coder,
        model = model,
        sample_rate = sample_rate,
        audio_scale = audio_scale,
        window_duration = window_duration,
    )

    # write bottleneck to file
    statistics = write_bottleneck(
        bottleneck = bottleneck,
        path = path,
    )

    # return statistics if requested
    if return_statistics:
        return statistics
    
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