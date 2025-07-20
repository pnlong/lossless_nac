# README
# Phillip Long
# July 12, 2025

# Naive DAC Compressor.

# IMPORTS
##################################################

import numpy as np
from typing import List, Tuple, Dict, Any
import torch

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))
sys.path.insert(0, f"{dirname(dirname(dirname(realpath(__file__))))}/dac") # for dac import

from ldac_compressor import *
from entropy_coders.entropy_coder import EntropyCoder
from entropy_coders.serialize import serialize, deserialize
import bitstream
import dac

##################################################


# BOTTLENECK TYPE
##################################################

# type of bottleneck subframes
BOTTLENECK_SUBFRAME_TYPE = Tuple[int, int, np.ndarray, bytes] # bottleneck subframe type is a tuple of the number of samples, DAC time dimension, DAC codes, and encoded residuals

# type of bottleneck frames
BOTTLENECK_FRAME_TYPE = List[BOTTLENECK_SUBFRAME_TYPE] # bottleneck frame type is a list of subframes

# type of bottleneck
BOTTLENECK_TYPE = Tuple[int, int, List[BOTTLENECK_FRAME_TYPE]] # bottleneck type is a tuple of the codebook level, serialized entropy coder, and list of frames

##################################################


# NAIVE ENCODING FUNCTIONS
##################################################

def encode_subframes_batch_optimized(
    subframes_data: List[np.ndarray],
    subframes_metadata: List[Dict[str, Any]],
    entropy_coder: EntropyCoder,
    model: dac.model.dac.DAC,
    sample_rate: int,
    codebook_level: int = CODEBOOK_LEVEL_DEFAULT,
    batch_size: int = BATCH_SIZE_DEFAULT,
) -> List[BOTTLENECK_SUBFRAME_TYPE]:
    """
    Encode multiple subframes using batched DAC processing.
    
    Parameters
    ----------
    subframes_data : List[np.ndarray]
        List of subframes to encode.
    subframes_metadata : List[Dict[str, Any]]
        Metadata for each subframe.
    entropy_coder : EntropyCoder
        The entropy coder to use.
    model : dac.model.dac.DAC
        The DAC model to use for lossy estimation.
    sample_rate : int
        Sample rate.
    codebook_level : int, default = CODEBOOK_LEVEL_DEFAULT
        Number of codebooks to use.
    batch_size : int
        Batch size for processing
        
    Returns
    -------
    List[BOTTLENECK_SUBFRAME_TYPE]
        List of encoded subframes
    """

    # initialize list of encoded subframes
    encoded_subframes = []
    
    # process subframes in batches
    for i in range(0, len(subframes_data), batch_size):

        # get batch info
        batch_end = min(i + batch_size, len(subframes_data)) # batch end index
        batch_subframes = subframes_data[i:batch_end] # batch of subframes
        batch_metadata = subframes_metadata[i:batch_end] # batch of metadata
        
        # pad subframes to same length
        padded_batch = pad_subframes_to_batch(subframes = batch_subframes)
        
        # batch process through DAC
        codes_batch, approximate_batch = batch_dac_encode_full(
            subframes_batch = padded_batch,
            model = model,
            sample_rate = sample_rate,
            codebook_level = codebook_level,
        )
        
        # process each item in the batch
        for j, (codes, approximate, metadata) in enumerate(zip(codes_batch, approximate_batch, batch_metadata)):

            # get original length and subframe
            original_length = metadata["original_length"]
            original_subframe = batch_subframes[j]

            # truncate approximate to original length
            approximate_truncated = approximate[:original_length]
            approximate_truncated = convert_audio_floating_to_fixed(
                waveform = approximate_truncated,
                output_dtype = original_subframe.dtype,
            )
            
            # compute residuals
            residuals = original_subframe - approximate_truncated
            
            # entropy encode residuals
            encoded_residuals = entropy_coder.encode(nums = residuals)
            
            # create bottleneck subframe
            bottleneck_subframe = (original_length, codes.shape[-1], codes, encoded_residuals) # create bottleneck subframe
            encoded_subframes.append(bottleneck_subframe) # add bottleneck subframe to list of encoded subframes
    
    return encoded_subframes

##################################################


# NAIVE DECODING FUNCTIONS
##################################################

def decode_subframes_batch_optimized(
    bottleneck_subframes: List[BOTTLENECK_SUBFRAME_TYPE],
    entropy_coder: EntropyCoder,
    model: dac.model.dac.DAC,
    batch_size: int = BATCH_SIZE_DEFAULT,
) -> List[np.ndarray]:
    """
    Decode multiple subframes using batched DAC processing.
    
    Parameters
    ----------
    bottleneck_subframes : List[BOTTLENECK_SUBFRAME_TYPE]
        List of encoded subframes.
    entropy_coder : EntropyCoder
        Entropy coder to use.
    model : dac.model.dac.DAC
        The DAC model to use for reconstruction.
    batch_size : int, default = BATCH_SIZE_DEFAULT
        Batch size for processing.
        
    Returns
    -------
    List[np.ndarray]
        List of decoded subframes
    """

    # initialize list of decoded subframes
    decoded_subframes = []
    
    # process subframes in batches
    for i in range(0, len(bottleneck_subframes), batch_size):

        # get batch info
        batch_end = min(i + batch_size, len(bottleneck_subframes)) # batch end index
        batch_bottlenecks = bottleneck_subframes[i:batch_end] # batch of bottleneck subframes
        
        # extract codes and prepare for batching
        lengths_list = []
        dac_time_dimensions_list = []
        codes_list = []
        residuals_list = []
        for n_samples, dac_time_dimension, codes, encoded_residuals in batch_bottlenecks:
            lengths_list.append(n_samples) # append length
            dac_time_dimensions_list.append(dac_time_dimension) # append DAC time dimension
            codes_list.append(codes) # append codes
            residuals = entropy_coder.decode(stream = encoded_residuals, num_samples = n_samples) # decode residuals
            residuals_list.append(residuals) # append residuals
        
        # stack codes for batch processing
        codes_batch = np.stack(codes_list, axis = 0)
        
        # batch decode through DAC
        approximate_batch = batch_dac_decode(codes_batch = codes_batch, model = model)
        
        # process each item in the batch
        for j, (approximate, residuals, length, dac_time_dimension) in enumerate(zip(approximate_batch, residuals_list, lengths_list, dac_time_dimensions_list)):

            # truncate and convert approximate
            approximate_truncated = approximate[:length]
            approximate_truncated = convert_audio_floating_to_fixed(
                waveform = approximate_truncated, 
                output_dtype = np.int32, # use int32 for compatibility
            )
            
            # reconstruct subframe
            reconstructed_subframe = approximate_truncated + residuals # add residuals to approximate to get reconstructed subframe
            decoded_subframes.append(reconstructed_subframe) # add to list of decoded subframes
    
    return decoded_subframes

def decode_frames_batch_optimized(
    bottleneck: BOTTLENECK_TYPE,
    entropy_coder: EntropyCoder,
    model: dac.model.dac.DAC,
    batch_size: int = BATCH_SIZE_DEFAULT,
) -> List[np.ndarray]:
    """
    Decode frames using batched DAC processing.
    
    Parameters
    ----------
    bottleneck : BOTTLENECK_TYPE
        Encoded frames.
    entropy_coder : EntropyCoder
        Entropy coder to use.
    model : dac.model.dac.DAC
        DAC model.
    batch_size : int, default = BATCH_SIZE_DEFAULT
        Batch size for processing.
        
    Returns
    -------
    List[np.ndarray]
        List of decoded frames
    """

    # collect all subframes and create index mapping
    all_subframes, subframe_index_map = create_subframe_index_map(bottleneck = bottleneck)
    
    # batch decode all subframes
    decoded_subframes = decode_subframes_batch_optimized(
        bottleneck_subframes = all_subframes,
        entropy_coder = entropy_coder,
        model = model,
        batch_size = batch_size,
    )
    
    # organize back into frames
    decoded_frames = reconstruct_frames_from_subframes(
        decoded_subframes = decoded_subframes,
        bottleneck = bottleneck,
        subframe_index_map = subframe_index_map,
    )
    
    return decoded_frames

##################################################


# SIZE CALCULATION FUNCTIONS
##################################################

def get_compressed_subframe_size(
    bottleneck_subframe: BOTTLENECK_SUBFRAME_TYPE,
) -> float:
    """
    Get the size of a compressed subframe in bytes.

    Parameters
    ----------
    bottleneck_subframe : BOTTLENECK_SUBFRAME_TYPE
        The compressed subframe as (n_samples, dac_time_dimension, dac_codes, encoded_residuals).
        
    Returns
    -------
    float
        The size of the compressed subframe in bytes
    """

    # unpack bottleneck subframe
    n_samples, dac_time_dimension, codes, encoded_residuals = bottleneck_subframe

    # add size for storing number of samples
    total_size = MAXIMUM_BLOCK_SIZE_ASSUMPTION_BYTES

    # add size for DAC codes
    total_size += MAXIMUM_DAC_TIME_DIMENSION_ASSUMPTION_BYTES # we can store the DAC time dimension as one byte, as a 1-byte unsigned integer
    total_size += get_numpy_dtype_bit_size(dtype = codes.dtype) # number of bits per sample for dtype
    total_size += codes.nbytes

    # add size for encoded residuals
    total_size += 4 # 4 bytes for the number of bytes for encoded residuals as 32 bit unsigned integer
    total_size += len(encoded_residuals)

    return total_size

def get_compressed_frame_size(
    bottleneck_frame: BOTTLENECK_FRAME_TYPE,
) -> float:
    """
    Get the size of a compressed frame in bytes.
    
    Parameters
    ----------
    bottleneck_frame : BOTTLENECK_FRAME_TYPE
        The compressed frame as list_of_subframes.
        
    Returns
    -------
    float
        The size of the compressed frame in bytes
    """

    # add size for each subframe
    total_size = 1 / 8 # 1 bit for number of subframes in frame
    for bottleneck_subframe in bottleneck_frame:
        total_size += get_compressed_subframe_size(bottleneck_subframe = bottleneck_subframe)
    
    return total_size

##################################################


# MAIN ENCODE AND DECODE FUNCTIONS
##################################################

def encode(
    data: np.ndarray,
    entropy_coder: EntropyCoder,
    model: dac.model.dac.DAC,
    sample_rate: int,
    block_size: int = BLOCK_SIZE_DEFAULT,
    codebook_level: int = CODEBOOK_LEVEL_DEFAULT,
    batch_size: int = BATCH_SIZE_DEFAULT,
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
    block_size : int, default = BLOCK_SIZE_DEFAULT
        The block size to use for encoding.
    codebook_level : int, default = CODEBOOK_LEVEL_DEFAULT
        The number of codebooks to use for DAC encoding.
    batch_size : int, default = BATCH_SIZE_DEFAULT
        The batch size to use for encoding.

    Returns
    -------
    BOTTLENECK_TYPE
        The bottleneck.
    """
    
    # ensure input is valid
    validate_input_data(data = data)
    assert sample_rate == model.sample_rate, f"Sample rate must match the sample rate of the model. Model sample rate: {model.sample_rate}, provided sample rate: {sample_rate}."
    assert block_size > 0 and block_size <= MAXIMUM_BLOCK_SIZE_ASSUMPTION, f"Block size must be positive and less than or equal to {MAXIMUM_BLOCK_SIZE_ASSUMPTION}."
    assert codebook_level > 0 and codebook_level <= MAXIMUM_CODEBOOK_LEVEL, f"Codebook level must be between 1 and {MAXIMUM_CODEBOOK_LEVEL}."
    assert batch_size > 0, "Batch size must be positive."
    
    # split data into frames
    frames = partition_data_into_frames(data = data, block_size = block_size)
    
    # collect subframes for batch processing
    subframes_metadata, subframes_data = collect_subframes_for_batch_processing(frames = frames)
    
    # batch encode subframes
    with torch.no_grad():
        encoded_subframes = encode_subframes_batch_optimized(
            subframes_data = subframes_data,
            subframes_metadata = subframes_metadata,
            entropy_coder = entropy_coder,
            model = model,
            sample_rate = sample_rate,
            codebook_level = codebook_level,
            batch_size = batch_size,
        )
    
    # organize subframes back into frame structure
    bottleneck = organize_subframes_into_frames(
        encoded_subframes = encoded_subframes,
        subframes_metadata = subframes_metadata,
        n_frames = len(frames),
    )

    # pack final bottleneck
    bottleneck = (codebook_level, serialize(entropy_coder = entropy_coder), bottleneck)
    
    return bottleneck

def decode(
    bottleneck: BOTTLENECK_TYPE,
    model: dac.model.dac.DAC,
    batch_size: int = BATCH_SIZE_DEFAULT,
) -> np.ndarray:
    """
    Decode the bottleneck into the original data.

    Parameters
    ----------
    bottleneck : BOTTLENECK_TYPE
        The bottleneck to decode.
    model : dac.model.dac.DAC
        The DAC model to use.
    batch_size : int, default = BATCH_SIZE_DEFAULT
        The batch size to use for decoding.

    Returns
    -------
    np.ndarray
        The decoded original data.
    """

    # unpack bottleneck
    codebook_level, serialized_entropy_coder, bottleneck = bottleneck
    entropy_coder = deserialize(header_byte = serialized_entropy_coder)

    # batch decode frames
    with torch.no_grad():
        decoded_frames = decode_frames_batch_optimized(
            bottleneck = bottleneck,
            entropy_coder = entropy_coder,
            model = model,
            batch_size = batch_size,
        )
    
    # concatenate all decoded frames
    reconstructed_data = np.concatenate(decoded_frames, axis = 0)

    # ensure output is valid
    validate_output_data(reconstructed_data = reconstructed_data)
    
    return reconstructed_data

##################################################


# WRITE BOTTLENECK FUNCTIONS
##################################################

def write_subframe(
    subframe: BOTTLENECK_SUBFRAME_TYPE,
    bitstream: bitstream.BitOutputStream,
) -> None:
    """
    Write a subframe to a bitstream.

    Parameters
    ----------
    subframe : BOTTLENECK_SUBFRAME_TYPE
        The subframe to write.
    bitstream : bitstream.BitOutputStream
        The bitstream to write to.

    Returns
    -------
    None
    """

    # unpack subframe
    n_samples, dac_time_dimension, codes, encoded_residuals = subframe
    
    # write number of samples
    bitstream.write_bits(bits = n_samples, n = MAXIMUM_BLOCK_SIZE_ASSUMPTION_BYTES * 8)
    
    # write dac_time_dimension
    bitstream.write_bits(bits = dac_time_dimension, n = MAXIMUM_DAC_TIME_DIMENSION_ASSUMPTION_BYTES * 8)
    
    # write codes array
    codes_bits_per_sample = get_numpy_dtype_bit_size(dtype = codes.dtype)
    bitstream.write_bits(bits = codes_bits_per_sample, n = 8) # number of bits per sample for dtype
    for code in codes.flatten():
        bitstream.write_bits(bits = int(code), n = codes_bits_per_sample)
    
    # write encoded residuals
    bitstream.write_bits(bits = len(encoded_residuals), n = 32) # number of bytes for encoded residuals as 32 bit unsigned integer
    for byte in encoded_residuals:
        bitstream.write_bits(bits = byte, n = 8)

    return

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
    
    # write number of subframes in this frame
    bitstream.write_bit(bit = len(frame) == 2) # there are either one or two subframes in a frame
    
    # write each subframe
    for subframe in frame:
        write_subframe(subframe = subframe, bitstream = bitstream)

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
    codebook_level, serialized_entropy_coder, bottleneck = bottleneck

    # create bitstream
    bit_output = bitstream.BitOutputStream(path = path)

    # write codebook level
    bit_output.write_bits(bits = codebook_level, n = MAXIMUM_CODEBOOK_LEVEL_BITS)
    bit_output.align_to_byte()

    # write serialized entropy coder
    bit_output.write_byte(byte = serialized_entropy_coder)
    
    # write number of frames
    bit_output.write_uint(value = len(bottleneck))
    
    # write each frame
    for frame in bottleneck:
        write_frame(frame = frame, bitstream = bit_output)
    
    # close bitstream (writes to file)
    bit_output.flush()
    bit_output.close()

    return
        
##################################################


# READ BOTTLENECK FUNCTIONS
##################################################

def read_subframe(
    bitstream: bitstream.BitInputStream,
    codebook_level: int = CODEBOOK_LEVEL_DEFAULT,
) -> BOTTLENECK_SUBFRAME_TYPE:
    """
    Read a subframe from a bitstream.

    Parameters
    ----------
    bitstream : bitstream.BitInputStream
        The bitstream to read from.
    codebook_level : int, default = CODEBOOK_LEVEL_DEFAULT
        The number of codebooks to use for DAC encoding.

    Returns
    -------
    BOTTLENECK_SUBFRAME_TYPE
        The subframe.
    """

    # read n_samples
    n_samples = bitstream.read_bits(n = MAXIMUM_BLOCK_SIZE_ASSUMPTION_BYTES * 8)
    
    # read dac_time_dimension
    dac_time_dimension = bitstream.read_bits(n = MAXIMUM_DAC_TIME_DIMENSION_ASSUMPTION_BYTES * 8)
    
    # read codes array
    codes_bits_per_sample = bitstream.read_bits(n = 8)
    codes = [bitstream.read_bits(n = codes_bits_per_sample) for _ in range(codebook_level * dac_time_dimension)]
    codes = np.array(codes).astype(get_numpy_dtype_from_bit_size(bit_size = codes_bits_per_sample))
    codes = codes.reshape(codebook_level, dac_time_dimension)

    # read encoded residuals
    encoded_residuals_len = bitstream.read_bits(n = 32)
    encoded_residuals = bytes([bitstream.read_bits(n = 8) for _ in range(encoded_residuals_len)])
    
    # compile subframe
    subframe = (n_samples, dac_time_dimension, codes, encoded_residuals)
    
    return subframe

def read_frame(
    bitstream: bitstream.BitInputStream,
    codebook_level: int = CODEBOOK_LEVEL_DEFAULT,
) -> BOTTLENECK_FRAME_TYPE:
    """
    Read a frame from a bitstream.

    Parameters
    ----------
    bitstream : bitstream.BitInputStream
        The bitstream to read from.
    codebook_level : int, default = CODEBOOK_LEVEL_DEFAULT
        The number of codebooks to use for DAC encoding.

    Returns
    -------
    BOTTLENECK_FRAME_TYPE
        The frame.
    """

    # read number of subframes
    n_subframes = 2 if bitstream.read_bit() else 1 # there are either one or two subframes in a frame
    
    # read each subframe
    frame = [None] * n_subframes
    for i in range(n_subframes):
        subframe = read_subframe(bitstream = bitstream, codebook_level = codebook_level)
        frame[i] = subframe
    
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
    bit_input.align_to_byte()

    # read serialized entropy coder
    serialized_entropy_coder = bit_input.read_byte()
    
    # read number of frames
    n_frames = bit_input.read_uint()
    
    # read each frame
    bottleneck = [None] * n_frames
    for i in range(n_frames):
        frame = read_frame(bitstream = bit_input, codebook_level = codebook_level)
        bottleneck[i] = frame

    # pack bottleneck
    bottleneck = (codebook_level, serialized_entropy_coder, bottleneck)
    
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
    block_size: int = BLOCK_SIZE_DEFAULT,
    codebook_level: int = CODEBOOK_LEVEL_DEFAULT,
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
    block_size : int, default = BLOCK_SIZE_DEFAULT
        The block size to use for encoding.
    codebook_level : int, default = CODEBOOK_LEVEL_DEFAULT
        The number of codebooks to use for DAC encoding.
    batch_size : int, default = BATCH_SIZE_DEFAULT
        The batch size to use for encoding.

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
        block_size = block_size,
        codebook_level = codebook_level,
        batch_size = batch_size,
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
    batch_size: int = BATCH_SIZE_DEFAULT,
) -> np.ndarray:
    """
    Decode the data from a file.

    Parameters
    ----------
    path : str
        The path to the file to read.
    model : dac.model.dac.DAC
        The DAC model to use.
    batch_size : int, default = BATCH_SIZE_DEFAULT
        The batch size to use for decoding.

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
        batch_size = batch_size,
    )

    return data

##################################################