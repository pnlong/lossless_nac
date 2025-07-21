# README
# Phillip Long
# July 12, 2025

# Adaptive DAC Compressor.

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
from entropy_coders.serialize import serialize, deserialize
import bitstream
import dac

##################################################


# BOTTLENECK TYPE
##################################################

# type of bottleneck subframes - now includes codebook level
BOTTLENECK_SUBFRAME_TYPE = Tuple[int, int, int, np.array, bytes] # bottleneck subframe type is a tuple of the number of samples, codebook level, DAC time dimension, DAC codes, and encoded residuals

# type of bottleneck frames
BOTTLENECK_FRAME_TYPE = List[BOTTLENECK_SUBFRAME_TYPE] # bottleneck frame type is a list of subframes

# type of bottleneck
BOTTLENECK_TYPE = Tuple[int, int, int, int, List[BOTTLENECK_FRAME_TYPE]] # bottleneck type is a tuple of the codebook level, batch size bits, audio scale bits, serialized entropy coder, and list of frames

##################################################


# ADAPTIVE ENCODING FUNCTIONS
##################################################

def encode_subframes_batch_optimized(
    subframes_data: List[np.array],
    entropy_coder: EntropyCoder,
    model: dac.model.dac.DAC,
    sample_rate: int,
    batch_size: int = BATCH_SIZE_DEFAULT,
    audio_scale: float = AUDIO_SCALE_DEFAULT,
) -> List[BOTTLENECK_SUBFRAME_TYPE]:
    """
    Encode multiple subframes using adaptive DAC processing optimized for batching.
    
    Strategy:
    1. For each codebook level, batch encode all subframes at that specific level
    2. Batch decode and evaluate compression ratio for each level
    3. Select best codebook level for each subframe based on compression ratio
    
    Parameters
    ----------
    subframes_data : List[np.array]
        List of subframes to encode.
    entropy_coder : EntropyCoder
        The entropy coder to use.
    model : dac.model.dac.DAC
        The DAC model to use for lossy estimation.
    sample_rate : int
        Sample rate.
    batch_size : int, default = BATCH_SIZE_DEFAULT
        Batch size for processing.
    audio_scale : float, default = AUDIO_SCALE_DEFAULT
        The audio scale to use for encoding.
        
    Returns
    -------
    List[BOTTLENECK_SUBFRAME_TYPE]
        List of encoded subframes with optimal codebook levels
    """

    # initialize results
    n_subframes = len(subframes_data)
    best_subframes = [None] * n_subframes
    best_sizes = [float("inf")] * n_subframes
    
    # for each codebook level, evaluate all subframes
    for codebook_level in range(1, MAXIMUM_CODEBOOK_LEVEL + 1):
        
        # process subframes in batches for this codebook level
        for batch_start in range(0, n_subframes, batch_size):
            batch_end = min(batch_start + batch_size, n_subframes)
            batch_subframes = subframes_data[batch_start:batch_end]
            
            # pad batch to exact batch size to ensure consistent batch processing
            if len(batch_subframes) < batch_size:
                padding_needed = batch_size - len(batch_subframes)
                dummy_subframe = np.zeros_like(batch_subframes[0])
                batch_subframes = batch_subframes + ([dummy_subframe] * padding_needed)
            
            # encode this batch at the current codebook level
            padded_batch = pad_subframes_to_batch(subframes = batch_subframes)
            codes_batch = batch_dac_encode_codes_only(
                subframes_batch = padded_batch,
                model = model,
                sample_rate = sample_rate,
                codebook_level = codebook_level,
                audio_scale = audio_scale,
            )
            
            # batch decode
            codes_batch_stacked = np.stack(codes_batch, axis = 0)
            approximate_batch = batch_dac_decode(codes_batch = codes_batch_stacked, model = model)
            
            # evaluate each subframe in this batch (only non-dummy items)
            for i, (subframe_data, codes, approximate) in enumerate(zip(batch_subframes, codes_batch, approximate_batch)):
                
                subframe_idx = batch_start + i
                
                # skip dummy padding items
                if subframe_idx >= n_subframes:
                    break
                
                # truncate approximate to original length
                original_length = len(subframe_data)
                approximate_truncated = approximate[:original_length]
                approximate_truncated = convert_audio_floating_to_fixed(
                    waveform = approximate_truncated,
                    output_dtype = subframe_data.dtype,
                    audio_scale = audio_scale,
                )
                
                # compute residuals
                residuals = subframe_data - approximate_truncated
                
                # entropy encode residuals
                encoded_residuals = entropy_coder.encode(nums = residuals)
                
                # simple losslessness check, verify entropy coding is reversible
                decoded_residuals = entropy_coder.decode(stream = encoded_residuals, num_samples = len(residuals))
                if not np.array_equal(residuals, decoded_residuals): # only consider if entropy coding is lossless
                    continue # skip this level if entropy coding is not lossless

                # update best if this is smaller OR if no previous solution exists
                candidate_subframe = (original_length, codebook_level, codes.shape[-1], codes, encoded_residuals)
                candidate_size = get_compressed_subframe_size(bottleneck_subframe = candidate_subframe)
                if candidate_size < best_sizes[subframe_idx] or best_subframes[subframe_idx] is None: # update best if this is smaller OR if no previous solution exists
                    best_sizes[subframe_idx] = candidate_size
                    best_subframes[subframe_idx] = candidate_subframe
    
    return best_subframes

##################################################


# ADAPTIVE DECODING FUNCTIONS
##################################################

def decode_subframes_batch_adaptive(
    bottleneck_subframes: List[BOTTLENECK_SUBFRAME_TYPE],
    entropy_coder: EntropyCoder,
    model: dac.model.dac.DAC,
    batch_size: int = BATCH_SIZE_DEFAULT,
    audio_scale: float = AUDIO_SCALE_DEFAULT,
) -> List[np.array]:
    """
    Decode multiple subframes using batched DAC processing, grouping by codebook level for maximum efficiency.
    
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
    audio_scale : float, default = AUDIO_SCALE_DEFAULT
        The audio scale to use for decoding.
        
    Returns
    -------
    List[np.array]
        List of decoded subframes in original order
    """

    # initialize results
    n_subframes = len(bottleneck_subframes)
    decoded_subframes = [None] * n_subframes
    
    # group subframes by codebook level for efficient batching
    subframes_by_codebook_level = [[] for _ in range(MAXIMUM_CODEBOOK_LEVEL)]
    for i, bottleneck_subframe in enumerate(bottleneck_subframes):
        n_samples, codebook_level, dac_time_dimension, codes, encoded_residuals = bottleneck_subframe
        subframes_by_codebook_level[codebook_level - 1].append((i, bottleneck_subframe)) # adjust codebook level to be 0-indexed for list indexing
    
    # process each codebook level separately with batch decoding
    for codebook_level, subframes_list in enumerate(subframes_by_codebook_level):

        # adjust codebook level to be 1-indexed
        codebook_level += 1
        
        # process this codebook level in batches
        for batch_start in range(0, len(subframes_list), batch_size):

            # initialize batch data
            batch_end = min(batch_start + batch_size, len(subframes_list)) # don't go past the end of the subframes
            batch_items = subframes_list[batch_start:batch_end] # get batch of subframes
            
            # pad batch to exact batch size to ensure consistent batch processing
            if len(batch_items) < batch_size:
                
                # get reference shape from existing codes
                ref_item = batch_items[0]
                _, ref_bottleneck = ref_item
                _, ref_codebook_level, ref_dac_time_dimension, ref_codes, _ = ref_bottleneck
                dummy_codes = np.zeros((ref_codebook_level, ref_dac_time_dimension), dtype = ref_codes.dtype)
            
                # pad with dummy bottleneck subframes
                dummy_bottleneck = (0, dummy_codes.shape[0], dummy_codes.shape[-1], dummy_codes, bytes())  # dummy bottleneck subframe
                batch_items = batch_items + ([(-1, dummy_bottleneck)] * (batch_size - len(batch_items)))
            
            # prepare batch data
            batch_indices = [None] * len(batch_items)
            batch_codes = [None] * len(batch_items)
            batch_lengths = [None] * len(batch_items)
            batch_residuals = [None] * len(batch_items)
            
            # collect batch data
            for i, (original_index, (n_samples, codebook_level, dac_time_dimension, codes, encoded_residuals)) in enumerate(batch_items):
                batch_indices[i] = original_index # store original index
                batch_codes[i] = codes # store codes
                batch_lengths[i] = n_samples # store length
                batch_residuals[i] = entropy_coder.decode(stream = encoded_residuals, num_samples = n_samples) if n_samples > 0 else np.array([], dtype = np.int32) # decode residuals
            
            # batch decode DAC codes
            codes_batch_stacked = np.stack(batch_codes, axis = 0) # stack codes for batch processing
            approximate_batch = batch_dac_decode(codes_batch = codes_batch_stacked, model = model) # batch decode
            
            # reconstruct each subframe in the batch (only non-dummy items)
            for original_index, approximate, length, residuals in zip(batch_indices, approximate_batch, batch_lengths, batch_residuals):
                
                # skip dummy padding items
                if original_index == -1:
                    continue
                
                # truncate and convert approximate
                approximate_truncated = approximate[:length]
                approximate_truncated = convert_audio_floating_to_fixed(
                    waveform = approximate_truncated, 
                    output_dtype = np.int32, # use int32 for compatibility
                    audio_scale = audio_scale,
                )
                
                # reconstruct subframe
                reconstructed_subframe = approximate_truncated + residuals
                decoded_subframes[original_index] = reconstructed_subframe
    
    return decoded_subframes

def decode_frames_batch_optimized(
    bottleneck: BOTTLENECK_TYPE,
    entropy_coder: EntropyCoder,
    model: dac.model.dac.DAC,
    batch_size: int = BATCH_SIZE_DEFAULT,
    audio_scale: float = AUDIO_SCALE_DEFAULT,
) -> List[np.array]:
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
    audio_scale : float, default = AUDIO_SCALE_DEFAULT
        The audio scale to use for decoding.
        
    Returns
    -------
    List[np.array]
        List of decoded frames
    """

    # collect all subframes and create index mapping
    all_subframes, subframe_index_map = create_subframe_index_map(bottleneck = bottleneck)
    
    # batch decode all subframes (grouped by codebook level for efficiency)
    decoded_subframes = decode_subframes_batch_adaptive(
        bottleneck_subframes = all_subframes,
        entropy_coder = entropy_coder,
        model = model,
        batch_size = batch_size,
        audio_scale = audio_scale,
    )
    
    # organize back into frames
    decoded_frames = reconstruct_frames_from_subframes(
        decoded_subframes = decoded_subframes, bottleneck = bottleneck, subframe_index_map = subframe_index_map,
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
        The compressed subframe as (n_samples, codebook_level, dac_time_dimension, dac_codes, encoded_residuals).
        
    Returns
    -------
    float
        The size of the compressed subframe in bytes
    """

    # unpack bottleneck subframe
    n_samples, codebook_level, dac_time_dimension, codes, encoded_residuals = bottleneck_subframe

    # add size for storing number of samples
    total_size = MAXIMUM_BLOCK_SIZE_ASSUMPTION_BITS * 8

    # add size for DAC codes
    total_size += MAXIMUM_CODEBOOK_LEVEL_BITS / 8 # we can store the codebook level as one byte
    total_size += MAXIMUM_DAC_TIME_DIMENSION_ASSUMPTION_BITS * 8 # we can store the DAC time dimension as one byte, as a 1-byte unsigned integer
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
    data: np.array,
    entropy_coder: EntropyCoder,
    model: dac.model.dac.DAC,
    sample_rate: int,
    audio_scale: float = None,
    block_size: int = BLOCK_SIZE_DEFAULT,
    batch_size: int = BATCH_SIZE_DEFAULT,
) -> BOTTLENECK_TYPE:
    """
    Encode the original data into the bottleneck.

    Parameters
    ----------
    data : np.array
        The data to encode. Shape: (n_samples,) for mono, (n_samples, 2) for stereo.
    entropy_coder : EntropyCoder
        The entropy coder to use.
    model : dac.model.dac.DAC
        The DAC model to use.
    sample_rate : int
        The sample rate of the data.
    audio_scale : float, default = None
        The audio scale to use for encoding. If None, will be calculated using the data. If provided, will be used as is.
    block_size : int, default = BLOCK_SIZE_DEFAULT
        The block size to use for encoding.
    batch_size : int, default = BATCH_SIZE_DEFAULT
        The batch size to use for encoding.

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
        block_size = block_size,
        batch_size = batch_size,
        audio_scale = audio_scale,
    )
    
    # split data into frames
    frames = partition_data_into_frames(data = data, block_size = block_size)
    
    # collect subframes for batch processing
    subframes_metadata, subframes_data = collect_subframes_for_batch_processing(frames = frames)
    
    # batch encode subframes adaptively with optimized batching
    with torch.no_grad():
        encoded_subframes = encode_subframes_batch_optimized(
            subframes_data = subframes_data,
            entropy_coder = entropy_coder,
            model = model,
            sample_rate = sample_rate,
            batch_size = batch_size,
            audio_scale = audio_scale,
        )
    
    # organize subframes back into frame structure
    bottleneck = organize_subframes_into_frames(
        encoded_subframes = encoded_subframes,
        subframes_metadata = subframes_metadata,
        n_frames = len(frames),
    )

    # encode metadata
    batch_size_bits = encode_batch_size_bits(batch_size = batch_size)
    audio_scale_bits = encode_audio_scale_bits(audio_scale = audio_scale)

    # pack final bottleneck
    bottleneck = (0, batch_size_bits, audio_scale_bits, serialize(entropy_coder = entropy_coder), bottleneck)
    
    return bottleneck

def decode(
    bottleneck: BOTTLENECK_TYPE,
    model: dac.model.dac.DAC,
) -> np.array:
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
    np.array
        The decoded original data.
    """

    # unpack bottleneck
    _, batch_size_bits, audio_scale_bits, serialized_entropy_coder, bottleneck = bottleneck
    entropy_coder = deserialize(header_byte = serialized_entropy_coder)
    
    # decode metadata
    batch_size = decode_batch_size_bits(batch_size_bits = batch_size_bits)
    audio_scale = decode_audio_scale_bits(audio_scale_bits = audio_scale_bits)

    # batch decode frames with optimized codebook-level grouping
    with torch.no_grad():
        decoded_frames = decode_frames_batch_optimized(
            bottleneck = bottleneck,
            entropy_coder = entropy_coder,
            model = model,
            batch_size = batch_size,
            audio_scale = audio_scale,
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
    n_samples, codebook_level, dac_time_dimension, codes, encoded_residuals = subframe
    
    # write number of samples
    bitstream.write_bits(bits = n_samples, n = MAXIMUM_BLOCK_SIZE_ASSUMPTION_BITS)

    # write codebook level
    bitstream.write_bits(bits = codebook_level, n = MAXIMUM_CODEBOOK_LEVEL_BITS)
    
    # write dac_time_dimension (4-byte unsigned integer)  
    bitstream.write_bits(bits = dac_time_dimension, n = MAXIMUM_DAC_TIME_DIMENSION_ASSUMPTION_BITS)
    
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
    _, batch_size_bits, audio_scale_bits, serialized_entropy_coder, bottleneck = bottleneck

    # create bitstream
    bit_output = bitstream.BitOutputStream(path = path)

    # write codebook level
    bit_output.write_bit(bit = False) # we don't use codebook level for adaptive dac, so we write 0, but we include this first byte to make it easily compatible with Naive DAC
    
    # write batch size bits
    bit_output.write_bits(bits = batch_size_bits, n = MAXIMUM_BATCH_SIZE_BITS)
    
    # write audio scale bits
    bit_output.write_bits(bits = audio_scale_bits, n = MAXIMUM_AUDIO_SCALE_BITS)
    
    # align to byte
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
) -> BOTTLENECK_SUBFRAME_TYPE:
    """
    Read a subframe from a bitstream.

    Parameters
    ----------
    bitstream : bitstream.BitInputStream
        The bitstream to read from.

    Returns
    -------
    BOTTLENECK_SUBFRAME_TYPE
        The subframe.
    """

    # read n_samples
    n_samples = bitstream.read_bits(n = MAXIMUM_BLOCK_SIZE_ASSUMPTION_BITS)

    # read codebook level
    codebook_level = bitstream.read_bits(n = MAXIMUM_CODEBOOK_LEVEL_BITS)
    
    # read dac_time_dimension
    dac_time_dimension = bitstream.read_bits(n = MAXIMUM_DAC_TIME_DIMENSION_ASSUMPTION_BITS)
    
    # read codes array
    codes_bits_per_sample = bitstream.read_bits(n = 8)
    codes = [bitstream.read_bits(n = codes_bits_per_sample) for _ in range(codebook_level * dac_time_dimension)]
    codes = np.array(codes).astype(get_numpy_dtype_from_bit_size(bit_size = codes_bits_per_sample))
    codes = codes.reshape(codebook_level, dac_time_dimension)

    # read encoded residuals
    encoded_residuals_len = bitstream.read_bits(n = 32)
    encoded_residuals = bytes([bitstream.read_bits(n = 8) for _ in range(encoded_residuals_len)])
    
    # compile subframe
    subframe = (n_samples, codebook_level, dac_time_dimension, codes, encoded_residuals)
    
    return subframe

def read_frame(
    bitstream: bitstream.BitInputStream,
) -> BOTTLENECK_FRAME_TYPE:
    """
    Read a frame from a bitstream.

    Parameters
    ----------
    bitstream : bitstream.BitInputStream
        The bitstream to read from.

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
        subframe = read_subframe(bitstream = bitstream)
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

    # read redundant codebook level
    _ = bit_input.read_bit()
    
    # read batch size bits
    batch_size_bits = bit_input.read_bits(n = MAXIMUM_BATCH_SIZE_BITS)
    
    # read audio scale bits
    audio_scale_bits = bit_input.read_bits(n = MAXIMUM_AUDIO_SCALE_BITS)
    
    # align to byte
    bit_input.align_to_byte()

    # read serialized entropy coder
    serialized_entropy_coder = bit_input.read_byte()
    
    # read number of frames
    n_frames = bit_input.read_uint()
    
    # read each frame
    bottleneck = [None] * n_frames
    for i in range(n_frames):
        frame = read_frame(bitstream = bit_input)
        bottleneck[i] = frame

    # pack bottleneck
    bottleneck = (0, batch_size_bits, audio_scale_bits, serialized_entropy_coder, bottleneck)
    
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

    # get bottleneck
    bottleneck = encode(
        data = data,
        entropy_coder = entropy_coder,
        model = model,
        sample_rate = sample_rate,
        audio_scale = audio_scale,
        block_size = block_size,
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