# README
# Phillip Long
# July 12, 2025

# DAC Compressor.

# IMPORTS
##################################################

import numpy as np
from typing import List, Tuple, Dict, Any
import warnings
import torch
from audiotools import AudioSignal
from math import ceil, log2

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(dirname(realpath(__file__))))
sys.path.insert(0, f"{dirname(dirname(dirname(realpath(__file__))))}/dac") # for dac import

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

BLOCK_SIZE_DEFAULT = 4096 # default block size for partitioning data into frames
MAXIMUM_BLOCK_SIZE_ASSUMPTION = (2 ** 16) - 1 # maximum block size value (we expect the block size to be this value or lesser)
MAXIMUM_BLOCK_SIZE_ASSUMPTION_BITS = ceil(log2(MAXIMUM_BLOCK_SIZE_ASSUMPTION + 1) / 8) * 8 # convert into number of bits
MAXIMUM_DAC_TIME_DIMENSION_ASSUMPTION = (2 ** 8) - 1 # maximum DAC time dimension value (we expect the DAC time dimension to be this value or lesser)
MAXIMUM_DAC_TIME_DIMENSION_ASSUMPTION_BITS = ceil(log2(MAXIMUM_DAC_TIME_DIMENSION_ASSUMPTION + 1) / 8) * 8 # convert into number of bits
MAXIMUM_BATCH_SIZE = (2 ** 7) # maximum batch size for GPU processing
MAXIMUM_BATCH_SIZE_BITS = ceil(log2(log2(MAXIMUM_BATCH_SIZE) + 1)) # convert into number of bits
BATCH_SIZE_DEFAULT = 128 # optimal batch size for GPU processing
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

def encode_batch_size_bits(batch_size: int = BATCH_SIZE_DEFAULT) -> int:
    """
    Encode batch size as 3 bits (0-7) where actual batch size = 2^x.
    
    Parameters
    ----------
    batch_size : int, default = BATCH_SIZE_DEFAULT
        The actual batch size to encode.
        
    Returns
    -------
    int
        Encoded batch size bits (0-7).
    """
    batch_size_bits = int(log2(batch_size))
    return batch_size_bits

def decode_batch_size_bits(batch_size_bits: int = int(log2(BATCH_SIZE_DEFAULT))) -> int:
    """
    Decode batch size from 3 bits (0-7) where actual batch size = 2^x.
    
    Parameters
    ----------
    batch_size_bits : int, default = int(log2(BATCH_SIZE_DEFAULT))
        Encoded batch size bits (0-7).
        
    Returns
    -------
    int
        Actual batch size.
    """
    return 2 ** batch_size_bits

encode_audio_scale_bits = constants.encode_audio_scale_bits

decode_audio_scale_bits = constants.decode_audio_scale_bits

##################################################


# FRAME OPERATIONS FUNCTIONS
##################################################

def pad_to_batch_size(items: List, target_batch_size: int, pad_item: Any = None) -> List:
    """
    Pad a list to the target batch size with padding items.
    
    Parameters
    ----------
    items : List
        List of items to pad.
    target_batch_size : int
        Target batch size.
    pad_item : Any, default = None
        Item to use for padding.
        
    Returns
    -------
    List
        Padded list.
    """

    # truncate if too long
    if len(items) >= target_batch_size:
        return items[:target_batch_size]
    
    # add padding
    padding_needed = target_batch_size - len(items)
    padded_items = items + ([pad_item] * padding_needed)
    
    return padded_items

def partition_data_into_frames(data: np.ndarray, block_size: int = BLOCK_SIZE_DEFAULT) -> List[np.ndarray]:
    """
    Partition the data into frames.

    Parameters
    ----------
    data : np.ndarray
        The data to partition.
    block_size : int, default = BLOCK_SIZE_DEFAULT
        The block size to use for partitioning.

    Returns
    -------
    List[np.ndarray]
        The partitioned data.
    """
    n_samples = len(data)
    n_frames = ceil(n_samples / block_size)
    frames = [None] * n_frames
    for i in range(n_frames):
        start_index = i * block_size
        end_index = min(start_index + block_size, n_samples)
        frames[i] = data[start_index:end_index]
    return frames

##################################################


# BATCH PREPARATION FUNCTIONS
##################################################

def pad_subframes_to_batch(subframes: List[np.ndarray], target_length: int = None) -> np.ndarray:
    """
    Pad subframes to target length and stack into batch.
    
    Parameters
    ----------
    subframes : List[np.ndarray]
        List of subframes with potentially different lengths.
    target_length : int, default = None
        Target length to pad all subframes to, if None, use the maximum length of the subframes.
        
    Returns
    -------
    np.ndarray
        Batched subframes of shape (batch_size, target_length)
    """

    # if target length is not provided, use the maximum length of the subframes
    if target_length is None:
        target_length = max(len(subframe) for subframe in subframes)

    # pad subframes to target length
    padded_subframes = []
    for subframe in subframes:
        if len(subframe) < target_length:
            padded = np.pad(subframe, (0, target_length - len(subframe)), mode = "constant", constant_values = 0)
        else:
            padded = subframe[:target_length] # truncate if longer
        padded_subframes.append(padded)
    
    # stack subframes into batch
    batch = np.stack(padded_subframes, axis = 0)

    return batch

def collect_subframes_for_batch_processing(frames: List[np.ndarray]) -> Tuple[List[Dict[str, Any]], List[np.ndarray]]:
    """
    Collect subframes from all frames.
    
    Parameters
    ----------
    frames : List[np.ndarray]
        List of frames to process.
        
    Returns
    -------
    Tuple[List[Dict[str, Any]], List[np.ndarray]]
        Metadata for each subframe and list of subframes ready for batch processing
    """

    # initialize metadata and data lists
    subframes_metadata = []
    subframes_data = []
    
    # process each frame
    for i, frame_data in enumerate(frames):

        # handle mono case
        if len(frame_data.shape) == 1:
            subframes_metadata.append({
                "frame_idx": i,
                "channel_idx": 0,
                "original_length": len(frame_data),
            })
            subframes_data.append(frame_data)

        # handle stereo case
        else:
            left_channel = frame_data[:, 0]
            right_channel = frame_data[:, 1]
            subframes_metadata.extend([
                {
                    "frame_idx": i,
                    "channel_idx": 0,
                    "original_length": len(left_channel),
                },
                {
                    "frame_idx": i,
                    "channel_idx": 1,
                    "original_length": len(right_channel),
                }
            ])
            subframes_data.extend([left_channel, right_channel])
    
    return subframes_metadata, subframes_data

##################################################


# DAC ENCODING/DECODING FUNCTIONS
##################################################

def batch_dac_encode_full(subframes_batch: np.ndarray, model: dac.model.dac.DAC, sample_rate: int, codebook_level: int, audio_scale: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Encode batch of subframes through DAC and return both codes and reconstructions.
    
    Parameters
    ----------
    subframes_batch : np.ndarray
        Batch of subframes of shape (batch_size, max_length).
    model : dac.model.dac.DAC
        The DAC model to use for lossy estimation.
    sample_rate : int
        Sample rate of the audio.
    codebook_level : int
        The number of codebooks to use for DAC encoding.
    audio_scale : float
        The audio scale to use for DAC encoding.
        
    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        DAC codes and approximate reconstructions
    """
    
    # convert to AudioSignal format for batch processing
    subframes_batch = np.expand_dims(subframes_batch, axis = 1) # add channel dimension
    batch_audio = AudioSignal(audio_path_or_array = subframes_batch, sample_rate = sample_rate)
    
    # preprocess batch
    x = model.preprocess(
        audio_data = torch.from_numpy(convert_audio_fixed_to_floating(waveform = batch_audio.audio_data.numpy(), audio_scale = audio_scale)).to(model.device), # convert to float with correct audio scaling
        sample_rate = sample_rate,
    ).float()
    
    # batch encode
    _, codes_batch, _, _, _ = model.encode(x)
    codes_batch = codes_batch[:, :codebook_level, :] # truncate codes to desired codebook level
    
    # batch decode for approximate reconstruction
    z_batch = model.quantizer.from_codes(codes = codes_batch)[0].detach() # get z from codes
    approximate_batch = model.decode(z_batch) # decode z to approximate reconstruction
    
    # convert back to numpy and remove batch/channel dimensions
    codes_batch = codes_batch.detach().cpu().numpy()
    approximate_batch = approximate_batch.squeeze(dim = 1).detach().cpu().numpy() # remove channel dimension, which is 1
    
    return codes_batch, approximate_batch

def batch_dac_encode_codes_only(subframes_batch: np.ndarray, model: dac.model.dac.DAC, sample_rate: int, codebook_level: int, audio_scale: float) -> np.ndarray:
    """
    Encode batch of subframes through DAC and return only codes.
    
    Parameters
    ----------
    subframes_batch : np.ndarray
        Batch of subframes of shape (batch_size, max_length).
    model : dac.model.dac.DAC
        The DAC model to use for lossy estimation.
    sample_rate : int
        Sample rate of the audio.
    codebook_level : int
        The number of codebooks to use for DAC encoding.
    audio_scale : float
        The audio scale to use for DAC encoding.
        
    Returns
    -------
    np.ndarray
        DAC codes only (no decoding)
    """
    
    # convert to AudioSignal format for batch processing
    subframes_batch = np.expand_dims(subframes_batch, axis = 1) # add channel dimension
    batch_audio = AudioSignal(audio_path_or_array = subframes_batch, sample_rate = sample_rate)
    
    # preprocess batch
    x = model.preprocess(
        audio_data = torch.from_numpy(convert_audio_fixed_to_floating(waveform = batch_audio.audio_data.numpy(), audio_scale = audio_scale)).to(model.device), # convert to float with correct audio scaling
        sample_rate = sample_rate,
    ).float()
    
    # batch encode only
    _, codes_batch, _, _, _ = model.encode(x)
    codes_batch = codes_batch[:, :codebook_level, :] # truncate codes to desired codebook level
    
    # convert back to numpy
    codes_batch = codes_batch.detach().cpu().numpy()
    
    return codes_batch

def batch_dac_decode(
    codes_batch: np.array,
    model: dac.model.dac.DAC,
) -> np.ndarray:
    """
    Batch decode multiple codes using the DAC model.

    Parameters
    ----------
    codes_batch : np.array
        Batch of codes to decode. Shape: (batch_size, codebook_level, time_dimension).
    model : dac.model.dac.DAC
        The DAC model to use for decoding.

    Returns
    -------
    np.ndarray
        Batch of approximate reconstructions
    """

    # convert to tensor and add batch dimension if needed
    codes_tensor = torch.from_numpy(codes_batch).to(model.device)
    if len(codes_tensor.shape) == 2:
        codes_tensor = codes_tensor.unsqueeze(0) # add batch dimension if needed
    
    # batch decode
    z_batch = model.quantizer.from_codes(codes = codes_tensor)[0].detach() # get z from codes
    approximate_batch = model.decode(z_batch) # decode z to approximate reconstruction
    
    # convert back to numpy and remove batch/channel dimensions
    approximate_batch = approximate_batch.squeeze(dim = 1).detach().cpu().numpy() # remove channel dimension, which is 1
    
    return approximate_batch

##################################################


# FRAME ORGANIZATION FUNCTIONS
##################################################

def organize_subframes_into_frames(encoded_subframes: List, subframes_metadata: List[Dict[str, Any]], n_frames: int) -> List:
    """
    Organize encoded subframes back into frame structure.
    
    Parameters
    ----------
    encoded_subframes : List
        List of encoded subframes.
    subframes_metadata : List[Dict[str, Any]]
        Metadata for each subframe.
    n_frames : int
        Number of frames.
        
    Returns
    -------
    List
        Organized bottleneck frames
    """

    # initialize list of bottleneck frames
    bottleneck_frames = []
    
    # group subframes by frame
    subframes_by_frame = [[] for _ in range(n_frames)]
    
    # process each subframe
    for subframe, metadata in zip(encoded_subframes, subframes_metadata):
        i = metadata["frame_idx"]
        subframes_by_frame[i].append(subframe) # add subframe to frame
    
    # process each frame
    for i in range(n_frames):
        frame_subframes = subframes_by_frame[i] # get subframes for frame
        bottleneck_frames.append(frame_subframes)
        
    return bottleneck_frames

def reconstruct_frames_from_subframes(decoded_subframes: List[np.ndarray], bottleneck: List, subframe_index_map: Dict) -> List[np.ndarray]:
    """
    Reconstruct frames from decoded subframes using index mapping.
    
    Parameters
    ----------
    decoded_subframes : List[np.ndarray]
        List of decoded subframes.
    bottleneck : List
        Original bottleneck structure for frame organization.
    subframe_index_map : Dict
        Mapping from (frame_idx, subframe_idx) to global subframe index.
        
    Returns
    -------
    List[np.ndarray]
        List of decoded frames
    """

    # organize back into frames
    decoded_frames = [None] * len(bottleneck)
    for i, subframes in enumerate(bottleneck):

        # collect subframes for this frame
        frame_subframes = []
        for j in range(len(subframes)):
            subframe_index = subframe_index_map[(i, j)] # get index of subframe in all_subframes
            frame_subframes.append(decoded_subframes[subframe_index])
        
        # reconstruct frame
        if len(frame_subframes) == 1: # mono case
            decoded_frames[i] = frame_subframes[0] # add mono frame to list of decoded frames
        else: # stereo case
            decoded_frames[i] = np.stack(frame_subframes, axis = -1) # add stereo frame to list of decoded frames
        decoded_frames[i] = decoded_frames[i].astype(np.int32) # ensure output is int32
    
    return decoded_frames

def create_subframe_index_map(bottleneck: List) -> Tuple[List, Dict]:
    """
    Create mapping from frame/subframe indices to global subframe list.
    
    Parameters
    ----------
    bottleneck : List
        Bottleneck structure containing frames and subframes.
        
    Returns
    -------
    Tuple[List, Dict]
        List of all subframes and mapping from (frame_idx, subframe_idx) to global index
    """

    # collect all subframes and create index mapping
    all_subframes = [] # list of all subframes
    subframe_index_map = {} # map of (frame_idx, subframe_idx) to index in all_subframes
    for i, subframes in enumerate(bottleneck):
        for j, subframe in enumerate(subframes):
            subframe_index = len(all_subframes) # current index in all_subframes
            all_subframes.append(subframe) # add subframe to list of all subframes
            subframe_index_map[(i, j)] = subframe_index # create mapping for O(1) lookup
    
    return all_subframes, subframe_index_map

##################################################


# DATA VALIDATION FUNCTIONS
##################################################

validate_input_data = constants.validate_input_data

validate_output_data = constants.validate_output_data

def validate_input_args(
    sample_rate: int,
    model: dac.model.dac.DAC,
    block_size: int = BLOCK_SIZE_DEFAULT,
    codebook_level: int = None,
    batch_size: int = BATCH_SIZE_DEFAULT,
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
    block_size : int, default = BLOCK_SIZE_DEFAULT
        The block size to use for partitioning data into frames.
    codebook_level : int, default = None
        The number of codebooks to use for DAC encoding.
    batch_size : int, default = BATCH_SIZE_DEFAULT
        The batch size to use for encoding.
    audio_scale : float, default = AUDIO_SCALE_DEFAULT
        The audio scale to use for encoding.
    """
    assert sample_rate == model.sample_rate, f"Sample rate must match the sample rate of the model. Model sample rate: {model.sample_rate}, provided sample rate: {sample_rate}."
    assert block_size > 0 and block_size <= MAXIMUM_BLOCK_SIZE_ASSUMPTION, f"Block size must be positive and less than or equal to {MAXIMUM_BLOCK_SIZE_ASSUMPTION}."
    assert batch_size > 0 and batch_size <= MAXIMUM_BATCH_SIZE and log2(batch_size) % 1 == 0, f"Batch size must be a power of 2 between 1 and {MAXIMUM_BATCH_SIZE}."
    assert audio_scale > 0 and audio_scale <= MAXIMUM_AUDIO_SCALE and ((log2(audio_scale) + 1) / 8) % 1 == 0, f"Audio scale must be less than or equal to {MAXIMUM_AUDIO_SCALE} and satisfy 2^((8*x)-1) = audio_scale for some integer x."
    if codebook_level is not None: # if codebook level is provided, validate it
        assert codebook_level > 0 and codebook_level <= MAXIMUM_CODEBOOK_LEVEL, f"Codebook level must be between 1 and {MAXIMUM_CODEBOOK_LEVEL}."

##################################################