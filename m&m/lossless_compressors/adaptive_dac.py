# README
# Phillip Long
# July 12, 2025

# Adaptive DAC Compressor.

# IMPORTS
##################################################

import numpy as np
from typing import List, Tuple, Dict, Any
import warnings
import multiprocessing
import torch
from audiotools import AudioSignal
import logging

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, f"{dirname(dirname(realpath(__file__)))}/entropy_coders")
sys.path.insert(0, dirname(dirname(dirname(realpath(__file__)))))
sys.path.insert(0, f"{dirname(dirname(dirname(realpath(__file__))))}/dac") # for dac import

from lossless_compressors import LosslessCompressor, partition_data_into_frames, BLOCK_SIZE_DEFAULT, INTERCHANNEL_DECORRELATION_DEFAULT, INTERCHANNEL_DECORRELATION_SCHEMES_MAP, REVERSE_INTERCHANNEL_DECORRELATION_SCHEMES_MAP, JOBS_DEFAULT
from entropy_coders import EntropyCoder
import naive_dac
import utils
import dac

# ignore deprecation warning from pytorch
warnings.filterwarnings(action = "ignore", message = "torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm")

##################################################


# CONSTANTS
##################################################

DAC_PATH = "/home/pnlong/.cache/descript/dac/weights_44khz_8kbps_0.0.1.pth" # path to descript audio codec pretrained model
MAXIMUM_CODEBOOK_LEVEL = naive_dac.MAXIMUM_CODEBOOK_LEVEL # maximum codebook level for the pretrained descript audio codec model
DEVICE_DEFAULT = naive_dac.DEVICE_DEFAULT # default device for running the DAC model
BATCH_SIZE_DEFAULT = naive_dac.BATCH_SIZE_DEFAULT # optimal batch size for GPU processing

##################################################


# BOTTLENECK TYPE
##################################################

# type of bottleneck subframes - now includes codebook level
BOTTLENECK_SUBFRAME_TYPE = Tuple[int, int, int, np.array, bytes] # bottleneck subframe type is a tuple of the number of samples, codebook level, DAC time dimension, DAC codes, and encoded residuals

# type of bottleneck frames
BOTTLENECK_FRAME_TYPE = Tuple[int, List[BOTTLENECK_SUBFRAME_TYPE]] # bottleneck frame type is a tuple of the interchannel decorrelation scheme index and list of subframes

# type of bottleneck
BOTTLENECK_TYPE = List[BOTTLENECK_FRAME_TYPE]

##################################################


# BATCH PROCESSING HELPER FUNCTIONS
##################################################

def pad_subframes_to_batch(subframes: List[np.array], target_length: int = None) -> np.array:
    """
    Pad subframes to target length and stack into batch.
    
    Parameters
    ----------
    subframes : List[np.array]
        List of subframes with potentially different lengths.
    target_length : int, default = None
        Target length to pad all subframes to, if None, use the maximum length of the subframes.
        
    Returns
    -------
    np.array
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


def collect_subframes_for_batch_processing(frames: List[np.array], interchannel_decorrelation: bool = INTERCHANNEL_DECORRELATION_DEFAULT) -> Tuple[List[Dict[str, Any]], List[np.array]]:
    """
    Collect subframes from all frames and apply interchannel decorrelation.
    
    Parameters
    ----------
    frames : List[np.array]
        List of frames to process.
    interchannel_decorrelation : bool, default = INTERCHANNEL_DECORRELATION_DEFAULT
        Whether to apply different interchannel decorrelation schemes.
        
    Returns
    -------
    Tuple[List[Dict[str, Any]], List[np.array]]
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
                "interchannel_decorrelation_scheme_idx": 0,
                "is_mono": True
            })
            subframes_data.append(frame_data)

        # handle stereo case
        else:

            # handle stereo case and try all interchannel decorrelation schemes
            left_channel = frame_data[:, 0]
            right_channel = frame_data[:, 1]
            
            # if interchannel decorrelation is not enabled, just use left/right
            if not interchannel_decorrelation:
                subframes_metadata.extend([
                    {
                        "frame_idx": i,
                        "channel_idx": 0,
                        "original_length": len(left_channel),
                        "interchannel_decorrelation_scheme_idx": 0,
                        "is_mono": False
                    },
                    {
                        "frame_idx": i,
                        "channel_idx": 1,
                        "original_length": len(right_channel),
                        "interchannel_decorrelation_scheme_idx": 0,
                        "is_mono": False
                    }
                ])
                subframes_data.extend([left_channel, right_channel])

            # if interchannel decorrelation is enabled, try all interchannel decorrelation schemes
            else:

                # try all interchannel decorrelation schemes
                for interchannel_decorrelation_scheme_idx, interchannel_decorrelation_scheme_func in enumerate(INTERCHANNEL_DECORRELATION_SCHEMES_MAP):
                    channel1_transformed, channel2_transformed = interchannel_decorrelation_scheme_func(left = left_channel, right = right_channel) # apply the scheme
                    subframes_metadata.extend([
                        {
                            "frame_idx": i,
                            "channel_idx": 0,
                            "original_length": len(channel1_transformed),
                            "interchannel_decorrelation_scheme_idx": interchannel_decorrelation_scheme_idx,
                            "is_mono": False
                        },
                        {
                            "frame_idx": i,
                            "channel_idx": 1,
                            "original_length": len(channel2_transformed),
                            "interchannel_decorrelation_scheme_idx": interchannel_decorrelation_scheme_idx,
                            "is_mono": False
                        }
                    ])
                    subframes_data.extend([channel1_transformed, channel2_transformed])
    
    return subframes_metadata, subframes_data


def batch_dac_encode(subframes_batch: np.array, model: dac.model.dac.DAC, sample_rate: int, codebook_level: int) -> np.array:
    """
    Encode batch of subframes through DAC.
    
    Parameters
    ----------
    subframes_batch : np.array
        Batch of subframes of shape (batch_size, max_length).
    model : dac.model.dac.DAC
        The DAC model to use for lossy estimation.
    sample_rate : int
        Sample rate of the audio.
    codebook_level : int
        The number of codebooks to use for DAC encoding.
        
    Returns
    -------
    np.array
        DAC codes only (no decoding)
    """
    
    # convert to AudioSignal format for batch processing
    subframes_batch = np.expand_dims(subframes_batch, axis = 1) # add channel dimension
    batch_audio = AudioSignal(audio_path_or_array = subframes_batch, sample_rate = sample_rate)
    
    # preprocess batch
    x = model.preprocess(
        audio_data = torch.from_numpy(naive_dac.convert_audio_fixed_to_floating(waveform = batch_audio.audio_data.numpy())).to(model.device), # convert to float with correct audio scaling
        sample_rate = sample_rate,
    ).float()
    
    # batch encode only
    _, codes_batch, _, _, _ = model.encode(audio_data = x)
    codes_batch = codes_batch[:, :codebook_level, :] # truncate codes to desired codebook level
    
    # convert back to numpy
    codes_batch = codes_batch.detach().cpu().numpy()
    
    return codes_batch


def batch_dac_decode(codes_batch: np.array, model: dac.model.dac.DAC) -> np.array:
    """
    Decode batch of DAC codes.
    
    Parameters
    ----------
    codes_batch : np.array
        Batch of DAC codes.
    model : dac.model.dac.DAC
        The DAC model to use for reconstruction.
        
    Returns
    -------
    np.array
        Batch of approximate reconstructions
    """

    # convert to tensor and add batch dimension if needed
    codes_tensor = torch.from_numpy(codes_batch).to(model.device)
    if len(codes_tensor.shape) == 2:
        codes_tensor = codes_tensor.unsqueeze(0) # add batch dimension if needed
    
    # batch decode
    z_batch = model.quantizer.from_codes(codes = codes_tensor)[0].detach() # get z from codes
    approximate_batch = model.decode(z = z_batch) # decode z to approximate reconstruction
    
    # convert back to numpy and remove batch/channel dimensions
    approximate_batch = approximate_batch.squeeze(dim = 1).detach().cpu().numpy() # remove channel dimension, which is 1
    
    return approximate_batch

##################################################


# OPTIMIZED ADAPTIVE DAC ESTIMATOR FUNCTIONS
##################################################

def encode_subframes_batch_adaptive(subframes_data: List[np.array], subframes_metadata: List[Dict[str, Any]], entropy_coder: EntropyCoder, model: dac.model.dac.DAC, sample_rate: int, batch_size: int = BATCH_SIZE_DEFAULT) -> List[BOTTLENECK_SUBFRAME_TYPE]:
    """
    Encode multiple subframes using adaptive DAC processing optimized for batching.
    
    Strategy:
    1. Batch encode all subframes at maximum codebook level
    2. For each codebook level, batch decode subframes that might use that level
    3. Select best codebook level for each subframe based on compression ratio
    
    Parameters
    ----------
    subframes_data : List[np.array]
        List of subframes to encode.
    subframes_metadata : List[Dict[str, Any]]
        Metadata for each subframe.
    entropy_coder : EntropyCoder
        The entropy coder to use.
    model : dac.model.dac.DAC
        The DAC model to use for lossy estimation.
    sample_rate : int
        Sample rate.
    batch_size : int, default = BATCH_SIZE_DEFAULT
        Batch size for processing.
        
    Returns
    -------
    List[BOTTLENECK_SUBFRAME_TYPE]
        List of encoded subframes with optimal codebook levels
    """

    # initialize results
    n_subframes = len(subframes_data)
    best_subframes = [None] * n_subframes
    best_sizes = [float("inf")] * n_subframes
    
    # batch encode all subframes with maximum codebook level
    full_codes_all = [None] * n_subframes
    for i in range(0, n_subframes, batch_size):
        batch_end = min(i + batch_size, n_subframes) # don't go past the end of the subframes
        batch_subframes = subframes_data[i:batch_end] # get batch of subframes
        padded_batch = pad_subframes_to_batch(subframes = batch_subframes) # pad subframes to same length for batch processing
        codes_batch = batch_dac_encode( # encode only
            subframes_batch = padded_batch,
            model = model,
            sample_rate = sample_rate,
            codebook_level = MAXIMUM_CODEBOOK_LEVEL,
        )
        for j, codes in enumerate(codes_batch): # store full codes for each subframe
            full_codes_all[i + j] = codes
    
    # for each codebook level, batch process all subframes
    for codebook_level in range(1, MAXIMUM_CODEBOOK_LEVEL + 1):
        
        # prepare batch data for this codebook level
        codes_batch_for_level = [None] * n_subframes
        
        # collect all subframes to test at this codebook level
        for i in range(n_subframes):
            full_codes = full_codes_all[i]
            codes_truncated = full_codes[:codebook_level, :]
            codes_batch_for_level[i] = codes_truncated
        
        # batch decode all subframes at this codebook level
        for batch_start in range(0, len(codes_batch_for_level), batch_size):

            # get batch of codes
            batch_end = min(batch_start + batch_size, len(codes_batch_for_level)) # don't go past the end of the subframes
            batch_codes = codes_batch_for_level[batch_start:batch_end]
            codes_batch_stacked = np.stack(batch_codes, axis = 0) # stack codes for batch processing
            
            # batch decode
            approximate_batch = batch_dac_decode(codes_batch = codes_batch_stacked, model = model)
            
            # evaluate each subframe in this batch
            for i, (subframe_idx, approximate, codes) in enumerate(zip(range(batch_start, batch_end), approximate_batch, batch_codes)):
                
                # get subframe data
                subframe_data = subframes_data[subframe_idx]
                
                # truncate approximate to original length
                original_length = len(subframe_data)
                approximate_truncated = approximate[:original_length]
                approximate_truncated = naive_dac.convert_audio_floating_to_fixed(
                    waveform = approximate_truncated,
                    output_dtype = subframe_data.dtype,
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


def decode_subframes_batch_adaptive(bottleneck_subframes: List[BOTTLENECK_SUBFRAME_TYPE], entropy_coder: EntropyCoder, model: dac.model.dac.DAC, batch_size: int = BATCH_SIZE_DEFAULT) -> List[np.array]:
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
                batch_residuals[i] = entropy_coder.decode(stream = encoded_residuals, num_samples = n_samples) # decode residuals
            
            # batch decode DAC codes
            codes_batch_stacked = np.stack(batch_codes, axis = 0) # stack codes for batch processing
            approximate_batch = batch_dac_decode(codes_batch = codes_batch_stacked, model = model) # batch decode
            
            # reconstruct each subframe in the batch
            for original_index, approximate, length, residuals in zip(batch_indices, approximate_batch, batch_lengths, batch_residuals):
                
                # truncate and convert approximate
                approximate_truncated = approximate[:length]
                logging.debug(f"adaptive_dac.decode_subframes_batch_adaptive: approximate_truncated dtype={approximate_truncated.dtype}, shape={approximate_truncated.shape}")
                logging.debug(f"adaptive_dac.decode_subframes_batch_adaptive: converting to int32 for compatibility")
                approximate_truncated = naive_dac.convert_audio_floating_to_fixed(
                    waveform = approximate_truncated, 
                    output_dtype = np.int32, # use int32 for compatibility
                )
                
                # reconstruct subframe
                reconstructed_subframe = approximate_truncated + residuals
                decoded_subframes[original_index] = reconstructed_subframe
    
    return decoded_subframes


def organize_subframes_into_frames(encoded_subframes: List[BOTTLENECK_SUBFRAME_TYPE], subframes_metadata: List[Dict[str, Any]], n_frames: int, interchannel_decorrelation: bool = INTERCHANNEL_DECORRELATION_DEFAULT) -> BOTTLENECK_TYPE:
    """
    Organize encoded subframes back into frame structure and select best interchannel scheme.
    
    Parameters
    ----------
    encoded_subframes : List[BOTTLENECK_SUBFRAME_TYPE]
        List of encoded subframes.
    subframes_metadata : List[Dict[str, Any]]
        Metadata for each subframe.
    n_frames : int
        Number of frames.
    interchannel_decorrelation : bool, default = INTERCHANNEL_DECORRELATION_DEFAULT
        Whether interchannel decorrelation was used.
        
    Returns
    -------
    BOTTLENECK_TYPE
        Organized bottleneck frames
    """

    # initialize list of bottleneck frames
    bottleneck_frames = []
    
    # group subframes by frame
    subframes_by_frame = [[] for _ in range(n_frames)]
    metadata_by_frame = [[] for _ in range(n_frames)]
    
    # process each subframe
    for subframe, metadata in zip(encoded_subframes, subframes_metadata):
        i = metadata["frame_idx"]
        subframes_by_frame[i].append(subframe) # add subframe to frame
        metadata_by_frame[i].append(metadata) # add metadata to frame
    
    # process each frame
    for i in range(n_frames):
        frame_subframes = subframes_by_frame[i] # get subframes for frame
        frame_metadata = metadata_by_frame[i] # get metadata for frame
        
        # handle mono case
        if len(frame_subframes) == 1 and frame_metadata[0]["is_mono"]:
            bottleneck_frames.append((0, frame_subframes)) # add mono frame to list of bottleneck frames
        
        # handle stereo case without interchannel decorrelation
        elif not interchannel_decorrelation:
            bottleneck_frames.append((0, frame_subframes)) # add stereo frame to list of bottleneck frames
        
        # handle stereo case with interchannel decorrelation - select best scheme
        else:

            # group subframes by interchannel decorrelation scheme
            interchannel_decorrelation_schemes = [[None, None] for _ in range(len(INTERCHANNEL_DECORRELATION_SCHEMES_MAP))] # group by interchannel scheme
            for subframe, metadata in zip(frame_subframes, frame_metadata):
                interchannel_decorrelation_scheme_idx = metadata["interchannel_decorrelation_scheme_idx"]
                interchannel_decorrelation_schemes[interchannel_decorrelation_scheme_idx][metadata["channel_idx"]] = subframe # add subframe to scheme
            
            # select best scheme based on compressed size
            best_interchannel_decorrelation_scheme_idx = 0 # initialize best scheme index
            best_size = float("inf") # initialize best size
            best_subframes = interchannel_decorrelation_schemes[best_interchannel_decorrelation_scheme_idx] # initialize best subframes
            for interchannel_decorrelation_scheme_idx, subframes in enumerate(interchannel_decorrelation_schemes):
                assert len(subframes) == 2, "Stereo subframes must have 2 channels." # ensure we have both channels
                size = sum(get_compressed_subframe_size(subframe) for subframe in subframes) # get compressed size of subframes
                if size < best_size: # update best scheme if current scheme is smaller
                    best_size = size # update best size
                    best_interchannel_decorrelation_scheme_idx = interchannel_decorrelation_scheme_idx # update best scheme index
                    best_subframes = subframes # update best subframes

            # add frame using best scheme to list of bottleneck frames
            bottleneck_frames.append((best_interchannel_decorrelation_scheme_idx, best_subframes))
    
    return bottleneck_frames


def decode_frames_batch_adaptive(bottleneck: BOTTLENECK_TYPE, entropy_coder: EntropyCoder, model: dac.model.dac.DAC, batch_size: int = BATCH_SIZE_DEFAULT) -> List[np.array]:
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
    List[np.array]
        List of decoded frames
    """

    # collect all subframes and create index mapping
    all_subframes = [] # list of all subframes
    subframe_index_map = {} # map of (frame_idx, subframe_idx, interchannel_decorrelation_scheme_idx) to index in all_subframes
    for i, (interchannel_decorrelation_scheme_idx, subframes) in enumerate(bottleneck):
        for j, subframe in enumerate(subframes):
            subframe_index = len(all_subframes) # current index in all_subframes
            all_subframes.append(subframe) # add subframe to list of all subframes
            subframe_index_map[(i, j, interchannel_decorrelation_scheme_idx)] = subframe_index # create mapping for O(1) lookup
    
    # batch decode all subframes (grouped by codebook level for efficiency)
    decoded_subframes = decode_subframes_batch_adaptive(bottleneck_subframes = all_subframes, entropy_coder = entropy_coder, model = model, batch_size = batch_size)
    
    # organize back into frames
    decoded_frames = [None] * len(bottleneck)
    for i, (interchannel_decorrelation_scheme_idx, subframes) in enumerate(bottleneck):

        # collect subframes for this frame
        frame_subframes = []
        for j in range(len(subframes)):
            subframe_index = subframe_index_map[(i, j, interchannel_decorrelation_scheme_idx)] # get index of subframe in all_subframes
            frame_subframes.append(decoded_subframes[subframe_index])
        
        # reconstruct frame
        if len(frame_subframes) == 1: # mono case
            decoded_frames[i] = frame_subframes[0] # add mono frame to list of decoded frames
        else: # stereo case, reverse interchannel decorrelation
            channel1, channel2 = frame_subframes[0], frame_subframes[1]
            reverse_func = REVERSE_INTERCHANNEL_DECORRELATION_SCHEMES_MAP[interchannel_decorrelation_scheme_idx]
            left_channel, right_channel = reverse_func(channel1 = channel1, channel2 = channel2)
            stereo_frame = np.stack((left_channel, right_channel), axis = -1)
            decoded_frames[i] = stereo_frame # add stereo frame to list of decoded frames
        decoded_frames[i] = decoded_frames[i].astype(np.int32) # ensure output is int32
    
    return decoded_frames


def get_compressed_subframe_size(bottleneck_subframe: BOTTLENECK_SUBFRAME_TYPE) -> int:
    """
    Get the size of a compressed subframe in bytes.

    Parameters
    ----------
    bottleneck_subframe : BOTTLENECK_SUBFRAME_TYPE
        The compressed subframe as (n_samples, codebook_level, dac_time_dimension, dac_codes, encoded_residuals).
        
    Returns
    -------
    int
        The size of the compressed subframe in bytes
    """

    # unpack bottleneck subframe
    n_samples, codebook_level, dac_time_dimension, codes, encoded_residuals = bottleneck_subframe

    # add size for storing number of samples
    total_size = utils.MAXIMUM_BLOCK_SIZE_ASSUMPTION_BYTES

    # add size for DAC codes
    total_size += 1 # we can store the codebook level as one byte
    total_size += 1 # we can store the DAC time dimension as one byte, as a 1-byte unsigned integer
    total_size += codes.nbytes

    # add size for encoded residuals
    total_size += len(encoded_residuals)

    return total_size


def get_compressed_frame_size(bottleneck_frame: BOTTLENECK_FRAME_TYPE) -> int:
    """
    Get the size of a compressed frame in bytes.
    
    Parameters
    ----------
    bottleneck_frame : BOTTLENECK_FRAME_TYPE
        The compressed frame as (interchannel_decorrelation_scheme_index, list_of_subframes).
        
    Returns
    -------
    int
        The size of the compressed frame in bytes
    """

    # initialize total size
    interchannel_decorrelation_scheme_index, subframes = bottleneck_frame
    total_size = 1 # we can store the interchannel decorrelation scheme index as one byte
    
    # add size for each subframe
    for bottleneck_subframe in subframes:
        total_size += get_compressed_subframe_size(bottleneck_subframe = bottleneck_subframe)
    
    return total_size

##################################################


# LOSSLESS COMPRESSOR INTERFACE
##################################################

class AdaptiveDAC(LosslessCompressor):
    """
    Adaptive DAC Compressor.
    """

    def __init__(self, entropy_coder: EntropyCoder, model_path: str = DAC_PATH, block_size: int = BLOCK_SIZE_DEFAULT, interchannel_decorrelation: bool = INTERCHANNEL_DECORRELATION_DEFAULT, device: str = DEVICE_DEFAULT, jobs: int = JOBS_DEFAULT, batch_size: int = BATCH_SIZE_DEFAULT):
        """
        Initialize the Adaptive DAC Compressor.

        Parameters
        ----------
        entropy_coder : EntropyCoder
            The entropy coder to use.
        model_path : str, default = DAC_PATH
            Path to the DAC model weights.
        block_size : int, default = BLOCK_SIZE_DEFAULT
            The block size to use for encoding.
        interchannel_decorrelation : bool, default = INTERCHANNEL_DECORRELATION_DEFAULT
            Whether to decorrelate channels.
        device : str, default = DEVICE_DEFAULT
            Device to run the DAC model on.
        jobs : int, default = JOBS_DEFAULT
            The number of jobs to use for multiprocessing.
        batch_size : int, default = BATCH_SIZE_DEFAULT
            Batch size for DAC processing.
        """
        self.entropy_coder = entropy_coder
        self.block_size = block_size
        assert self.block_size > 0 and self.block_size <= utils.MAXIMUM_BLOCK_SIZE_ASSUMPTION, f"Block size must be positive and less than or equal to {utils.MAXIMUM_BLOCK_SIZE_ASSUMPTION}."
        self.interchannel_decorrelation = interchannel_decorrelation
        self.device = device
        self.jobs = jobs
        assert self.jobs > 0 and self.jobs <= multiprocessing.cpu_count(), f"Number of jobs must be positive and less than or equal to {multiprocessing.cpu_count()}."
        self.batch_size = batch_size
        assert self.batch_size > 0, "Batch size must be positive."
        
        # load DAC model
        self.device = torch.device(device)
        self.model = dac.DAC.load(location = model_path).to(self.device)
        self.model.eval()
        
        # store sample rate from model
        self.sample_rate = self.model.sample_rate
        
    def encode(self, data: np.array) -> BOTTLENECK_TYPE:
        """
        Encode the original data into the bottleneck.

        Parameters
        ----------
        data : np.array
            The data to encode. Shape: (n_samples,) for mono, (n_samples, 2) for stereo.

        Returns
        -------
        BOTTLENECK_TYPE
            The bottleneck.
        """
        
        # ensure input is valid
        assert len(data.shape) == 1 or len(data.shape) == 2, "Data must be 1D or 2D."
        if len(data.shape) == 2:
            assert data.shape[1] == 2, "Data must be 2D with 2 channels."
        assert data.dtype == np.int32, "Data must be int32."
        
        # split data into frames
        frames = partition_data_into_frames(data = data, block_size = self.block_size)
        
        # collect subframes for batch processing
        subframes_metadata, subframes_data = collect_subframes_for_batch_processing(frames = frames, interchannel_decorrelation = self.interchannel_decorrelation)
        
        # batch encode subframes adaptively with optimized batching
        with torch.no_grad():
            encoded_subframes = encode_subframes_batch_adaptive(
                subframes_data = subframes_data,
                subframes_metadata = subframes_metadata,
                entropy_coder = self.entropy_coder,
                model = self.model,
                sample_rate = self.sample_rate,
                batch_size = self.batch_size,
            )
        
        # organize subframes back into frame structure
        bottleneck = organize_subframes_into_frames(
            encoded_subframes = encoded_subframes,
            subframes_metadata = subframes_metadata,
            n_frames = len(frames),
            interchannel_decorrelation = self.interchannel_decorrelation,
        )
        
        return bottleneck

    def decode(self, bottleneck: BOTTLENECK_TYPE) -> np.array:
        """
        Decode the bottleneck into the original data.

        Parameters
        ----------
        bottleneck : BOTTLENECK_TYPE
            The bottleneck to decode.

        Returns
        -------
        np.array
            The decoded original data.
        """

        # batch decode frames with optimized codebook-level grouping
        with torch.no_grad():
            decoded_frames = decode_frames_batch_adaptive(
                bottleneck = bottleneck,
                entropy_coder = self.entropy_coder,
                model = self.model,
                batch_size = self.batch_size,
            )
        
        # concatenate all decoded frames
        reconstructed_data = np.concatenate(decoded_frames, axis = 0)

        # ensure output is valid
        assert len(reconstructed_data.shape) == 1 or len(reconstructed_data.shape) == 2, "Reconstructed data must be 1D or 2D."
        if len(reconstructed_data.shape) == 2:
            assert reconstructed_data.shape[1] == 2, "Reconstructed data must be 2D with 2 channels."
        assert reconstructed_data.dtype == np.int32, "Reconstructed data must be int32."
        
        return reconstructed_data

    def get_compressed_size(self, bottleneck: BOTTLENECK_TYPE) -> int:
        """
        Get the size of the bottleneck in bytes.

        Parameters
        ----------
        bottleneck : BOTTLENECK_TYPE
            The bottleneck.

        Returns 
        -------
        int
            The size of the bottleneck in bytes.
        """

        # use multiprocessing to get compressed frame sizes in parallel
        with multiprocessing.Pool(processes = self.jobs) as pool:
            compressed_frame_sizes = pool.map(func = get_compressed_frame_size, iterable = bottleneck)

        # calculate total size as the sum of the compressed frame sizes
        total_size = sum(compressed_frame_sizes)

        return total_size

##################################################