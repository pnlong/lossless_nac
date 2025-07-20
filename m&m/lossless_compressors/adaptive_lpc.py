# README
# Phillip Long
# July 12, 2025

# Adaptive LPC Compressor.

# IMPORTS
##################################################

import numpy as np
from typing import List, Tuple
import multiprocessing
import functools
import logging

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, f"{dirname(dirname(realpath(__file__)))}/entropy_coders")
sys.path.insert(0, dirname(dirname(dirname(realpath(__file__)))))

from lossless_compressors import LosslessCompressor, partition_data_into_frames, BLOCK_SIZE_DEFAULT, INTERCHANNEL_DECORRELATION_DEFAULT, INTERCHANNEL_DECORRELATION_SCHEMES_MAP, REVERSE_INTERCHANNEL_DECORRELATION_SCHEMES_MAP, JOBS_DEFAULT
from entropy_coders import EntropyCoder
import naive_lpc
import utils

##################################################


# CONSTANTS
##################################################

LPC_COEFFICIENTS_DTYPE = naive_lpc.LPC_COEFFICIENTS_DTYPE # default LPC coefficients data type for quantization
MIN_ORDER = 1 # minimum LPC order
MAX_ORDER = 32 # maximum LPC order

##################################################


# BOTTLENECK TYPE
##################################################

# type of bottleneck subframes
BOTTLENECK_SUBFRAME_TYPE = Tuple[int, int, np.array, np.array, bytes] # bottleneck subframe type is a tuple of the number of samples, LPC order, warmup samples, LPC coefficients, and encoded residuals

# type of bottleneck frames
BOTTLENECK_FRAME_TYPE = Tuple[int, List[BOTTLENECK_SUBFRAME_TYPE]] # bottleneck frame type is a tuple of the interchannel decorrelation scheme index and list of subframes

# type of bottleneck
BOTTLENECK_TYPE = List[BOTTLENECK_FRAME_TYPE]

#################################################


# ADAPTIVE LPC ESTIMATOR FUNCTIONS
##################################################

def encode_subframe(subframe_data: np.array, entropy_coder: EntropyCoder) -> BOTTLENECK_SUBFRAME_TYPE:
    """
    Encode a single subframe of data using LPC.
    
    Parameters
    ----------
    subframe_data : np.array
        Subframe of data to encode.
    entropy_coder : EntropyCoder
        The entropy coder to use.
        
    Returns
    -------
    BOTTLENECK_SUBFRAME_TYPE
        Encoded subframe as (n_samples, lpc_order, warmup_samples, lpc_coefficients, encoded_residuals)
    """

    # try different orders
    best_order = MIN_ORDER
    best_size = float("inf")
    best_bottleneck_subframe = None
    for order in range(MIN_ORDER, MAX_ORDER + 1):
        candidate_bottleneck_subframe = naive_lpc.encode_subframe(subframe_data = subframe_data, entropy_coder = entropy_coder, order = order)
        candidate_size = naive_lpc.get_compressed_subframe_size(bottleneck_subframe = candidate_bottleneck_subframe) + 1 # add 1 for the order
        if candidate_size < best_size:
            best_order = order
            best_bottleneck_subframe = candidate_bottleneck_subframe

    # unpack best bottleneck subframe
    n_samples, warmup_samples, lpc_coefficients, encoded_residuals = best_bottleneck_subframe

    return n_samples, best_order, warmup_samples, lpc_coefficients, encoded_residuals


def decode_subframe(bottleneck_subframe: BOTTLENECK_SUBFRAME_TYPE, entropy_coder: EntropyCoder) -> np.array:
    """
    Decode a single subframe from LPC encoding.
    
    Parameters
    ----------
    bottleneck_subframe : BOTTLENECK_SUBFRAME_TYPE
        Encoded subframe as (n_samples, lpc_order, warmup_samples, lpc_coefficients, encoded_residuals)
    entropy_coder : EntropyCoder
        The entropy coder to use.
        
    Returns
    -------
    np.array
        Decoded subframe
    """

    # unpack bottleneck subframe
    n_samples, lpc_order, warmup_samples, lpc_coefficients, encoded_residuals = bottleneck_subframe

    # decode bottleneck subframe
    reconstructed_subframe = naive_lpc.decode_subframe(bottleneck_subframe = (n_samples, warmup_samples, lpc_coefficients, encoded_residuals), entropy_coder = entropy_coder)
    
    # ensure output is int32
    logging.debug(f"adaptive_lpc.decode_subframe: reconstructed_subframe dtype={reconstructed_subframe.dtype}, shape={reconstructed_subframe.shape}")
    reconstructed_subframe = reconstructed_subframe.astype(np.int32)
    
    return reconstructed_subframe


def encode_frame(frame_data: np.array, entropy_coder: EntropyCoder, interchannel_decorrelation: bool = INTERCHANNEL_DECORRELATION_DEFAULT) -> BOTTLENECK_FRAME_TYPE:
    """
    Encode a single frame of data using LPC.
    
    Parameters
    ----------
    frame_data : np.array
        Frame of data to encode. Shape: (n_samples,) for mono, (n_samples, 2) for stereo.
    entropy_coder : EntropyCoder
        The entropy coder to use.
    interchannel_decorrelation : bool, default = INTERCHANNEL_DECORRELATION_DEFAULT
        Whether to try different interchannel decorrelation schemes.
        
    Returns
    -------
    BOTTLENECK_FRAME_TYPE
        Encoded frame as (interchannel_decorrelation_scheme_index, list_of_subframes)
    """

    # handle mono case
    if len(frame_data.shape) == 1:
        bottleneck_subframe = encode_subframe(subframe_data = frame_data, entropy_coder = entropy_coder)
        return 0, [bottleneck_subframe]
    
    # handle stereo case
    left_channel = frame_data[:, 0]
    right_channel = frame_data[:, 1]
    
    # If interchannel decorrelation is off, just use right/left scheme
    if not interchannel_decorrelation:
        bottleneck_subframe1 = encode_subframe(subframe_data = left_channel, entropy_coder = entropy_coder)
        bottleneck_subframe2 = encode_subframe(subframe_data = right_channel, entropy_coder = entropy_coder)
        return 0, [bottleneck_subframe1, bottleneck_subframe2]
    
    # try all interchannel decorrelation schemes and pick the best
    best_interchannel_decorrelation_scheme_index = 0
    best_size = float("inf")
    best_bottleneck_frame = None
    for interchannel_decorrelation_scheme_index, interchannel_decorrelation_scheme_func in enumerate(INTERCHANNEL_DECORRELATION_SCHEMES_MAP):

        # apply the scheme
        channel1_transformed, channel2_transformed = interchannel_decorrelation_scheme_func(left = left_channel, right = right_channel)
        
        # encode both subframes
        bottleneck_subframe1 = encode_subframe(subframe_data = channel1_transformed, entropy_coder = entropy_coder)
        bottleneck_subframe2 = encode_subframe(subframe_data = channel2_transformed, entropy_coder = entropy_coder)
        
        # create bottleneck frame
        bottleneck_frame = (interchannel_decorrelation_scheme_index, [bottleneck_subframe1, bottleneck_subframe2])
        
        # calculate size
        size = get_compressed_frame_size(bottleneck_frame = bottleneck_frame)
        
        # update best if this is better
        if size < best_size:
            best_interchannel_decorrelation_scheme_index = interchannel_decorrelation_scheme_index
            best_size = size
            best_bottleneck_frame = bottleneck_frame
    
    return best_bottleneck_frame


def decode_frame(bottleneck_frame: BOTTLENECK_FRAME_TYPE, entropy_coder: EntropyCoder) -> np.array:
    """
    Decode a single frame from LPC encoding.
    
    Parameters
    ----------
    bottleneck_frame : BOTTLENECK_FRAME_TYPE
        Encoded frame as (interchannel_decorrelation_scheme_index, list_of_subframes)
    entropy_coder : EntropyCoder
        The entropy coder to use.
        
    Returns
    -------
    np.array
        Decoded frame
    """

    # unpack bottleneck frame
    interchannel_decorrelation_scheme_index, subframes = bottleneck_frame
    
    # handle mono case
    if len(subframes) == 1:
        mono_frame = decode_subframe(bottleneck_subframe = subframes[0], entropy_coder = entropy_coder)
        return mono_frame
    
    # handle stereo case
    channel1_decoded = decode_subframe(bottleneck_subframe = subframes[0], entropy_coder = entropy_coder)
    channel2_decoded = decode_subframe(bottleneck_subframe = subframes[1], entropy_coder = entropy_coder)
    
    # reverse the interchannel decorrelation
    reverse_func = REVERSE_INTERCHANNEL_DECORRELATION_SCHEMES_MAP[interchannel_decorrelation_scheme_index]
    left_channel, right_channel = reverse_func(channel1 = channel1_decoded, channel2 = channel2_decoded)
    
    # combine channels
    stereo_frame = np.stack((left_channel, right_channel), axis = -1)
    
    return stereo_frame


def get_compressed_subframe_size(bottleneck_subframe: BOTTLENECK_SUBFRAME_TYPE) -> int:
    """
    Get the size of a compressed subframe in bytes.

    Parameters
    ----------
    bottleneck_subframe : BOTTLENECK_SUBFRAME_TYPE
        The compressed subframe as (n_samples, lpc_order, warmup_samples, lpc_coefficients, encoded_residuals)
        
    Returns
    -------
    int
        The size of the compressed subframe in bytes
    """

    # unpack bottleneck subframe
    n_samples, lpc_order, warmup_samples, lpc_coefficients, encoded_residuals = bottleneck_subframe

    # add size for storing number of samples
    total_size = utils.MAXIMUM_BLOCK_SIZE_ASSUMPTION_BYTES

    # add size for storing LPC order
    total_size += 1

    # add size for storing warmup samples
    total_size += warmup_samples.nbytes

    # add size for storing LPC coefficients
    total_size += lpc_coefficients.nbytes

    # add size for encoded residuals
    total_size += len(encoded_residuals)

    return total_size


def get_compressed_frame_size(bottleneck_frame: BOTTLENECK_FRAME_TYPE) -> int:
    """
    Get the size of a compressed frame in bytes.
    
    Parameters
    ----------
    bottleneck_frame : BOTTLENECK_FRAME_TYPE
        The compressed frame as (interchannel_decorrelation_scheme_index, list_of_subframes)
        
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


def encode_frame_worker(frame_data: np.array, entropy_coder: EntropyCoder, interchannel_decorrelation: bool) -> BOTTLENECK_FRAME_TYPE:
    """
    Worker function for multiprocessing frame encoding.
    """
    return encode_frame(frame_data = frame_data, entropy_coder = entropy_coder, interchannel_decorrelation = interchannel_decorrelation)


def decode_frame_worker(bottleneck_frame: BOTTLENECK_FRAME_TYPE, entropy_coder: EntropyCoder) -> np.array:
    """
    Worker function for multiprocessing frame decoding.
    """
    return decode_frame(bottleneck_frame = bottleneck_frame, entropy_coder = entropy_coder)

##################################################


# LOSSLESS COMPRESSOR INTERFACE
##################################################

class AdaptiveLPC(LosslessCompressor):
    """
    Adaptive LPC Compressor.
    """

    def __init__(self, entropy_coder: EntropyCoder, block_size: int = BLOCK_SIZE_DEFAULT, interchannel_decorrelation: bool = INTERCHANNEL_DECORRELATION_DEFAULT, jobs: int = JOBS_DEFAULT):
        """
        Initialize the Adaptive LPC Compressor.

        Parameters
        ----------
        entropy_coder : EntropyCoder
            The entropy coder to use.
        block_size : int, default = BLOCK_SIZE_DEFAULT
            The block size to use for encoding.
        interchannel_decorrelation : bool, default = INTERCHANNEL_DECORRELATION_DEFAULT
            Whether to decorrelate channels.
        jobs : int, default = JOBS_DEFAULT
            The number of jobs to use for multiprocessing.
        """
        self.entropy_coder = entropy_coder
        self.block_size = block_size
        assert self.block_size > 0 and self.block_size <= utils.MAXIMUM_BLOCK_SIZE_ASSUMPTION, f"Block size must be positive and less than or equal to {utils.MAXIMUM_BLOCK_SIZE_ASSUMPTION}."
        self.interchannel_decorrelation = interchannel_decorrelation
        self.jobs = jobs
        assert self.jobs > 0 and self.jobs <= multiprocessing.cpu_count(), f"Number of jobs must be positive and less than or equal to {multiprocessing.cpu_count()}."
        
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
        
        # use multiprocessing to encode frames in parallel
        with multiprocessing.Pool(processes = self.jobs) as pool:
            worker_func = functools.partial(
                encode_frame_worker,
                entropy_coder = self.entropy_coder,
                interchannel_decorrelation = self.interchannel_decorrelation,
            )
            bottleneck = pool.map(func = worker_func, iterable = frames)
        
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

        # use multiprocessing to decode frames in parallel
        with multiprocessing.Pool(processes = self.jobs) as pool:
            worker_func = functools.partial(
                decode_frame_worker,
                entropy_coder = self.entropy_coder,
            )
            decoded_frames = pool.map(func = worker_func, iterable = bottleneck)
        
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