# README
# Phillip Long
# July 6, 2025

# Lossless Compressor interface.

# IMPORTS
##################################################

from abc import ABC, abstractmethod
import numpy as np
from typing import List, Any, Tuple
from math import ceil
from multiprocessing import cpu_count

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(dirname(realpath(__file__))))
sys.path.insert(0, f"{dirname(dirname(realpath(__file__)))}/entropy_coders")

from entropy_coders import EntropyCoder

##################################################


# CONSTANTS
##################################################

BLOCK_SIZE_DEFAULT = 2048 # default block size for lossless compressors
INTERCHANNEL_DECORRELATION_DEFAULT = True # default interchannel decorrelation
JOBS_DEFAULT = int(cpu_count() / 4) # default number of jobs for multiprocessing

##################################################


# LOSSLESS COMPRESSOR INTERFACE
##################################################

class LosslessCompressor(ABC):
    """
    Abstract base class for lossless compressors.
    Processes batched data into a bottleneck, which is batched to allow for parallel processing.
    The bottleneck is achieved by using a lossy estimator to approximate the data, and then 
    encoding the residuals with the given entropy coder.
    Assumes that the input waveform comes in one of two shapes: (n_samples,) if mono, (n_samples, 2) if stereo.
    Assumes that the input waveform is raw, 32-bit integer values.
    """
    
    @property
    def type_(self) -> str:
        """
        The type of lossless compressor. Defaults to the class name.

        Returns
        -------
        str
            The type of lossless compressor.
        """
        return self.__class__.__name__

    @abstractmethod 
    def __init__(self, entropy_coder: EntropyCoder, block_size: int = BLOCK_SIZE_DEFAULT, interchannel_decorrelation: bool = INTERCHANNEL_DECORRELATION_DEFAULT):
        """
        Initialize the lossless compressor.

        Parameters
        ----------
        entropy_coder : EntropyCoder
            The entropy coder to use.
        block_size : int, default = BLOCK_SIZE_DEFAULT
            The block size to use for encoding.
        interchannel_decorrelation : bool, default = INTERCHANNEL_DECORRELATION_DEFAULT
            Whether to try different interchannel decorrelation schemes.
        """
        self.entropy_coder = entropy_coder
        self.block_size = block_size

    @abstractmethod
    def encode(self, data: np.array) -> Any:
        """
        Encode the original data into the bottleneck.

        Parameters
        ----------
        data : np.array
            The data to encode.
        
        Returns
        -------
        Any
            The bottleneck.
        """
        pass

    @abstractmethod
    def decode(self, bottleneck: Any) -> np.array:
        """
        Decode the bottleneck into the original data.

        Parameters
        ----------
        bottleneck : Any
            The bottleneck to decode.

        Returns
        -------
        np.array
            The decoded original data.
        """
        pass

    @abstractmethod
    def get_compressed_size(self, bottleneck: Any) -> int:
        """
        Get the size of the bottleneck in bytes.

        Parameters
        ----------
        bottleneck : Any
            The bottleneck.

        Returns 
        -------
        int
            The size of the bottleneck in bytes.
        """
        pass
        
##################################################


# HELPER FUNCTIONS
##################################################

def partition_data_into_frames(data: np.array, block_size: int = BLOCK_SIZE_DEFAULT) -> List[np.array]:
    """
    Partition the data into frames.

    Parameters
    ----------
    data : np.array
        The data to partition.
    block_size : int, default = BLOCK_SIZE_DEFAULT
        The block size to use for partitioning.

    Returns
    -------
    List[np.array]
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


# INTERCHANNEL DECORRELATION SCHEMES
##################################################

def apply_left_right(left: np.array, right: np.array) -> Tuple[np.array, np.array]:
    """Apply left/right scheme (no decorrelation)."""
    return left, right

def apply_left_side(left: np.array, right: np.array) -> Tuple[np.array, np.array]:
    """Apply left/side scheme."""
    side = right.astype(np.int64) - left.astype(np.int64) # use int64 to prevent overflow during subtraction
    return left, side

def apply_right_side(left: np.array, right: np.array) -> Tuple[np.array, np.array]:
    """Apply right/side scheme."""
    side = left.astype(np.int64) - right.astype(np.int64) # use int64 to prevent overflow during subtraction
    return right, side

def apply_mid_side(left: np.array, right: np.array) -> Tuple[np.array, np.array]:
    """Apply mid/side scheme using nflac technique."""
    left_int64 = left.astype(np.int64)
    right_int64 = right.astype(np.int64)
    mid = ((left_int64 + right_int64) >> 1).astype(np.int32) # mid channel
    side = left_int64 - right_int64 # side channel
    return mid, side

def reverse_left_right(channel1: np.array, channel2: np.array) -> Tuple[np.array, np.array]:
    """Reverse left/right scheme."""
    return channel1.astype(np.int32), channel2.astype(np.int32) # left, right -> left, right

def reverse_left_side(channel1: np.array, channel2: np.array) -> Tuple[np.array, np.array]:
    """Reverse left/side scheme."""
    right = channel2.astype(np.int64) + channel1.astype(np.int64) # use int64 to prevent overflow during addition
    return channel1.astype(np.int32), right.astype(np.int32) # left, right

def reverse_right_side(channel1: np.array, channel2: np.array) -> Tuple[np.array, np.array]:
    """Reverse right/side scheme."""
    left = channel2.astype(np.int64) + channel1.astype(np.int64) # use int64 to prevent overflow during addition
    return left.astype(np.int32), channel1.astype(np.int32) # left, right

def reverse_mid_side(channel1: np.array, channel2: np.array) -> Tuple[np.array, np.array]:
    """Reverse mid/side scheme using nflac technique."""
    mid = channel1.astype(np.int64)
    side = channel2.astype(np.int64)
    left = mid + ((side + 1) >> 1) # left channel
    right = mid - (side >> 1) # right channel
    return left.astype(np.int32), right.astype(np.int32) # left, right

# interchannel decorrelation scheme mapping
INTERCHANNEL_DECORRELATION_SCHEMES_MAP = [apply_left_right, apply_left_side, apply_right_side, apply_mid_side]
REVERSE_INTERCHANNEL_DECORRELATION_SCHEMES_MAP = [reverse_left_right, reverse_left_side, reverse_right_side, reverse_mid_side]

##################################################