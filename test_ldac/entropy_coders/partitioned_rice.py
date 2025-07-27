# README
# Phillip Long
# July 26, 2025

# Partitioned Rice Coder.

# IMPORTS
##################################################

import numpy as np
from math import log2

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))

from entropy_coder import EntropyCoder
import adaptive_rice

##################################################


# CONSTANTS
##################################################

MINIMUM_PARTITION_SIZE = 2 ** 8 # 256 samples
MINIMUM_PARTITION_SIZE_LOG = int(log2(MINIMUM_PARTITION_SIZE))
PARTITION_SIZE_DEFAULT = 2 ** 12 # 4096 samples

##################################################


# HELPER FUNCTIONS
##################################################

# validate partition size factor
def validate_partition_size_factor(
    partition_size_factor: int,
) -> None:
    """
    Validate a partition size factor.

    Parameters
    ----------
    partition_size_factor : int
        The partition size factor to validate.
    """
    assert partition_size_factor >= 0, f"Partition size factor must be non-negative, but got {partition_size_factor}"
    assert partition_size_factor % 1 == 0, f"Partition size factor must be an integer, but got {partition_size_factor}"

# convert partition size factor to partition size
def convert_partition_size_factor_to_partition_size(
    partition_size_factor: int,
) -> int:
    """
    Convert a partition size factor to a partition size.

    Parameters
    ----------
    partition_size_factor : int
        The partition size factor to convert.

    Returns
    ------- 
    int
        The partition size.
    """
    validate_partition_size_factor(partition_size_factor = partition_size_factor)
    partition_size = 2 ** (partition_size_factor + MINIMUM_PARTITION_SIZE_LOG)
    return partition_size

# convert partition size to partition size factor
def convert_partition_size_to_partition_size_factor(
    partition_size: int,
) -> int:
    """
    Convert a partition size to a partition size factor.

    Parameters
    ----------
    partition_size : int
        The partition size to convert.

    Returns
    -------
    int
        The partition size factor.
    """
    partition_size_factor = log2(partition_size) - MINIMUM_PARTITION_SIZE_LOG
    validate_partition_size_factor(partition_size_factor = partition_size_factor)
    partition_size_factor = int(partition_size_factor) # ensure integer
    return partition_size_factor

##################################################


# PARTITIONED RICE ENTROPY CODING FUNCTIONS
##################################################

def encode(
    nums: np.ndarray,
    partition_size: int,
) -> bytes:
    """
    Encode the data.

    Parameters
    ----------
    nums : np.ndarray
        The data to encode.
    partition_size : int
        The partition size to use.

    Returns
    -------
    bytes
        The encoded data.
    """
    
    # determine optimal rice parameter
    k = get_optimal_rice_parameter(nums = nums)

    # encode data and return stream
    if k == 0:
        stream = verbatim.encode(nums = nums)
    else:
        stream = naive_rice.encode(nums = nums, k = k)

    # add rice parameter to stream
    stream = bytes([k]) + stream # prepend rice parameter to stream

    return stream

def decode(
    stream: bytes,
    num_samples: int,
    partition_size: int,
) -> np.ndarray:
    """
    Decode the data.

    Parameters
    ----------
    stream : bytes
        The encoded data to decode.
    num_samples : int
        The number of samples to decode.
    partition_size : int
        The partition size to use.

    Returns
    -------
    np.ndarray
        The decoded data. 
    """
    
    # get rice parameter
    k = stream[0]

    # decode data and return
    if k == 0:
        nums = verbatim.decode(stream = stream[1:], num_samples = num_samples)
    else:
        nums = naive_rice.decode(stream = stream[1:], num_samples = num_samples, k = k)

    return nums

##################################################


# ENTROPY CODER INTERFACE
##################################################

class PartitionedRiceCoder(EntropyCoder):
    """
    Partitioned Rice Coder.
    """

    @property
    def type_(self) -> str:
        """
        Get the type of the entropy coder.

        Returns
        -------
        str
            The type of the entropy coder.
        """
        return "partitioned_rice"

    def __init__(
        self,
        partition_size_factor: int = PARTITION_SIZE_DEFAULT,
    ):
        """
        Initialize the Partitioned Rice Coder.

        Parameters
        ----------
        partition_size_factor : int, default = PARTITION_SIZE_DEFAULT
            The partition size factor to use.
        """
        self.partition_size_factor = partition_size_factor
        self.partition_size = convert_partition_size_factor_to_partition_size(partition_size_factor = self.partition_size_factor) # in calling this, validates partition size factor
    
    def encode(
        self,
        nums: np.ndarray,
    ) -> bytes:
        """
        Encode the data.

        Parameters
        ----------
        nums : np.ndarray
            The data to encode.

        Returns
        -------
        bytes
            The encoded data.
        """
        return encode(nums = nums, partition_size = self.partition_size)

    def decode(
        self,
        stream: bytes,
        num_samples: int,
    ) -> np.ndarray:
        """
        Decode the data.

        Parameters
        ----------
        stream : bytes
            The encoded data to decode.
        num_samples : int
            The number of samples to decode.

        Returns
        -------
        np.ndarray
            The decoded data.
        """
        return decode(stream = stream, num_samples = num_samples, partition_size = self.partition_size)

##################################################