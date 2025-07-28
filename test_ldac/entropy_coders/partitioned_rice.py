# README
# Phillip Long
# July 26, 2025

# Partitioned Rice Coder.

# IMPORTS
##################################################

import numpy as np
from math import log2, ceil

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

from entropy_coder import EntropyCoder
import naive_rice
import adaptive_rice
import bitstream

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
    out: bitstream.BitOutputStream,
    nums: np.ndarray,
    partition_size: int,
) -> None:
    """
    Encode the data.

    Parameters
    ----------
    out : bitstream.BitOutputStream
        The output stream to write to.
    nums : np.ndarray
        The data to encode.
    partition_size : int
        The partition size to use.
    """

    # write partitions
    for start_index in range(0, len(nums), partition_size):

        # get partition
        end_index = min(start_index + partition_size, len(nums))
        partition = nums[start_index:end_index]

        # write optimally-rice-coded partition
        adaptive_rice.encode(out = out, nums = partition)

    return
    
def decode(
    inp: bitstream.BitInputStream,
    num_samples: int,
    partition_size: int,
) -> np.ndarray:
    """
    Decode the data.

    Parameters
    ----------
    inp : bitstream.BitInputStream
        The input stream to read from.
    num_samples : int
        The number of samples to decode.
    partition_size : int
        The partition size to use.

    Returns
    -------
    np.ndarray
        The decoded data. 
    """

    # initialize numbers list
    nums = [None] * ceil(num_samples / partition_size)
    
    # decode partitions
    for i, start_index in enumerate(range(0, num_samples, partition_size)):

        # get partition info
        end_index = min(start_index + partition_size, num_samples)
        partition_length = end_index - start_index

        # get results
        nums[i] = adaptive_rice.decode(inp = inp, num_samples = partition_length)

    # concatenate partitions
    nums = np.concatenate(nums, axis = 0)

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

        # encode data with output stream
        out = bitstream.BitOutputStream() # helper for writing bits and bytes to an output stream
        encode(out = out, nums = nums, partition_size = self.partition_size) # encode data
        stream = out.flush() # get bytes stream

        return stream

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

        # decode data with input stream
        inp = bitstream.BitInputStream(stream = stream) # helper for reading bits and bytes from an input stream
        nums = decode(inp = inp, num_samples = num_samples, partition_size = self.partition_size) # decode data

        return nums

##################################################