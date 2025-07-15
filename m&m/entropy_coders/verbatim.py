# README
# Phillip Long
# July 6, 2025

# Verbatim Entropy Coder.

# IMPORTS
##################################################

import numpy as np

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(dirname(realpath(__file__)))))

from entropy_coders import EntropyCoder, int_to_pos, inverse_int_to_pos, get_dtype_from_bytes_per_element
import utils

##################################################


# CONSTANTS
##################################################



##################################################



# VERBATIM ENTROPY CODING FUNCTIONS
##################################################

def encode(nums: np.array) -> bytes:
    """
    Encode the data.

    Parameters
    ----------
    nums : np.array
        The data to encode.

    Returns
    -------
    bytes
        The encoded data.
    """

    # helper for writing bits and bytes to an output stream
    out = utils.BitOutputStream()

    # ensure nums is a numpy array
    nums = np.array(nums)

    # write header, which contains the number of bytes per element as a single byte
    bytes_per_element = nums.itemsize
    out.write_byte(byte = bytes_per_element)
    bits_per_element = 8 * bytes_per_element

    # iterate through nums
    for x in nums:
        out.write_bits(bits = x, n = bits_per_element) # write each element in the correct number of bits

    # get bytes stream and return
    stream = out.flush()
    return stream

def decode(stream: bytes, num_samples: int) -> np.array:
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
    np.array
        The decoded data.
    """

    # helper for reading bits and bytes from an input stream
    inp = utils.BitInputStream(stream = stream)

    # initialize numbers list
    nums = utils.rep(x = 0, times = num_samples)

    # read in first byte, which is number of bytes per element
    bytes_per_element = inp.read_byte()
    bits_per_element = bytes_per_element * 8

    # read in numbers
    for i in range(num_samples):
        x = inp.read_bits(n = bits_per_element)
        nums[i] = x

    # convert results to numpy array with correct dtype
    nums = np.array(nums, dtype = get_dtype_from_bytes_per_element(bytes_per_element = bytes_per_element))

    # return list of numbers
    return nums

##################################################


# ENTROPY CODER INTERFACE
##################################################

class VerbatimCoder(EntropyCoder):
    """
    Verbatim Entropy Coder. Stores numbers directly in their binary representation.
    """

    def __init__(self):
        """
        Initialize the Verbatim Coder.
        """
        pass
    
    def encode(self, nums: np.array) -> bytes:
        """
        Encode the data.

        Parameters
        ----------
        nums : np.array
            The data to encode.

        Returns
        -------
        bytes
            The encoded data.
        """
        return encode(nums = nums)

    def decode(self, stream: bytes, num_samples: int) -> np.array:
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
        np.array
            The decoded data.
        """
        return decode(stream = stream, num_samples = num_samples)
        
##################################################