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
sys.path.insert(0, dirname(dirname(realpath(__file__))))

from entropy_coder import EntropyCoder
import bitstream

##################################################


# CONSTANTS
##################################################



##################################################


# HELPER FUNCTIONS
##################################################

# get numpy data type from bytes per element
def get_dtype_from_bytes_per_element(
    bytes_per_element: int,
) -> np.dtype:
    """
    Get the numpy dtype from the bytes per element.

    Parameters
    ----------
    bytes_per_element : int
        The number of bytes per element.

    Returns
    -------
    np.dtype
        The numpy dtype.
    """
    return np.dtype(f"int{bytes_per_element * 8}")

##################################################


# VERBATIM ENTROPY CODING FUNCTIONS
##################################################

def encode(
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

    # helper for writing bits and bytes to an output stream
    out = bitstream.BitOutputStream()

    # ensure nums is a numpy array
    nums = np.array(nums)

    # write header, which contains the number of bytes per element as a single byte
    bytes_per_element = nums.itemsize
    out.write_byte(byte = bytes_per_element)
    bits_per_element = 8 * bytes_per_element

    # iterate through nums
    for x in nums:
        out.write_bits(bits = x, n = bits_per_element) # write each element in the correct number of bits

    # get bytes stream
    stream = out.flush()

    return stream

def decode(
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

    # helper for reading bits and bytes from an input stream
    inp = bitstream.BitInputStream(stream = stream)

    # initialize numbers list
    nums = np.zeros(shape = num_samples)

    # read in first byte, which is number of bytes per element
    bytes_per_element = inp.read_byte()
    bits_per_element = bytes_per_element * 8

    # read in numbers
    for i in range(num_samples):
        x = inp.read_bits(n = bits_per_element)
        if x >= (1 << (bits_per_element - 1)): # convert unsigned to signed using two's complement if the sign bit is set
            x = x - (1 << bits_per_element) # convert to negative using two's complement
        nums[i] = x

    # convert results to numpy array with correct dtype (assuming signed integers)
    target_dtype = get_dtype_from_bytes_per_element(bytes_per_element = bytes_per_element)
    nums = np.array(nums, dtype = target_dtype)

    return nums

##################################################


# ENTROPY CODER INTERFACE
##################################################

class VerbatimCoder(EntropyCoder):
    """
    Verbatim Entropy Coder.
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
        return "verbatim"

    def __init__(
        self,
    ):
        """
        Initialize the Verbatim Coder.
        """
        pass
    
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
        return encode(nums = nums)

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
        return decode(stream = stream, num_samples = num_samples)
        
##################################################