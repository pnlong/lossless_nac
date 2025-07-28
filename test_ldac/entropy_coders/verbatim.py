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

BYTES_PER_ELEMENT_BITS = 4

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
    out: bitstream.BitOutputStream,
    nums: np.ndarray,
) -> None:
    """
    Encode the data.

    Parameters
    ----------
    out : bitstream.BitOutputStream
        The output stream to write to.
    nums : np.ndarray
        The data to encode.
    """

    # ensure nums is a numpy array
    nums = np.array(nums)

    # write header, which contains the number of bytes per element as a single byte
    bytes_per_element = nums.itemsize
    out.write_bits(bits = bytes_per_element, n = BYTES_PER_ELEMENT_BITS) # write number of bytes per element

    # write nums
    out.write_bytes(bytes_ = nums.tobytes())

    return

def decode(
    inp: bitstream.BitInputStream,
    num_samples: int,
) -> np.ndarray:
    """
    Decode the data.

    Parameters
    ----------
    inp : bitstream.BitInputStream
        The input stream to read from.
    num_samples : int
        The number of samples to decode.

    Returns
    -------
    np.ndarray
        The decoded data.
    """

    # initialize numbers list
    nums = np.zeros(shape = num_samples)

    # read in first byte, which is number of bytes per element
    bytes_per_element = inp.read_bits(n = BYTES_PER_ELEMENT_BITS)

    # read nums
    buffer = inp.read_bytes(n = num_samples * bytes_per_element)
    target_dtype = get_dtype_from_bytes_per_element(bytes_per_element = bytes_per_element)
    nums = np.frombuffer(buffer = buffer, dtype = target_dtype)

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

        # encode data with output stream
        out = bitstream.BitOutputStream() # helper for writing bits and bytes to an output stream
        encode(out = out, nums = nums) # encode data
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
        nums = decode(inp = inp, num_samples = num_samples) # decode data

        return nums
        
##################################################