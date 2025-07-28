# README
# Phillip Long
# July 6, 2025

# Adaptive Rice Coder.

# IMPORTS
##################################################

import numpy as np

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

from entropy_coder import EntropyCoder
import verbatim
import naive_rice
import bitstream

##################################################


# CONSTANTS
##################################################

PHI = (1 + np.sqrt(5)) / 2 # golden ratio
RICE_PARAMETER_BITS = 5 # bits to represent rice parameter

##################################################


# HELPER FUNCTIONS
##################################################

# get optimal rice parameter
def get_optimal_rice_parameter(
    nums: np.ndarray,
) -> int:
    """
    Get the optimal rice parameter for the data.
    Uses formula described in section III, part A, equation 8 (page 6) of https://tda.jpl.nasa.gov/progress_report/42-159/159E.pdf.

    Parameters
    ----------
    nums : np.ndarray
        The data to get the optimal rice parameter for.

    Returns
    -------
    int
        The optimal rice parameter.
    """

    # get mean of data
    mu = np.mean(np.array(list(map(naive_rice.zigzag_encode, nums))))

    # determine optimal rice parameter
    if mu < PHI:
        k = 0
    else: # uses formula described in section III, part A, equation 8 (page 6) of https://tda.jpl.nasa.gov/progress_report/42-159/159E.pdf
        k = 1 + int(np.log2(np.log(PHI - 1) / np.log(mu / (mu + 1))))

    return k

##################################################


# ADAPTIVE RICE ENTROPY CODING FUNCTIONS
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
    
    # determine optimal rice parameter and write to stream
    k = get_optimal_rice_parameter(nums = nums)
    out.write_bits(bits = k, n = RICE_PARAMETER_BITS) # write rice parameter

    # encode data
    if k == 0:
        verbatim.encode(out = out, nums = nums)
    else:
        naive_rice.encode(out = out, nums = nums, k = k)

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

    # get rice parameter
    k = inp.read_bits(n = RICE_PARAMETER_BITS)

    # decode data and return
    if k == 0:
        nums = verbatim.decode(inp = inp, num_samples = num_samples)
    else:
        nums = naive_rice.decode(inp = inp, num_samples = num_samples, k = k)

    return nums

##################################################


# ENTROPY CODER INTERFACE
##################################################

class AdaptiveRiceCoder(EntropyCoder):
    """
    Adaptive Rice Coder.
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
        return "adaptive_rice"

    def __init__(
        self,
    ):
        """
        Initialize the Adaptive Rice Coder.
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