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

from entropy_coder import EntropyCoder
import verbatim
import naive_rice

##################################################


# CONSTANTS
##################################################

PHI = (1 + np.sqrt(5)) / 2 # golden ratio

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
    stream: bytes, num_samples: int,
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

        Parameters
        ----------
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