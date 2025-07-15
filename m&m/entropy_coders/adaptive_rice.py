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
sys.path.insert(0, dirname(dirname(dirname(realpath(__file__)))))

from entropy_coders import EntropyCoder, int_to_pos
import verbatim
import naive_rice
import utils

##################################################


# CONSTANTS
##################################################

PHI = (1 + np.sqrt(5)) / 2 # golden ratio

##################################################


# ADAPTIVE RICE ENTROPY CODING FUNCTIONS
##################################################

def encode(nums: np.array, is_nums_signed: bool = False) -> bytes:
    """
    Encode the data.

    Parameters
    ----------
    nums : np.array
        The data to encode.
    is_nums_signed : bool, default = False
        Whether the numbers being encoded are signed, defaults to False.

    Returns
    -------
    bytes
        The encoded data.
    """
    
    # determine optimal rice parameter
    mu = np.mean(np.array(list(map(int_to_pos, nums)) if is_nums_signed else nums))
    if mu < PHI:
        k = 0
    else: # uses formula described in section III, part A, equation 8 (page 6) of https://tda.jpl.nasa.gov/progress_report/42-159/159E.pdf
        k = 1 + int(np.log2(np.log(PHI - 1) / np.log(mu / (mu + 1))))

    # encode data and return stream
    if k == 0:
        stream = verbatim.encode(nums = nums)
    else:
        stream = naive_rice.encode(nums = nums, k = k, is_nums_signed = is_nums_signed)

    # add rice parameter to stream
    stream = bytes([k]) + stream # prepend rice parameter to stream

    # return the final stream
    return stream

def decode(stream: bytes, num_samples: int, is_nums_signed: bool = False) -> np.array:
    """
    Decode the data.

    Parameters
    ----------
    stream : bytes
        The encoded data to decode.
    num_samples : int
        The number of samples to decode.
    is_nums_signed : bool, default = False
        Whether the numbers being decoded are signed, defaults to False.

    Returns
    -------
    np.array
        The decoded data.
    """
    
    # get rice parameter
    k = stream[0]

    # decode data and return
    if k == 0:
        nums = verbatim.decode(stream = stream[1:], num_samples = num_samples)
    else:
        nums = naive_rice.decode(stream = stream[1:], num_samples = num_samples, k = k, is_nums_signed = is_nums_signed)

    # return the final list of numbers
    return nums

##################################################


# ENTROPY CODER INTERFACE
##################################################

class AdaptiveRiceCoder(EntropyCoder):
    """
    Adaptive Rice Coder.
    """

    def __init__(self, is_nums_signed: bool = False):
        """
        Initialize the Adaptive Rice Coder.

        Parameters
        ----------
        is_nums_signed : bool, default = False
            Whether the numbers being encoded are signed, defaults to False.
        """
        self.is_nums_signed = is_nums_signed
    
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
        return encode(
            nums = nums,
            is_nums_signed = self.is_nums_signed,
        )

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
        return decode(
            stream = stream,
            num_samples = num_samples,
            is_nums_signed = self.is_nums_signed,
        )

##################################################