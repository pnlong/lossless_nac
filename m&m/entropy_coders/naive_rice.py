# README
# Phillip Long
# July 6, 2025

# Naive Rice Coder.

# IMPORTS
##################################################

import numpy as np

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(dirname(realpath(__file__)))))

from entropy_coders import EntropyCoder, int_to_pos, inverse_int_to_pos
import utils

##################################################


# CONSTANTS
##################################################

K_DEFAULT = 1 # default rice parameter

##################################################


# NAIVE RICE ENTROPY CODING FUNCTIONS
##################################################

def encode(nums: np.array, k: int = K_DEFAULT, is_nums_signed: bool = False) -> bytes:
    """
    Encode the data.

    Parameters
    ----------
    nums : np.array
        The data to encode.
    k : int, default = K_DEFAULT
        The Rice parameter k, defaults to K_DEFAULT.
    is_nums_signed : bool, default = False
        Whether the numbers being encoded are signed, defaults to False.

    Returns
    -------
    bytes
        The encoded data.
    """

    # helper for writing bits and bytes to an output stream
    out = utils.BitOutputStream()

    # ensure nums is a numpy array
    if is_nums_signed:
        nums = np.array(list(map(int_to_pos, nums))) # convert from potentially negative number to non-negative
    else:
        nums = np.array(nums)

    # iterate through numbers
    for x in nums:

        # compute quotient and remainder
        q = x >> k # quotient = n // 2^k
        r = x & ((1 << k) - 1) # remainder = n % 2^k

        # encode the quotient with unary coding (q ones followed by a zero)
        for _ in range(q):
            out.write_bit(bit = True)
        out.write_bit(bit = False)

        # encode the remainder with binary coding using k bits
        out.write_bits(bits = r, n = k)

    # get bytes stream and return
    stream = out.flush()
    return stream

def decode(stream: bytes, num_samples: int, k: int = K_DEFAULT, is_nums_signed: bool = False) -> np.array:
    """
    Decode the data.

    Parameters
    ----------
    stream : bytes
        The encoded data to decode.
    num_samples : int
        The number of samples to decode.
    k : int, default = K_DEFAULT
        The Rice parameter k, defaults to K_DEFAULT.
    is_nums_signed : bool, default = False
        Whether the numbers being decoded are signed, defaults to False.

    Returns
    -------
    np.array
        The decoded data.
    """
    
    # helper for reading bits and bytes from an input stream
    inp = utils.BitInputStream(stream = stream)

    # initialize numbers list
    nums = utils.rep(x = 0, times = num_samples)

    # read in numbers
    for i in range(num_samples):
        # read unary-coded quotient
        q = 0
        while inp.read_bit() == True:
            q += 1

        # read k-bit remainder
        r = inp.read_bits(n = k)

        # reconstruct original number
        x = (q << k) | r
        nums[i] = x

    # convert results to numpy array
    if is_nums_signed: # convert back to signed numbers
        nums = list(map(inverse_int_to_pos, nums))
    nums = np.array(nums)

    # return list of numbers
    return nums

##################################################


# ENTROPY CODER INTERFACE
##################################################

class NaiveRiceCoder(EntropyCoder):
    """
    Naive Rice Coder.
    """

    def __init__(self, k: int = K_DEFAULT, is_nums_signed: bool = False):
        """
        Initialize the Naive Rice Coder.

        Parameters
        ----------
        k : int, default = K_DEFAULT
            The Rice parameter k, defaults to K_DEFAULT.
        is_nums_signed : bool, default = False
            Whether the numbers being encoded are signed, defaults to False.
        """
        self.k = k
        assert self.k > 0, "Rice parameter k must be positive."
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
            k = self.k,
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
            k = self.k,
            is_nums_signed = self.is_nums_signed,
        )
        
##################################################