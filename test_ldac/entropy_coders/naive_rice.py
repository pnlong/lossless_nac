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
sys.path.insert(0, dirname(dirname(realpath(__file__))))

from entropy_coder import EntropyCoder
import bitstream

##################################################


# CONSTANTS
##################################################

K_DEFAULT = 1 # default rice parameter

##################################################


# HELPER FUNCTIONS
##################################################

# zigzag encode
def zigzag_encode(
    x: int,
) -> int:
    """
    Maps any signed integer onto an unsigned integer.

    Parameters
    ----------
    x : int
        The signed integer to encode.

    Returns
    -------
    int
        The unsigned integer.
    """
    if x >= 0: # if x is non-negative
        return 2 * x # map positive values onto even numbers
    else: # if x < 0 (x is negative)
        return (-2 * x) - 1 # map negative values onto odd numbers

# zigzag decode
def zigzag_decode(
    x: int,
) -> int:
    """
    Inverse to the previous function.

    Parameters
    ----------
    x : int
        The unsigned integer to decode.

    Returns
    -------
    int
        The signed integer.
    """
    if x % 2 == 0: # if x is an even number
        return x // 2 # then x must be non-negative
    else: # if x is an odd number
        return (x + 1) // -2 # then x must be negative

##################################################


# NAIVE RICE ENTROPY CODING FUNCTIONS
##################################################

def encode(
    nums: np.ndarray,
    k: int = K_DEFAULT,
) -> bytes:
    """
    Encode the data.

    Parameters
    ----------
    nums : np.ndarray
        The data to encode.
    k : int, default = K_DEFAULT
        The Rice parameter k, defaults to K_DEFAULT.

    Returns
    -------
    bytes
        The encoded data.
    """

    # helper for writing bits and bytes to an output stream
    out = bitstream.BitOutputStream()

    # ensure nums is a numpy array
    nums = np.array(list(map(zigzag_encode, nums))) # convert from potentially negative number to non-negative

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

    # get bytes stream
    stream = out.flush()

    return stream

def decode(
    stream: bytes,
    num_samples: int,
    k: int = K_DEFAULT,
) -> np.ndarray:
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

    Returns
    -------
    np.ndarray
        The decoded data.
    """
    
    # helper for reading bits and bytes from an input stream
    inp = bitstream.BitInputStream(stream = stream)

    # initialize numbers list
    nums = np.zeros(shape = num_samples)

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
    nums = np.array(list(map(zigzag_decode, nums))) # convert back to signed numbers

    return nums

##################################################


# ENTROPY CODER INTERFACE
##################################################

class NaiveRiceCoder(EntropyCoder):
    """
    Naive Rice Coder using C helpers for performance.
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
        return "naive_rice"

    def __init__(
        self,
        k: int = K_DEFAULT,
    ):
        """
        Initialize the Naive Rice Coder.

        Parameters
        ----------
        k : int, default = K_DEFAULT
            The Rice parameter k, defaults to K_DEFAULT.
        """
        self.k = k
        assert self.k > 0, "Rice parameter k must be positive."
    
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
        return encode(nums = nums, k = self.k)

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
        return decode(stream = stream, num_samples = num_samples, k = self.k)

##################################################