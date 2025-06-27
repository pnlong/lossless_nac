# README
# Phillip Long
# May 11, 2025

# Implementation of Rice coding.
# Note that Rice codes are a subset of Golomb codes (https://en.wikipedia.org/wiki/Golomb_coding).
# Whereas a Golomb code has a tunable parameter `M` that can be any positive integer value, 
# Rice codes are those in which the tunable parameter is a power of two (`M = 2 ** K`).
# Furthermore, Rice codes can only handle non-negative integers. To combat this shortcoming,
# we first apply a function to any input list of numbers that maps all integers onto positive integers.

# IMPORTS
##################################################

import numpy as np
from typing import Union, List

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))

import utils

##################################################


# CONSTANTS
##################################################

# default rice parameter
K = 12 # Golomb coding equivalent M is calculated as M = 2 ** K, so probably want to use a small value for K

# golden ratio for determining optimal rice parameter
PHI = (1 + np.sqrt(5)) / 2

# for reconstructing arrays in case where k = 0
BYTES_PER_ELEMENT_TO_DTYPE = dict(zip((1, 2, 4, 8), (np.int8, np.int16, np.int32, np.int64)))

##################################################


# MAP ALL INTEGERS ONTO NON-NEGATIVE INTEGERS
##################################################

def int_to_pos(x: int) -> int:
    """Maps any integer onto a non-negative integer."""
    if x >= 0: # if x is non-negative
        return 2 * x # map positive values onto even numbers
    else: # if x < 0 (x is negative)
        return (-2 * x) - 1 # map negative values onto odd numbers

def inverse_int_to_pos(x: int) -> int:
    """Inverse to the previous function (see `int_to_pos(x)`)."""
    if x % 2 == 0: # if x is an even number
        return x // 2 # then x must be non-negative
    else: # if x is an odd number
        return (x + 1) // -2 # then x must be negative

##################################################


# DETERMINE OPTIMAL K
##################################################

# given a list of numbers, determines the optimal rice parameter
def get_optimal_k(nums: Union[List[int], np.array]) -> int:
    """
    Given a list of integers, return the optimal Rice parameter `k` that would yield the best compression.
    Uses formula described in section III, part A, equation 8 (page 6) of https://tda.jpl.nasa.gov/progress_report/42-159/159E.pdf.
    """

    # convert to numpy array of zigzagged (to ensure only positive numbers)
    nums = np.array(list(map(int_to_pos, nums)))

    # get mean of nums
    mu = np.mean(nums)

    # get optimal rice parameter
    if mu < PHI:
        return 0
    else:
        return 1 + int(np.log2(np.log(PHI - 1) / np.log(mu / (mu + 1))))

##################################################


# ENCODE
##################################################

def encode(nums: Union[List[int], np.array], k: int = K) -> bytes:
    """
    Encode a list of integers (can be negative or positive) using Rice coding.
    """
        
    # helper for writing bits and bytes to an output stream
    out = utils.BitOutputStream()

    # special case where k = 0, then don't use rice coding
    if k == 0:

        # ensure nums is a numpy array
        nums = np.array(nums)

        # write header, which contains the number of bytes per element as a single byte
        bytes_per_element = nums.itemsize
        out.write_byte(byte = bytes_per_element)
        bits_per_element = 8 * bytes_per_element

        # iterate through nums
        for x in nums:

            # convert from potentially negative number to non-negative
            x = int_to_pos(x = x)

            # write each element in the correct number of bits
            out.write_bits(bits = x, n = bits_per_element)

    # otherwise, do normal rice coding
    else:

        # iterate through numbers
        for x in nums:

            # convert from potentially negative number to non-negative
            x = int_to_pos(x = x)

            # compute quotient and remainder
            q = x >> k # quotient = n // 2^k
            r = x & ((1 << k) - 1) # remainder = n % 2^k

            # encode the quotient with unary coding (q ones followed by a zero)
            # out.write_bits(bits = (((1 << q) - 1) << 1), n = q + 1)
            for _ in range(q):
                out.write_bit(bit = True)
            out.write_bit(bit = False)

            # encode the remainder with binary coding using k bits, since the remainder will range from 0 to 2^k - 1, which can be encoded in k bits
            out.write_bits(bits = r, n = k)

    # get bytes stream and return
    stream = out.flush()
    return stream

##################################################


# DECODE
##################################################

def decode(stream: bytes, n: int, k: int = K) -> np.array:
    """
    Decode an input stream using Rice coding (terminates after `n` numbers have been read).
    """

    # helper for reading bits and bytes from an input stream
    inp = utils.BitInputStream(stream = stream)  

    # initialize numbers list
    nums = utils.rep(x = 0, times = n)

    # special case where k = 0, then don't use rice coding
    if k == 0:

        # read in first byte, which is number of bytes per element
        bytes_per_element = inp.read_byte()
        bits_per_element = bytes_per_element * 8

        # read in numbers
        for i in range(n):

            # reconstruct original number
            x = inp.read_bits(n = bits_per_element)
            x = inverse_int_to_pos(x = x)
            nums[i] = x

        # convert results to numpy array
        nums = np.array(nums, dtype = BYTES_PER_ELEMENT_TO_DTYPE[bytes_per_element])
    
    # otherwise, do normal rice coding
    else:

        # read in numbers
        for i in range(n):

            # read unary-coded quotient
            q = 0
            while inp.read_bit() == True:
                q += 1
            
            # read k-bit remainder
            r = inp.read_bits(n = k)

            # reconstruct original number
            x = (q << k) | r
            x = inverse_int_to_pos(x = x)
            nums[i] = x

        # convert results to numpy array
        nums = np.array(nums)

    # return list of numbers
    return nums

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":
    
    pass

##################################################