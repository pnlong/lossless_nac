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
import warnings

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))

import utils

##################################################


# CONSTANTS
##################################################

K = 12 # Golomb coding equivalent M is calculated as M = 2 ** K, so probably want to use a small value for K

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


# ENCODE
##################################################

# def encode(out: BitOutputStream, nums: Union[List[int], np.array], k: int = K):
#     """
#     Encode a list of integers (can be negative or positive) using Rice coding.
#     """

#     # iterate through numbers
#     for x in nums:

#         # convert from potentially negative number to non-negative
#         x = int_to_pos(x = x)

#         # compute quotient and remainder
#         q = x >> k # quotient = n // 2^k
#         r = x & ((1 << k) - 1) # remainder = n % 2^k

#         # encode the quotient with unary coding (q ones followed by a zero)
#         for _ in range(q):
#             out.write_bit(bit = True)
#         out.write_bit(bit = False)

#         # encode the remainder with binary coding using k bits, since the remainder will range from 0 to 2^k - 1, which can be encoded in k bits
#         out.write_bits(bits = r, n = k)

#     # flush output stream
#     out.flush()

def encode(nums: Union[List[int], np.array], k: int = K) -> bytes:
    """
    Encode a list of integers (can be negative or positive) using Rice coding.
    """

    # helper for writing bits and bytes to an output stream
    out = utils.BitOutputStream()

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

# def decode(inp: BitInputStream, n: int = None, k: int = K) -> np.array:
#     """
#     Decode an input stream using Rice coding (terminates at end of file or after `n` numbers have been read).
#     """

#     # initialize numbers list
#     nums = []

#     # read in numbers
#     i = 0
#     while n is None or i < n:

#         # try to read bits
#         try:

#             # read unary-coded quotient
#             q = 0
#             while inp.read_bit() == True:
#                 q += 1
            
#             # read k-bit remainder
#             r = inp.read_bits(n = k)

#             # reconstruct original number
#             x = (q << k) | r
#             x = inverse_int_to_pos(x = x)
#             nums.append(x)

#             # increment i
#             i += 1

#         # break out of while loop when the end of the file is reached
#         except EOFError:
#             break

#     # convert results to numpy array and return
#     nums = np.array(nums)
#     return nums

def decode(stream: bytes, n: int = None, k: int = K) -> np.array:
    """
    Decode an input stream using Rice coding (terminates at end of stream or after `n` numbers have been read).
    """

    # initialize numbers list
    nums = []

    # helper for reading bits and bytes from an input stream
    inp = utils.BitInputStream(stream = stream)    
    
    # read in numbers
    i = 0
    while n is None or i < n:

        # try to read bits
        try:

            # read unary-coded quotient
            q = 0
            while inp.read_bit() == True:
                q += 1
            
            # read k-bit remainder
            r = inp.read_bits(n = k)

            # reconstruct original number
            x = (q << k) | r
            x = inverse_int_to_pos(x = x)
            nums.append(x)

            # increment i
            i += 1

        # break out of while loop when the end of the stream is reached
        except RuntimeError:
            break

    # convert results to numpy array and return
    nums = np.array(nums)
    return nums

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":
    
    pass

##################################################