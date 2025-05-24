# README
# Phillip Long
# May 11, 2025

# Implementation of Free Lossless Audio Codec (FLAC) for use as a baseline.
# See https://xiph.org/flac/documentation_format_overview.html for more.

# IMPORTS
##################################################

import numpy as np
from typing import List, Tuple, Callable, Union
from math import ceil
import librosa
import scipy

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

import utils
import rice

##################################################


# CONSTANTS
##################################################

# number of samples in a block
BLOCK_SIZE = 4096 # see https://xiph.org/flac/documentation_format_overview.html#:~:text=flac%20defaults%20to%20a%20block%20size%20of%204096

# use interchannel decorrelation
INTERCHANNEL_DECORRELATE = True
INTERCHANNEL_DECORRELATE_DTYPE = np.int64 # using interchannel decorrelation can cause bugs with overflow, so we must use the proper data type

# linear predictive coding
LPC_ORDER = 9 # order (see https://xiph.org/flac/documentation_format_overview.html#:~:text=Also%2C%20at%20some%20point%20(usually%20around%20order%209))
LPC_DTYPE = np.int16 # data type of linear prediction coefficients

##################################################


# ENCODE
##################################################

def encode_block(
        block: np.array, # block of integers of shape (n_samples_in_block,)
        order: int = LPC_ORDER, # order for linear predictive coding
    ) -> Tuple[np.array, bytes, int]: # returns tuple of compressed material, rice encoded residuals, and the number of samples in the block
    """FLAC encoder helper function that encodes blocks."""

    # convert block to float
    block_float = block.astype(np.float32)

    # fit linear prediction coefficients, then quantize
    linear_prediction_coefficients = librosa.lpc(y = block_float, order = order)
    linear_prediction_coefficients = np.round(linear_prediction_coefficients).astype(LPC_DTYPE)
    
    # autoregressive prediction using linear prediction coefficients
    approximate_block = scipy.signal.lfilter(b = np.concatenate(([0], -linear_prediction_coefficients), axis = 0, dtype = LPC_DTYPE), a = [1], x = block_float)
    approximate_block = np.round(approximate_block).astype(block.dtype) # ensure approximate waveform is integer values
    
    # compute residual and encode with rice coding
    residuals = block - approximate_block
    residuals_rice = rice.encode(nums = residuals) # rice encoding
    
    # return compressed materials, rice encoded residuals, and number of samples in block
    return linear_prediction_coefficients, residuals_rice, len(block)

def encode(
        waveform: np.array, # waveform of integers of shape (n_samples, n_channels) (if multichannel) or (n_samples,) (if mono)
        block_size: int = BLOCK_SIZE, # block size
        interchannel_decorrelate: bool = INTERCHANNEL_DECORRELATE, # use interchannel decorrelation
        order: int = LPC_ORDER, # order for linear predictive coding
    ) -> Tuple[List[Union[Tuple[np.array, bytes, int], List[Tuple[np.array, bytes, int]]]], type]: # returns tuple of blocks and data type of original data
    """Naive FLAC encoder."""

    # ensure waveform is correct type
    waveform_dtype = waveform.dtype
    assert waveform_dtype in (np.int16, np.int32)

    # deal with different size inputs
    is_mono = waveform.ndim == 1
    if is_mono: # if mono
        waveform = np.expand_dims(a = waveform, axis = -1) # add channel to represent single channel
    elif interchannel_decorrelate and waveform.ndim == 2 and waveform.shape[-1] == 2: # if stereo, perform inter-channel decorrelation (https://xiph.org/flac/documentation_format_overview.html#:~:text=smaller%20frame%20header.-,INTER%2DCHANNEL%20DECORRELATION,-In%20the%20case)
        left, right = waveform.T.astype(INTERCHANNEL_DECORRELATE_DTYPE) # extract left and right channels, cast as int64 so there are no overflow bugs
        center = (left + right) >> 1 # center channel
        side = left - right # side channel
        waveform = np.stack(arrays = (center, side), axis = -1)
        del left, right, center, side
    
    # go through blocks and encode them each
    n_samples, n_channels = waveform.shape
    n_blocks = ceil(n_samples / block_size)
    blocks = [[None] * n_blocks for _ in range(n_channels)]
    for channel_index in range(n_channels):
        for i in range(n_blocks):
            start_index = i * block_size
            end_index = (start_index + block_size) if (i < (n_blocks - 1)) else n_samples
            blocks[channel_index][i] = encode_block(block = waveform[start_index:end_index, channel_index], order = order)

    # don't have multiple channels if mono
    if is_mono:
        blocks = blocks[0]
    
    # return blocks and waveform data type
    return blocks, waveform_dtype

##################################################


# DECODE
##################################################

def decode_block(
        block: np.array, # block tuple with elements (bottleneck, residuals_rice, n_samples_in_block)
    ) -> np.array:
    """FLAC decoder helper function that decodes blocks."""

    # split block
    linear_prediction_coefficients, residuals_rice, n_samples_in_block = block # lpc_order = len(linear_prediction_coefficients) - 1

    # get residuals
    residuals = rice.decode(stream = residuals_rice, n = n_samples_in_block)
    residuals = residuals.astype(np.float32) # ensure residuals are correct data type for waveform reconstruction

    # reconstruct the waveform for the block
    block = scipy.signal.lfilter(b = [1], a = np.concatenate(([1], linear_prediction_coefficients), axis = 0), x = residuals)

    # quantize reconstructed block
    block = np.round(block)

    # return reconstructed waveform for block
    return block

def decode(
        bottleneck: Tuple[List[Union[Tuple[np.array, bytes, int], List[Tuple[np.array, bytes, int]]]], type], # tuple of blocks and the datatype of the original waveform
        interchannel_decorrelate: bool = INTERCHANNEL_DECORRELATE, # was interchannel decorrelation used
    ) -> np.array: # returns the reconstructed waveform of shape (n_samples, n_channels) (if multichannel) or (n_samples,) (if mono)
    """Naive FLAC decoder."""

    # split blocks
    blocks, waveform_dtype = bottleneck

    # determine if mono
    is_mono = type(blocks[0]) is not list
    if is_mono:
        blocks = [blocks] # add multiple channels if mono
    
    # go through blocks
    n_channels, n_blocks = len(blocks), len(blocks[0])
    waveform = [[None] * n_blocks for _ in range(n_channels)]
    for channel_index in range(n_channels):
        for i in range(n_blocks):
            waveform[channel_index][i] = decode_block(block = blocks[channel_index][i]).astype(INTERCHANNEL_DECORRELATE_DTYPE if interchannel_decorrelate else waveform_dtype)

    # reconstruct final waveform
    waveform = [np.concatenate(channel, axis = 0) for channel in waveform]
    waveform = np.stack(arrays = waveform, axis = -1)

    # don't have multiple channels if mono
    if is_mono: # if mono, ensure waveform is one dimension
        waveform = waveform[:, 0]
    elif interchannel_decorrelate and n_channels == 2: # if stereo, perform inter-channel decorrelation
        center, side = waveform.T.astype(INTERCHANNEL_DECORRELATE_DTYPE) # extract center and side channels, cast as int64 so there are no overflow bugs
        left = center + ((side + 1) >> 1) # left channel
        right = center - (side >> 1) # right channel
        waveform = np.stack(arrays = (left, right), axis = -1)
        del center, side, left, right

    # return final reconstructed waveform
    waveform = waveform.astype(waveform_dtype) # ensure correct data type
    return waveform

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":
    
    pass

##################################################