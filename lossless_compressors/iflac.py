# README
# Phillip Long
# June 21, 2025

# Implementation of Improved Free Lossless Audio Codec (IFLAC), which is closer to what FLAC is actually doing.

# IMPORTS
##################################################

import numpy as np
from typing import List, Tuple, Union
from math import ceil
# import librosa
import scipy
import argparse
import time
import warnings

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

import utils
import rice
import logging_for_zach
from flac import lpc_autocorrelation_method, LPC_DTYPE

##################################################


# CONSTANTS
##################################################

# lpc order
MIN_LPC_ORDER, MAX_LPC_ORDER = 5, 20 # since adaptive LPC order is used, this defines the minimum and maximum possible LPC orders

# golden ratio for optimal rice parameter calculation
PHI = (1 + np.sqrt(5)) / 2

##################################################


# ENCODE
##################################################

def encode_block(
        block: np.array, # block of integers of shape (n_samples_in_block,)
    ) -> Tuple[int, int, np.array, int, bytes]: # returns tuple of number of samples in the block, LPC order, compressed material, rice parameter K, and rice encoded residuals
    """IFLAC encoder helper function that encodes blocks."""

    # convert block to float
    block_float = block.astype(np.float32)

    # determine the most optimal lpc order and rice parameter
    min_cost = float("inf")
    for lpc_order in range(MIN_LPC_ORDER, MAX_LPC_ORDER + 1):

        # fit linear prediction coefficients, then quantize
        # linear_prediction_coefficients = librosa.lpc(y = block_float, order = lpc_order) # does not guarantee numerical stability
        linear_prediction_coefficients = lpc_autocorrelation_method(y = block_float, order = lpc_order)
        linear_prediction_coefficients = np.round(linear_prediction_coefficients).astype(LPC_DTYPE)
        if not np.all(np.abs(np.roots(linear_prediction_coefficients) < 1)): # ensure lpc coefficients are stable
            warnings.warn(message = "Linear prediction coefficients are unstable!", category = RuntimeWarning)
        
        # autoregressive prediction using linear prediction coefficients
        approximate_block = scipy.signal.lfilter(b = np.concatenate(([0], -linear_prediction_coefficients), axis = 0, dtype = linear_prediction_coefficients.dtype), a = [1], x = block_float)
        approximate_block = np.round(approximate_block).astype(block.dtype) # ensure approximate waveform is integer values
        
        # compute residual and encode with rice coding
        residuals = block - approximate_block
        mu = np.mean(np.apply_along_axis(rice.int_to_pos, axis = 0, arr = residuals)) # mean of positive zigzagged residuals
        try:
            k = 1 + int(np.log2(np.log(PHI - 1) / np.log(mu / (mu + 1)))) # determine optimal k from residuals; see section III, part A, equation 8 (page 6) of https://tda.jpl.nasa.gov/progress_report/42-159/159E.pdf
        except (OverflowError): # when mu = 0, there is an overflow error
            k = rice.K
        residuals_rice = rice.encode(nums = residuals, k = k) # rice encoding

        # determine cost
        cost = linear_prediction_coefficients.nbytes + len(residuals_rice) # cost is the size of linear prediction coefficients (which is variable for different orders) plus the size of riced residuals
        if cost < min_cost:
            min_cost = cost
            best_lpc_order = lpc_order
            best_linear_prediction_coefficients = linear_prediction_coefficients.copy()
            best_k = k
            best_residuals_rice = residuals_rice

        # free up memory
        del linear_prediction_coefficients, approximate_block, residuals, mu, k, residuals_rice
    
    # return number of samples in block, LPC order, compressed materials, rice parameter K, and rice encoded residuals
    return len(block), best_lpc_order, best_linear_prediction_coefficients, best_k, best_residuals_rice


def encode(
        waveform: np.array, # waveform of integers of shape (n_samples, n_channels) (if multichannel) or (n_samples,) (if mono)
        block_size: int = utils.BLOCK_SIZE, # block size
        interchannel_decorrelate: bool = utils.INTERCHANNEL_DECORRELATE, # use interchannel decorrelation
        log_for_zach_kwargs: dict = None, # available keyword arguments for log_for_zach() function
    ) -> Tuple[List[Union[Tuple[int, int, np.array, int, bytes], List[Tuple[int, int, np.array, int, bytes]]]], type]: # returns tuple of blocks and data type of original data
    """IFLAC encoder."""

    # ensure waveform is correct type
    waveform_dtype = waveform.dtype
    assert any(waveform_dtype == dtype for dtype in utils.VALID_AUDIO_DTYPES)

    # deal with different size inputs
    is_mono = waveform.ndim == 1
    if is_mono: # if mono
        waveform = np.expand_dims(a = waveform, axis = -1) # add channel to represent single channel
    elif interchannel_decorrelate and waveform.ndim == 2 and waveform.shape[-1] == 2: # if stereo, perform inter-channel decorrelation (https://xiph.org/flac/documentation_format_overview.html#:~:text=smaller%20frame%20header.-,INTER%2DCHANNEL%20DECORRELATION,-In%20the%20case)
        left, right = waveform.T.astype(utils.INTERCHANNEL_DECORRELATE_DTYPE) # extract left and right channels, cast as int64 so there are no overflow bugs
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
            blocks[channel_index][i] = encode_block(block = waveform[start_index:end_index, channel_index])

    # log for zach
    if log_for_zach_kwargs is not None:
        residuals = np.stack([np.concatenate([rice.decode(stream = block[-1], n = block[0], k = block[-2]) for block in channel], axis = 0) for channel in blocks], axis = -1)
        if is_mono:
            residuals = residuals.squeeze(dim = -1)
        residuals_rice = rice.encode(nums = residuals.flatten())
        logging_for_zach.log_for_zach(
            residuals = residuals,
            residuals_rice = residuals_rice,
            **log_for_zach_kwargs)

    # don't have multiple channels if mono
    if is_mono:
        blocks = blocks[0]
    
    # return blocks and waveform data type
    return blocks, waveform_dtype

##################################################


# DECODE
##################################################

def decode_block(
        block: Tuple[int, int, np.array, int, bytes], # block tuple with elements (n_samples_in_block, lpc_order, bottleneck, k, residuals_rice)
    ) -> np.array:
    """IFLAC decoder helper function that decodes blocks."""

    # split block
    n_samples_in_block, lpc_order, linear_prediction_coefficients, k, residuals_rice = block # lpc_order = len(linear_prediction_coefficients) - 1; len(linear_prediction_coefficients) = lpc_order + 1

    # get residuals
    residuals = rice.decode(stream = residuals_rice, n = n_samples_in_block, k = k)
    residuals = residuals.astype(np.float32) # ensure residuals are correct data type for waveform reconstruction

    # reconstruct the waveform for the block
    block = scipy.signal.lfilter(b = [1], a = np.concatenate(([1], linear_prediction_coefficients), axis = 0), x = residuals)

    # quantize reconstructed block
    block = np.round(block)

    # return reconstructed waveform for block
    return block


def decode(
        bottleneck: Tuple[List[Union[Tuple[int, int, np.array, int, bytes], List[Tuple[int, int, np.array, int, bytes]]]], type], # tuple of blocks and the datatype of the original waveform
        interchannel_decorrelate: bool = utils.INTERCHANNEL_DECORRELATE, # was interchannel decorrelation used
    ) -> np.array: # returns the reconstructed waveform of shape (n_samples, n_channels) (if multichannel) or (n_samples,) (if mono)
    """IFLAC decoder."""

    # split bottleneck
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
            waveform[channel_index][i] = decode_block(block = blocks[channel_index][i])
            waveform[channel_index][i] = waveform[channel_index][i].astype(utils.INTERCHANNEL_DECORRELATE_DTYPE if interchannel_decorrelate else waveform_dtype)

    # reconstruct final waveform
    waveform = [np.concatenate(channel, axis = 0) for channel in waveform]
    waveform = np.stack(arrays = waveform, axis = -1)

    # don't have multiple channels if mono
    if is_mono: # if mono, ensure waveform is one dimension
        waveform = waveform[:, 0]
    elif interchannel_decorrelate and n_channels == 2: # if stereo, perform inter-channel decorrelation
        center, side = waveform.T.astype(utils.INTERCHANNEL_DECORRELATE_DTYPE) # extract center and side channels, cast as int64 so there are no overflow bugs
        left = center + ((side + 1) >> 1) # left channel
        right = center - (side >> 1) # right channel
        waveform = np.stack(arrays = (left, right), axis = -1)
        del center, side, left, right

    # return final reconstructed waveform
    waveform = waveform.astype(waveform_dtype) # ensure correct data type
    return waveform

##################################################


# HELPER FUNCTION TO GET THE SIZE IN BYTES OF THE BOTTLENECK
##################################################

def get_bottleneck_size(
    bottleneck: Tuple[List[Union[Tuple[int, int, np.array, int, bytes], List[Tuple[int, int, np.array, int, bytes]]]], type],
) -> int:
    """Returns the size of the given bottleneck in bytes."""

    # tally of the size in bytes of the bottleneck
    size = 0

    # split bottleneck
    blocks, waveform_dtype = bottleneck
    # size += 1 # use a single byte to encode the data type of the original waveform, but assume the effect of waveform_dtype is negligible on the total size in bytes

    # determine if mono
    is_mono = type(blocks[0]) is not list
    if is_mono:
        blocks = [blocks] # add multiple channels if mono

    # iterate through blocks
    for channel in blocks:
        for block in channel:
            block_length, lpc_order, linear_prediction_coefficients, k, residuals_rice = block
            size += utils.MAXIMUM_BLOCK_SIZE_ASSUMPTION_BYTES # we assume the block length can be encoded as an unsigned integer in 2-bytes (np.int16); in other words, block_length must be strictly less than (2 ** 16 = 65536)
            size += 1 # we assume that we can encode the LPC order in 5 bits and the rice parameter K in another 3 bits, so together, 1 byte total
            size += linear_prediction_coefficients.nbytes # the size of the linear_prediction_coefficients is easily known from the LPC order, which is a fixed hyperparameter
            size += len(residuals_rice) # rice residuals can be easily decoded since we know the block_length
    
    # return the size in bytes
    return size

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":
    
    # read in arguments
    def parse_args(args = None, namespace = None):
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(prog = "Evaluate", description = "Evaluate IFLAC Implementation on a Test File") # create argument parser
        parser.add_argument("-p", "--path", type = str, default = f"{dirname(realpath(__file__))}/test.wav", help = "Absolute filepath to the WAV file.")
        parser.add_argument("--mono", action = "store_true", help = "Ensure that the WAV file is mono (single-channeled).")
        parser.add_argument("--block_size", type = int, default = utils.BLOCK_SIZE, help = "Block size.")
        parser.add_argument("--no_interchannel_decorrelate", action = "store_true", help = "Turn off interchannel-decorrelation.")
        args = parser.parse_args(args = args, namespace = namespace) # parse arguments
        args.interchannel_decorrelate = not args.no_interchannel_decorrelate # infer interchannel decorrelation
        return args # return parsed arguments
    args = parse_args()

    # load in wav file
    sample_rate, waveform = scipy.io.wavfile.read(filename = args.path)

    # force to mono if necessary
    if args.mono and waveform.ndim == 2:
        waveform = np.mean(a = waveform, axis = -1)

    # print statistics about waveform
    print(f"Waveform Shape: {tuple(waveform.shape)}")
    print(f"Waveform Sample Rate: {sample_rate:,} Hz")
    print(f"Waveform Data Type: {waveform.dtype}")
    waveform_size = utils.get_waveform_size(waveform = waveform)
    print(f"Waveform Size: {waveform_size:,} bytes")
    
    # encode
    print("Encoding...")
    start_time = time.perf_counter()
    bottleneck = encode(waveform = waveform, block_size = args.block_size, interchannel_decorrelate = args.interchannel_decorrelate)
    compression_speed = utils.get_compression_speed(duration_audio = len(waveform) / sample_rate, duration_encoding = time.perf_counter() - start_time)
    del start_time # free up memory
    bottleneck_size = get_bottleneck_size(bottleneck = bottleneck) # compute size of bottleneck in bytes
    print(f"Bottleneck Size: {bottleneck_size:,} bytes")
    print(f"Compression Rate: {100 * utils.get_compression_rate(size_original = waveform_size, size_compressed = bottleneck_size):.4f}%")
    print(f"Compression Speed: {compression_speed:.4f}")

    # decode
    print("Decoding...")
    round_trip = decode(bottleneck = bottleneck, interchannel_decorrelate = args.interchannel_decorrelate)

    # verify losslessness
    assert np.array_equal(waveform, round_trip), "Original and reconstructed waveforms do not match!"
    print("Encoding is lossless!")

##################################################