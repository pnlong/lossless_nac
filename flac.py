# README
# Phillip Long
# May 11, 2025

# Implementation of Free Lossless Audio Codec (FLAC) for use as a baseline.
# See https://xiph.org/flac/documentation_format_overview.html for more.

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

import utils
import rice

##################################################


# HELPER FUNCTION FOR COMPUTING LINEAR PREDICTION COEFFICIENTS USING THE AUTOCOVARIANCE/AUTOCORRELATION METHOD
##################################################

def levinson_durbin(r: np.array, order: int = utils.LPC_ORDER) -> Tuple[np.array, float, np.array]:
    """
    Levinson-Durbin recursion to solve Toeplitz systems for linear predictive coding.

    Parameters
    ----------
    r : np.array
        autocorrelation sequence of length >= order + 1
    order : int, default: `utils.LPC_ORDER`
        desired linear predictive coding order
        
    Returns
    -------
    a : np.array
        linear predictive coding coefficients of shape (order + 1,)
    e : float
        prediction error (residual energy)
    k : np.array
        reflection coefficients (for optional analysis)
    """
    a = np.zeros(shape = order + 1, dtype = np.float64)
    e = r[0]
    
    if e == 0:
        return a, e, np.zeros(shape = order)
    
    a[0] = 1.0
    k = np.zeros(shape = order, dtype = np.float64)
    
    for i in range(1, order + 1):
        acc = r[i]
        acc += np.dot(a[1:i], np.flip(r[1:i]))
        for j in range(1, i):
            acc += a[j] * r[i - j]
        
        k_i = -acc / e
        k[i - 1] = k_i
        
        a[1:(i + 1)] += k_i * np.flip(a[1:(i + 1)])
        
        e *= (1 - (k_i ** 2))
        if e <= 0 or not np.isfinite(e): # numerical issues fallback
            break
    
    return a, e, k

def lpc_autocorrelation_method(y: np.array, order: int = utils.LPC_ORDER) -> np.array:
    """
    Compute linear prediction coefficients using autocorrelation method for guaranteed stability.
    See https://speechprocessingbook.aalto.fi/Representations/Linear_prediction.html#:~:text=A%20benefit%20of,signal%20with%20prediction.
    """
    y = y.astype(np.float64)
    r = np.correlate(a = y, v = y, mode = "full")
    r = r[(len(y) - 1):(len(y) - 1 + order + 1)] # take autocorrelations from lag 0 to lag order
    a, e, k = levinson_durbin(r = r, order = order)
    return a

##################################################


# ENCODE
##################################################

def encode_block(
        block: np.array, # block of integers of shape (n_samples_in_block,)
        order: int = utils.LPC_ORDER, # order for linear predictive coding
    ) -> Tuple[int, np.array, bytes]: # returns tuple of number of samples in the block, compressed material, and rice encoded residuals
    """FLAC encoder helper function that encodes blocks."""

    # convert block to float
    block_float = block.astype(np.float32)

    # fit linear prediction coefficients, then quantize
    # linear_prediction_coefficients = librosa.lpc(y = block_float, order = order) # does not guarantee numerical stability
    linear_prediction_coefficients = lpc_autocorrelation_method(y = block_float, order = order)
    linear_prediction_coefficients = np.round(linear_prediction_coefficients).astype(utils.LPC_DTYPE)
    if not np.all(np.abs(np.roots(linear_prediction_coefficients) < 1)): # ensure lpc coefficients are stable
        warnings.warn(message = "Linear prediction coefficients are unstable!", category = RuntimeWarning)
    
    # autoregressive prediction using linear prediction coefficients
    approximate_block = scipy.signal.lfilter(b = np.concatenate(([0], -linear_prediction_coefficients), axis = 0, dtype = linear_prediction_coefficients.dtype), a = [1], x = block_float)
    approximate_block = np.round(approximate_block).astype(block.dtype) # ensure approximate waveform is integer values
    
    # compute residual and encode with rice coding
    residuals = block - approximate_block
    residuals_rice = rice.encode(nums = residuals) # rice encoding
    
    # return number of samples in block, compressed materials, and rice encoded residuals
    return len(block), linear_prediction_coefficients, residuals_rice


def encode(
        waveform: np.array, # waveform of integers of shape (n_samples, n_channels) (if multichannel) or (n_samples,) (if mono)
        block_size: int = utils.BLOCK_SIZE, # block size
        interchannel_decorrelate: bool = utils.INTERCHANNEL_DECORRELATE, # use interchannel decorrelation
        order: int = utils.LPC_ORDER, # order for linear predictive coding
    ) -> Tuple[List[Union[Tuple[int, np.array, bytes], List[Tuple[int, np.array, bytes]]]], type]: # returns tuple of blocks and data type of original data
    """Naive FLAC encoder."""

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
        block: Tuple[int, np.array, bytes], # block tuple with elements (n_samples_in_block, bottleneck, residuals_rice)
    ) -> np.array:
    """FLAC decoder helper function that decodes blocks."""

    # split block
    n_samples_in_block, linear_prediction_coefficients, residuals_rice = block # lpc_order = len(linear_prediction_coefficients) - 1; len(linear_prediction_coefficients) = lpc_order + 1

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
        bottleneck: Tuple[List[Union[Tuple[int, np.array, bytes], List[Tuple[int, np.array, bytes]]]], type], # tuple of blocks and the datatype of the original waveform
        interchannel_decorrelate: bool = utils.INTERCHANNEL_DECORRELATE, # was interchannel decorrelation used
    ) -> np.array: # returns the reconstructed waveform of shape (n_samples, n_channels) (if multichannel) or (n_samples,) (if mono)
    """Naive FLAC decoder."""

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
    bottleneck: Tuple[List[Union[Tuple[int, np.array, bytes], List[Tuple[int, np.array, bytes]]]], type],
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
            block_length, linear_prediction_coefficients, residuals_rice = block
            size += utils.MAXIMUM_BLOCK_SIZE_ASSUMPTION_BYTES # we assume the block length can be encoded as an unsigned integer in 2-bytes (np.int16); in other words, block_length must be strictly less than (2 ** 16 = 65536)
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
        parser = argparse.ArgumentParser(prog = "Evaluate", description = "Evaluate Naive-FLAC Implementation on a Test File") # create argument parser
        parser.add_argument("-p", "--path", type = str, default = f"{dirname(realpath(__file__))}/test.wav", help = "Absolute filepath to the WAV file.")
        parser.add_argument("--mono", action = "store_true", help = "Ensure that the WAV file is mono (single-channeled).")
        parser.add_argument("--block_size", type = int, default = utils.BLOCK_SIZE, help = "Block size.")
        parser.add_argument("--no_interchannel_decorrelate", action = "store_true", help = "Turn off interchannel-decorrelation.")
        parser.add_argument("--lpc_order", type = int, default = utils.LPC_ORDER, help = "Order for linear predictive coding.")
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
    bottleneck = encode(waveform = waveform, block_size = args.block_size, interchannel_decorrelate = args.interchannel_decorrelate, order = args.lpc_order)
    compression_speed = utils.convert_duration_to_speed(duration_audio = len(waveform) / sample_rate, duration_encoding = time.perf_counter() - start_time)
    del start_time # free up memory
    bottleneck_size = get_bottleneck_size(bottleneck = bottleneck) # compute size of bottleneck in bytes
    print(f"Bottleneck Size: {bottleneck_size:,} bytes")
    print(f"Compression Rate: {100 * (bottleneck_size / waveform_size):.4f}%")
    print(f"Compression Speed: {compression_speed:.4f}")

    # decode
    print("Decoding...")
    round_trip = decode(bottleneck = bottleneck, interchannel_decorrelate = args.interchannel_decorrelate)

    # verify losslessness
    assert np.array_equal(waveform, round_trip), "Original and reconstructed waveforms do not match!"
    print("Encoding is lossless!")

##################################################