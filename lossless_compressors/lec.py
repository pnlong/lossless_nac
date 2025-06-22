# README
# Phillip Long
# June 12, 2025

# Implementation of Lossless EnCodec, which uses the Meta's EnCodec 
# to compress audio (see https://github.com/facebookresearch/encodec).

# IMPORTS
##################################################

import numpy as np
from typing import List, Tuple
from math import ceil
import librosa
import scipy
import argparse
import torch
from os.path import dirname, realpath
import time
import warnings

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))
sys.path.insert(0, f"{dirname(dirname(realpath(__file__)))}/encodec") # import encodec package

import utils
import rice
import encodec
import logging_for_zach

# ignore deprecation warning from pytorch
warnings.filterwarnings(action = "ignore", message = "torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm")

##################################################


# CONSTANTS
##################################################

# encodec target bandwidth
POSSIBLE_ENCODEC_TARGET_BANDWIDTHS = (3.0, 6.0, 12.0, 24.0)
TARGET_BANDWIDTH = POSSIBLE_ENCODEC_TARGET_BANDWIDTHS[1]

##################################################


# ENCODE
##################################################

def encode_block(
        block: np.array, # block tensor of shape (n_samples, n_channels)
        model: encodec.model.EncodecModel = encodec.EncodecModel.encodec_model_48khz(), # encodec model
        device: torch.device = torch.device("cpu"), # device the model is on
        sample_rate: int = utils.SAMPLE_RATE, # sample rate of waveform
    ) -> Tuple[int, int, np.array, bytes]: # returns tuple of the number of samples in the block, the size of the last dimension of codes, compressed material, and rice encoded residuals
    """LEC encoder helper function that encodes blocks."""

    # preprocess block for encodec
    block_array = block.copy() # of shape (n_samples, n_channels)
    n_samples_in_block = len(block_array)
    is_mono = block_array.shape[-1] == 1 # we assume that the waveform was either mono (n_channels = 1) or stereo (n_channels = 2)
    block = utils.convert_waveform_fixed_to_floating(waveform = block, output_dtype = np.float32)
    block = torch.from_numpy(block).T # convert to torch tensor, transpose to shape (n_channels, n_samples)
    block = encodec.utils.convert_audio(wav = block, sr = sample_rate, target_sr = model.sample_rate, target_channels = model.channels)
    block = block.unsqueeze(dim = 0) # add batch size dimension
    block = block.to(device) # ensure on correct device

    # encode to discrete codes (bottleneck)
    encoded_frames = model.encode(x = block) # returns list of (codes, scales)
    codes = torch.cat(tensors = [encoded_frame[0] for encoded_frame in encoded_frames], dim = -1) # shape (1, n_q, T)
    codes_T_dimension_size = codes.shape[-1] # size of the last dimension of codes, needed for reconstructing codes from bits

    # decode approximation from codes
    approximate_block = model.decode(encoded_frames = [(codes, None)]) # shape (1, n_channels, n_samples)
    approximate_block = approximate_block.squeeze(dim = 0)[:, :n_samples_in_block].detach().cpu().numpy() # truncate to match, also remove batch_size to shape (n_channels, n_samples)
    approximate_block = approximate_block[0].unsqueeze(dim = 0) if is_mono else approximate_block # if the waveform is mono, only have one channel
    approximate_block = approximate_block.T # tranpose to shape (n_samples, n_channels)
    approximate_block = utils.convert_waveform_floating_to_fixed(waveform = approximate_block, output_dtype = block_array.dtype) # convert approximate waveform to fixed-point for residual calculation

    # remove batch_size dimension from codes
    codes = codes.squeeze(dim = 0)
    codes = codes.detach().cpu().numpy() # convert to numpy
    codes = codes.flatten() # we flatten codes, as this is more similar to what we store as bits

    # compute residual and encode with rice coding
    residuals = block_array - approximate_block # compute residual
    residuals = residuals.flatten() # flatten residuals
    residuals_rice = rice.encode(nums = residuals) # rice encoding
    
    # free up gpu memory immediately
    del block, block_array, encoded_frames, approximate_block, residuals
    
    # return number of samples in block, size of the last dimension of codes, compressed materials, and rice encoded residuals
    return n_samples_in_block, codes_T_dimension_size, codes, residuals_rice


def encode(
        waveform: np.array, # waveform of integers of shape (n_samples, n_channels) (if multichannel) or (n_samples,) (if mono)
        sample_rate: int = utils.SAMPLE_RATE, # sample rate of waveform
        model: encodec.model.EncodecModel = encodec.EncodecModel.encodec_model_48khz(), # encodec model
        device: torch.device = torch.device("cpu"), # device the model is on
        block_size: int = utils.BLOCK_SIZE, # block size
        log_for_zach_kwargs: dict = None, # available keyword arguments for log_for_zach() function
    ) -> Tuple[List[Tuple[int, int, np.array, bytes]], type, bool]: # returns tuple of blocks, data type of original data, and whether the original data was mono
    """Naive LEC encoder."""

    # ensure waveform is correct type
    waveform_dtype = waveform.dtype
    assert any(waveform_dtype == dtype for dtype in utils.VALID_AUDIO_DTYPES)

    # deal with different size inputs
    is_mono = waveform.ndim == 1
    if is_mono: # if mono
        waveform = np.expand_dims(a = waveform, axis = -1) # add channel to represent single channel

    # go through blocks and encode them each
    n_samples = len(waveform)
    n_blocks = ceil(n_samples / block_size)
    blocks = utils.rep(x = None, times = n_blocks)
    for i in range(n_blocks):
        start_index = i * block_size
        end_index = (start_index + block_size) if (i < (n_blocks - 1)) else n_samples
        blocks[i] = encode_block(block = waveform[start_index:end_index], model = model, device = device, sample_rate = sample_rate)
    
    # log for zach
    if log_for_zach_kwargs is not None:
        residuals = [(rice.decode(stream = block[-1], n = block[0] * (1 if is_mono else 2)), block[0]) for block in blocks]
        residuals = np.concatenate([block if is_mono else block.reshape(n_samples_in_block, -1) for block, n_samples_in_block in residuals], axis = 0)
        residuals_rice = rice.encode(nums = residuals.flatten())
        logging_for_zach.log_for_zach(
            residuals = residuals,
            residuals_rice = residuals_rice,
            **log_for_zach_kwargs)
    
    # return blocks, waveform data type, and whether the original waveform was mono
    return blocks, waveform_dtype, is_mono

##################################################


# DECODE
##################################################

def decode_block(
        block: Tuple[int, int, np.array, bytes], # block tuple with elements (n_samples_in_block, codes_T_dimension_size, bottleneck, residuals_rice)
        model: encodec.model.EncodecModel = encodec.EncodecModel.encodec_model_48khz(), # encodec model
        device: torch.device = torch.device("cpu"), # device the model is on
        waveform_dtype: type = utils.DEFAULT_AUDIO_DTYPE, # the data type of the original data
        is_mono: bool = False, # whether the original waveform was mono
    ) -> np.array:
    """LEC decoder helper function that decodes blocks."""

    # split block
    n_samples_in_block, codes_T_dimension_size, codes, residuals_rice = block
    codes = codes.reshape(-1, codes_T_dimension_size) # reshape codes using codes_T_dimension size to shape (n_q, T)
    codes = torch.from_numpy(codes).to(device) # ensure on correct device
    codes = codes.unsqueeze(dim = 0) # add batch_size dimension

    # decode approximation from codes
    approximate_block = model.decode(encoded_frames = [(codes, None)]) # shape (1, n_channels, n_samples)
    approximate_block = approximate_block.squeeze(dim = 0)[:, :n_samples_in_block].cpu().numpy() # truncate to match, also remove batch_size to shape (n_channels, n_samples)
    approximate_block = approximate_block[0] if is_mono else approximate_block.T # tranpose to shape (n_samples, n_channels)
    approximate_block = utils.convert_waveform_floating_to_fixed(waveform = approximate_block, output_dtype = waveform_dtype) # convert approximate waveform to fixed-point for residual calculation

    # free up gpu memory immediately
    del codes, codes_T_dimension_size

    # get residuals
    residuals = rice.decode(stream = residuals_rice, n = n_samples_in_block * (1 if is_mono else 2)) # shape is (n_samples * (1 if is_mono else 2), )
    residuals = residuals.astype(approximate_block.dtype) # ensure correct data type
    if not is_mono: # reshape to stereo if not mono
        residuals = residuals.reshape(n_samples_in_block, -1) # shape to (n_samples, 2) if stereo, otherwise (n_samples,)

    # free up memory
    del residuals_rice, n_samples_in_block

    # reconstruct the exact waveform
    block = approximate_block + residuals

    # return reconstructed waveform for block of shape (n_samples,) if mono else (n_samples, 2)
    return block


def decode(
        bottleneck: Tuple[List[Tuple[int, int, np.array, bytes]], type, bool], # tuple of blocks, the datatype of the original waveform, and whether the original data was mono
        model: encodec.model.EncodecModel = encodec.EncodecModel.encodec_model_48khz(), # encodec model
        device: torch.device = torch.device("cpu"), # device the model is on
    ) -> np.array: # returns the reconstructed waveform of shape (n_samples, n_channels) (if multichannel) or (n_samples,) (if mono)
    """Naive LEC decoder."""

    # split bottleneck
    blocks, waveform_dtype, is_mono = bottleneck
    
    # go through blocks
    n_blocks = len(blocks)
    waveform = utils.rep(x = None, times = n_blocks)
    for i in range(n_blocks):
        waveform[i] = decode_block(block = blocks[i], model = model, device = device, waveform_dtype = waveform_dtype, is_mono = is_mono)

    # reconstruct final waveform
    waveform = np.concatenate(waveform, axis = 0) # concatenate blocks into single waveform

    # return final reconstructed waveform
    return waveform

##################################################


# HELPER FUNCTION TO GET THE SIZE IN BYTES OF THE BOTTLENECK
##################################################

def get_bottleneck_size(
    bottleneck: Tuple[List[Tuple[int, int, np.array, bytes]], type, bool],
) -> int:
    """Returns the size of the given bottleneck in bytes."""

    # tally of the size in bytes of the bottleneck
    size = 0

    # split bottleneck
    blocks, waveform_dtype, is_mono = bottleneck
    # size += 1 # use a 7 bits to encode the data type of the original waveform and the final bit (so one byte total) to encode whether or not the waveform was mono, but assume the effect of waveform_dtype is negligible on the total size in bytes

    # iterate through blocks
    for block in blocks:
        block_length, codes_T_dimension_size, codes, residuals_rice = block
        size += utils.MAXIMUM_BLOCK_SIZE_ASSUMPTION_BYTES # we assume the block length can be encoded as an unsigned integer in 2-bytes (np.int16); in other words, block_length must be strictly less than (2 ** 16 = 65536)
        size += 1 # we assume that we can encode the codes_T_dimension size in a single unsigned 8-bit integer (1 byte), so that means we assume the codes_T_dimension size is strictly less than (2 ** 8 = 256)
        size += codes.nbytes # the size of codes is determined by the target bandwidth of the model (n_q) and codes_T_dimension_size
        size += len(residuals_rice) # rice residuals can be easily decoded since we know the block_length and whether or not the waveform was mono

    # return the size in bytes
    return size

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":
    
    # read in arguments
    def parse_args(args = None, namespace = None):
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(prog = "Evaluate", description = "Evaluate Naive-LEC Implementation on a Test File") # create argument parser
        parser.add_argument("-p", "--path", type = str, default = f"{dirname(realpath(__file__))}/test.wav", help = "Absolute filepath to the WAV file.")
        parser.add_argument("--mono", action = "store_true", help = "Ensure that the WAV file is mono (single-channeled).")
        parser.add_argument("--target_bandwidth", type = float, choices = POSSIBLE_ENCODEC_TARGET_BANDWIDTHS, default = TARGET_BANDWIDTH, help = "Target bandwidth for EnCodec model. The number of codebooks used will be determined by the bandwidth selected (see https://github.com/facebookresearch/encodec#:~:text=The%20number%20of%20codebooks%20used%20will%20be%20determined%20bythe%20bandwidth%20selected.).")
        parser.add_argument("--block_size", type = int, default = utils.BLOCK_SIZE, help = "Block size.") # int(model.sample_rate * 0.99) # the 48 kHz encodec model processes audio in one-second chunks with 1% overlap
        parser.add_argument("-g", "--gpu", type = int, default = -1, help = "GPU (-1 for CPU).")
        args = parser.parse_args(args = args, namespace = namespace) # parse arguments
        return args # return parsed arguments
    args = parse_args()

    # load in wav file
    sample_rate, waveform = scipy.io.wavfile.read(filename = args.path)

    # print statistics about waveform
    print(f"Waveform Shape: {tuple(waveform.shape)}")
    print(f"Waveform Sample Rate: {sample_rate:,} Hz")
    print(f"Waveform Data Type: {waveform.dtype}")
    waveform_size = utils.get_waveform_size(waveform = waveform)
    print(f"Waveform Size: {waveform_size:,} bytes")
    waveform_reshaped = False

    # force to mono if necessary
    if args.mono and waveform.ndim == 2:
        print(f"Forcing waveform to mono!")
        waveform = np.mean(a = waveform, axis = -1)
        waveform_reshaped = True

    # load encodec
    device = torch.device(f"cuda:{abs(args.gpu)}" if torch.cuda.is_available() and args.gpu != -1 else "cpu")
    model = encodec.EncodecModel.encodec_model_48khz().to(device)
    model.set_target_bandwidth(bandwidth = args.target_bandwidth)

    # resample if necessary
    if sample_rate != model.sample_rate:
        print(f"Resampling waveform from {sample_rate:,} Hz to {model.sample_rate:,} Hz to match the sample rate required by EnCodec.")
        waveform = utils.convert_waveform_floating_to_fixed(
            waveform = librosa.resample(y = utils.convert_waveform_fixed_to_floating(waveform = waveform), orig_sr = sample_rate, target_sr = model.sample_rate, axis = 0),
            output_dtype = waveform.dtype)
        waveform_reshaped = True
        sample_rate = model.sample_rate

    # print new waveform shape if necessary
    if waveform_reshaped:
        print(f"New Waveform Shape: {tuple(waveform.shape)}")
        print(f"New Waveform Sample Rate: {sample_rate:,} Hz")
        print(f"New Waveform Data Type: {waveform.dtype}")
        waveform_size = utils.get_waveform_size(waveform = waveform)
        print(f"New Waveform Size: {waveform_size:,} bytes")

    # turn off gradients, since the model is pretrained
    model.eval()
    with torch.no_grad():

        # encode
        print("Encoding...")
        start_time = time.perf_counter()
        bottleneck = encode(waveform = waveform, sample_rate = sample_rate, model = model, device = device, block_size = args.block_size)
        compression_speed = utils.get_compression_speed(duration_audio = len(waveform) / sample_rate, duration_encoding = time.perf_counter() - start_time)
        del start_time # free up memory
        bottleneck_size = get_bottleneck_size(bottleneck = bottleneck) # compute size of bottleneck in bytes
        print(f"Bottleneck Size: {bottleneck_size:,} bytes")
        print(f"Compression Rate: {100 * utils.get_compression_rate(size_original = waveform_size, size_compressed = bottleneck_size):.4f}%")
        print(f"Compression Speed: {compression_speed:.4f}")

        # decode
        print("Decoding...")
        round_trip = decode(bottleneck = bottleneck, model = model, device = device)

    # verify losslessness
    assert np.array_equal(waveform, round_trip), "Original and reconstructed waveforms do not match!"
    print("Encoding is lossless!")

##################################################