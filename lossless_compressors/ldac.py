# README
# Phillip Long
# May 24, 2025

# Implementation of Lossless Descript Audio Codec, which uses the Descript Audio Codec 
# to compress audio (see https://github.com/descriptinc/descript-audio-codec).

# IMPORTS
##################################################

import numpy as np
from typing import List, Tuple, Union
import librosa
import scipy
import argparse
from audiotools import AudioSignal
import torch
from os.path import dirname, realpath
import time
import warnings

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))
sys.path.insert(0, f"{dirname(dirname(realpath(__file__)))}/dac") # import dac package

import utils
import rice
import dac
import logging_for_zach

# ignore deprecation warning from pytorch
warnings.filterwarnings(action = "ignore", message = "torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm")

##################################################


# CONSTANTS
##################################################

# path to descript audio codec
DAC_PATH = "/home/pnlong/.cache/descript/dac/weights_44khz_8kbps_0.0.1.pth" # path to descript audio codec pretrained

# number of codebooks for descript audio codec model
POSSIBLE_DAC_N_CODEBOOKS = tuple(range(1, 10)) # upper-bounded by the number of codebooks for the pretrained descript audio codec model, 9
N_CODEBOOKS = POSSIBLE_DAC_N_CODEBOOKS[-1] # use the upper bound as the default

# scaling factor for DAC processing
AUDIO_SCALE = 32768.0 # audio-appropriate scaling factor for DAC processing

##################################################


# CONVERSION FUNCTIONS
##################################################

def convert_audio_fixed_to_floating(waveform: np.array, audio_scale: float = AUDIO_SCALE) -> np.array:
    """
    Convert fixed-point audio to floating-point using appropriate audio scaling.
    
    This replaces utils.convert_waveform_fixed_to_floating which uses full dtype range
    and creates artificially tiny values that lead to huge residuals after DAC processing.
    """
    return waveform.astype(np.float32) / audio_scale

def convert_audio_floating_to_fixed(waveform: np.array, output_dtype: type = np.int32, audio_scale: float = AUDIO_SCALE) -> np.array:
    """
    Convert floating-point audio to fixed-point using appropriate audio scaling.
    
    This replaces utils.convert_waveform_floating_to_fixed which uses full dtype range
    and creates artificially huge residuals from tiny DAC outputs.
    """
    return np.round(waveform * audio_scale).astype(output_dtype)


# ENCODE
##################################################

def encode_block(
        block: AudioSignal, # AudioSignal object for block of shape (batch_size = 1, n_channels = 1, n_samples_in_block)
        model: dac.model.dac.DAC = dac.DAC.load(DAC_PATH), # descript audio codec model already on the relevant device
        n_codebooks: int = N_CODEBOOKS, # number of codbeooks to use for descript audio codec model
        k: int = rice.K, # rice parameter
    ) -> Tuple[int, np.array, bytes]: # returns tuple of number of samples in the block, compressed material, and rice encoded residuals
    """LDAC encoder helper function that encodes blocks."""

    # get bottleneck
    block_numpy = block.audio_data.numpy()
    block_array = block_numpy.squeeze(axis = 0).squeeze(axis = 0) # get version of block that is 1d array
    n_samples_in_block = len(block_array)
    x = model.preprocess(
        audio_data = torch.from_numpy(convert_audio_fixed_to_floating(waveform = block_numpy)).to(model.device), # ensure floating point and on correct device
        sample_rate = block.sample_rate,
    ).float() # convert to float to avoid RuntimeError: Input type (torch.cuda.DoubleTensor) and weight type (torch.cuda.FloatTensor) should be the same
    _, codes, _, _, _ = model.encode(audio_data = x)
    codes = codes[:, :n_codebooks, :] # truncate codes to desired number of codebooks
    del block, block_numpy, x # free up memory immediately

    # approximate block
    z = model.quantizer.from_codes(codes = codes)[0].detach() # get z from codes
    approximate_block = model.decode(z = z)
    approximate_block = approximate_block.squeeze(dim = (0, 1)).detach().cpu().numpy() # get rid of unnecessary dimensions
    approximate_block = approximate_block[:n_samples_in_block] # truncate to correct length
    approximate_block = convert_audio_floating_to_fixed(waveform = approximate_block, output_dtype = block_array.dtype) # ensure approximate waveform is integer values
    del z # free up memory immediately

    # remove batch_size dimension from codes
    codes = codes.squeeze(dim = 0)
    codes = codes.detach().cpu().numpy() # convert to numpy

    # compute residual and encode with rice coding
    residuals = block_array - approximate_block
    residuals_rice = rice.encode(nums = residuals, k = k) # rice encoding
    del block_array, approximate_block, residuals # free up memory immediately
    
    # return number of samples in block, compressed materials, and rice encoded residuals
    return n_samples_in_block, codes, residuals_rice


def encode(
        waveform: np.array, # waveform of integers of shape (n_samples, n_channels) (if multichannel) or (n_samples,) (if mono)
        sample_rate: int = utils.SAMPLE_RATE, # sample rate of waveform, needed for audiotools
        model: dac.model.dac.DAC = dac.DAC.load(DAC_PATH), # descript audio codec model already on the relevant device
        block_size: int = utils.BLOCK_SIZE, # block size
        interchannel_decorrelate: bool = utils.INTERCHANNEL_DECORRELATE, # use interchannel decorrelation
        n_codebooks: int = N_CODEBOOKS, # number of codbeooks to use for descript audio codec model
        k: int = rice.K, # rice parameter
        overlap: float = utils.OVERLAP, # block overlap (as a percentage in range [0, 100))
        log_for_zach_kwargs: dict = None, # available keyword arguments for log_for_zach() function
    ) -> Tuple[type, type, List[Union[Tuple[int, np.array, bytes], List[Tuple[int, np.array, bytes]]]]]: # returns tuple of the datatype of the original waveform, the datatype of the waveform as passed to blocks to encode, and the blocks themselves
    """Naive LDAC encoder."""

    # ensure waveform is correct type
    waveform_dtype = waveform.dtype
    assert any(waveform_dtype == dtype for dtype in utils.VALID_AUDIO_DTYPES)

    # deal with different size inputs
    is_mono = waveform.ndim == 1        
    if is_mono: # if mono
        waveform = np.expand_dims(a = waveform, axis = -1) # add channel to represent single channel
        interchannel_decorrelate = False # ensure interchannel decorrelate is false if the signal is mono
    elif interchannel_decorrelate and waveform.ndim == 2 and waveform.shape[-1] == 2: # if stereo, perform inter-channel decorrelation (https://xiph.org/flac/documentation_format_overview.html#:~:text=smaller%20frame%20header.-,INTER%2DCHANNEL%20DECORRELATION,-In%20the%20case)
        left, right = waveform.T.astype(utils.INTERCHANNEL_DECORRELATE_DTYPE_BY_AUDIO_DTYPE[str(waveform_dtype)]) # extract left and right channels, cast as int64 so there are no overflow bugs
        center = (left + right) >> 1 # center channel
        side = left - right # side channel
        waveform = np.stack(arrays = (center, side), axis = -1)
        del left, right, center, side
    encoding_dtype = waveform.dtype

    # convert waveform to audio signal for DAC
    waveform = AudioSignal(audio_path_or_array = waveform.T, sample_rate = sample_rate)

    # go through blocks and encode them each
    _, n_channels, n_samples = waveform.shape
    samples_overlap = int(block_size * (overlap / 100))
    blocks = [[] for _ in range(n_channels)]
    for channel_index in range(n_channels):
        i = 0
        while (start_index := (i * (block_size - samples_overlap))) < n_samples:
            end_index = min(start_index + block_size, n_samples)
            block = encode_block(block = waveform[:, channel_index, start_index:end_index], model = model, n_codebooks = n_codebooks, k = k)
            blocks[channel_index].append(block)
            i += 1

    # log for zach
    if log_for_zach_kwargs is not None:
        residuals = np.stack([np.concatenate([rice.decode(stream = block[-1], n = block[0], k = k) for block in channel], axis = 0) for channel in blocks], axis = -1)
        if is_mono:
            residuals = residuals.squeeze(dim = -1)
        residuals_rice = rice.encode(nums = residuals.flatten(), k = k)
        logging_for_zach.log_for_zach(
            residuals = residuals,
            residuals_rice = residuals_rice,
            **log_for_zach_kwargs)

    # don't have multiple channels if mono
    if is_mono:
        blocks = blocks[0]
    
    # return waveform data type, data type of waveform passed to blocks to encode, and the blocks
    return waveform_dtype, encoding_dtype, blocks

##################################################


# DECODE
##################################################

def decode_block(
        block: Tuple[int, np.array, bytes], # block tuple with elements (n_samples_in_block, bottleneck, residuals_rice)
        model: dac.model.dac.DAC = dac.DAC.load(DAC_PATH), # descript audio codec model already on the relevant device
        encoding_dtype: type = utils.DEFAULT_AUDIO_DTYPE, # data type of the resulting reconstructed waveform
        k: int = rice.K, # rice parameter
    ) -> np.array:
    """LDAC decoder helper function that decodes blocks."""

    # split block
    n_samples_in_block, codes, residuals_rice = block
    codes = torch.from_numpy(codes).to(model.device) # ensure on correct device
    codes = codes.unsqueeze(dim = 0) # add batch_size dimension

    # reconstruct the approximate waveform
    z = model.quantizer.from_codes(codes = codes)[0].detach() # get z from codes
    approximate_block = model.decode(z = z)
    approximate_block = approximate_block.squeeze(dim = (0, 1)).detach().cpu().numpy() # convert from AudioSignal to numpy array
    approximate_block = approximate_block[:n_samples_in_block] # truncate to correct length
    approximate_block = convert_audio_floating_to_fixed(waveform = approximate_block, output_dtype = encoding_dtype) # ensure approximate waveform is integer values

    # free up gpu memory immediately
    del codes, z

    # get residuals
    residuals = rice.decode(stream = residuals_rice, n = n_samples_in_block, k = k)
    residuals = residuals.astype(approximate_block.dtype)

    # reconstruct the exact waveform
    block = approximate_block + residuals

    # return reconstructed waveform for block
    return block


def decode(
        bottleneck: Tuple[type, type, List[Union[Tuple[int, np.array, bytes], List[Tuple[int, np.array, bytes]]]]], # tuple of the datatype of the original waveform, the datatype of the waveform as passed to blocks to encode, and the blocks themselves
        model: dac.model.dac.DAC = dac.DAC.load(DAC_PATH), # descript audio codec model already on the relevant device
        interchannel_decorrelate: bool = utils.INTERCHANNEL_DECORRELATE, # was interchannel decorrelation used
        k: int = rice.K, # rice parameter
        overlap: float = utils.OVERLAP, # block overlap (as a percentage in range [0, 100))
    ) -> np.array: # returns the reconstructed waveform of shape (n_samples, n_channels) (if multichannel) or (n_samples,) (if mono)
    """Naive LDAC decoder."""

    # split bottleneck
    waveform_dtype, encoding_dtype, blocks = bottleneck

    # determine if mono
    is_mono = type(blocks[0]) is not list
    if is_mono:
        blocks = [blocks] # add multiple channels if mono
        interchannel_decorrelate = False # ensure interchannel decorrelate is false if the signal is mono
    
    # go through blocks
    n_channels, n_blocks = len(blocks), len(blocks[0])
    block_size = blocks[0][0][0] # get block size
    samples_overlap = int(block_size * (overlap / 100))
    samples_overlap_first_half = int(samples_overlap / 2)
    samples_overlap_second_half = samples_overlap - samples_overlap_first_half
    waveform = [[None] * n_blocks for _ in range(n_channels)]
    for channel_index in range(n_channels):
        for i in range(n_blocks):
            waveform[channel_index][i] = decode_block(block = blocks[channel_index][i], model = model, encoding_dtype = encoding_dtype, k = k)
            waveform[channel_index][i] = waveform[channel_index][i][(samples_overlap_first_half if i > 0 else 0):(len(waveform[channel_index][i]) - (samples_overlap_second_half if i < (n_blocks - 1) else 0))] # truncate to account for overlap

    # reconstruct final waveform
    waveform = [np.concatenate(channel, axis = 0) for channel in waveform]
    waveform = np.stack(arrays = waveform, axis = -1)

    # don't have multiple channels if mono
    if is_mono: # if mono, ensure waveform is one dimension
        waveform = waveform[:, 0]
    elif interchannel_decorrelate and n_channels == 2: # if stereo, perform inter-channel decorrelation
        center, side = waveform.T # extract center and side channels, cast as int64 so there are no overflow bugs
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
    bottleneck: Tuple[type, type, List[Union[Tuple[int, np.array, bytes], List[Tuple[int, np.array, bytes]]]]],
) -> int:
    """Returns the size of the given bottleneck in bytes."""

    # tally of the size in bytes of the bottleneck
    size = 0

    # split bottleneck
    waveform_dtype, encoding_dtype, blocks = bottleneck
    # size += 1 # use a single byte to encode the data type of the original waveform and the data type of the waveform as passed to the block encoder, but assume the effect of these data types is negligible on the total size in bytes

    # determine if mono
    is_mono = type(blocks[0]) is not list
    if is_mono:
        blocks = [blocks] # add multiple channels if mono

    # iterate through blocks
    for channel in blocks:
        for block in channel:
            block_length, codes, residuals_rice = block
            size += utils.MAXIMUM_BLOCK_SIZE_ASSUMPTION_BYTES # we assume the block length can be encoded as an unsigned integer in 2-bytes (np.int16); in other words, block_length must be strictly less than (2 ** 16 = 65536)
            size += codes.nbytes # the size of codes is constant from the descript audio codec model, which is a fixed hyperparameter
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
        parser = argparse.ArgumentParser(prog = "Evaluate", description = "Evaluate Naive-LDAC Implementation on a Test File") # create argument parser
        parser.add_argument("-p", "--path", type = str, default = f"{dirname(dirname(realpath(__file__)))}/test.wav", help = "Absolute filepath to the WAV file.")
        parser.add_argument("-mp", "--model_path", type = str, default = DAC_PATH, help = "Absolute filepath to the Descript Audio Codec model weights.")
        parser.add_argument("--mono", action = "store_true", help = "Ensure that the WAV file is mono (single-channeled).")
        parser.add_argument("--block_size", type = int, default = utils.BLOCK_SIZE, help = "Block size.")
        parser.add_argument("--no_interchannel_decorrelate", action = "store_true", help = "Turn off interchannel-decorrelation.")
        parser.add_argument("--n_codebooks", type = int, choices = POSSIBLE_DAC_N_CODEBOOKS, default = N_CODEBOOKS, help = "Number of codebooks for DAC model.")
        parser.add_argument("--rice_parameter", type = int, default = rice.K, help = "Rice coding parameter.")
        parser.add_argument("--overlap", type = float, default = utils.OVERLAP, help = "Block overlap (as a percentage 0-100).")
        parser.add_argument("-g", "--gpu", type = int, default = -1, help = "GPU (-1 for CPU).")
        args = parser.parse_args(args = args, namespace = namespace) # parse arguments
        args.interchannel_decorrelate = not args.no_interchannel_decorrelate # infer interchannel decorrelation
        if args.overlap < 0 or args.overlap >= 100: # ensure overlap is valid
            raise RuntimeError(f"Overlap must be in range [0, 100), but received {args.overlap}!")
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
        waveform = np.round(np.mean(a = waveform, axis = -1)).astype(waveform.dtype)
        waveform_reshaped = True

    # load descript audio codec
    device = torch.device(f"cuda:{abs(args.gpu)}" if torch.cuda.is_available() and args.gpu != -1 else "cpu")
    model = dac.DAC.load(location = args.model_path).to(device)

    # resample if necessary
    if sample_rate != model.sample_rate:
        print(f"Resampling waveform from {sample_rate:,} Hz to {model.sample_rate:,} Hz to match the sample rate required by the Descript Audio Codec.")
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
        bottleneck = encode(waveform = waveform, sample_rate = sample_rate, model = model, block_size = args.block_size, interchannel_decorrelate = args.interchannel_decorrelate, n_codebooks = args.n_codebooks, k = args.rice_parameter, overlap = args.overlap)
        compression_speed = utils.get_compression_speed(duration_audio = len(waveform) / sample_rate, duration_encoding = time.perf_counter() - start_time)
        del start_time # free up memory
        bottleneck_size = get_bottleneck_size(bottleneck = bottleneck) # compute size of bottleneck in bytes
        print(f"Bottleneck Size: {bottleneck_size:,} bytes")
        print(f"Compression Rate: {100 * utils.get_compression_rate(size_original = waveform_size, size_compressed = bottleneck_size):.4f}%")
        print(f"Compression Speed: {compression_speed:.4f}")

        # decode
        print("Decoding...")
        round_trip = decode(bottleneck = bottleneck, model = model, interchannel_decorrelate = args.interchannel_decorrelate, k = args.rice_parameter, overlap = args.overlap)

    # verify losslessness
    assert np.array_equal(waveform, round_trip), "Original and reconstructed waveforms do not match!"
    print("Encoding is lossless!")

##################################################