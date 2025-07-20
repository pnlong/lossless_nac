# README
# Phillip Long
# July 6, 2025

# FLAC LPC Compressor.

# IMPORTS
##################################################

import numpy as np
import subprocess
import tempfile
from os import remove
from os.path import dirname, realpath, exists
import multiprocessing
import functools
from typing import List, Tuple
import logging

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, f"{dirname(dirname(realpath(__file__)))}/entropy_coders")
sys.path.insert(0, dirname(dirname(dirname(realpath(__file__)))))

from lossless_compressors import LosslessCompressor, partition_data_into_frames, INTERCHANNEL_DECORRELATION_DEFAULT, INTERCHANNEL_DECORRELATION_SCHEMES_MAP, REVERSE_INTERCHANNEL_DECORRELATION_SCHEMES_MAP, JOBS_DEFAULT, BLOCK_SIZE_DEFAULT
from entropy_coders import EntropyCoder
import utils

##################################################


# CONSTANTS
##################################################

FLAC_LPC_HELPERS_DIR = f"{dirname(realpath(__file__))}/flac_lpc_helpers" # directory that contains the FLAC LPC encode and decode scripts
FLAC_LPC_ENCODE_SCRIPT_FILEPATH = f"{FLAC_LPC_HELPERS_DIR}/flac_lpc_encode.py" # filepath to FLAC LPC encode script
FLAC_LPC_DECODE_SCRIPT_FILEPATH = f"{FLAC_LPC_HELPERS_DIR}/flac_lpc_decode.py" # filepath to FLAC LPC decode script
FLAC_ENCODING_METHODS = ["verbatim", "constant", "fixed", "lpc"] # encoding methods for FLAC
FLAC_ENCODING_METHODS_WITH_ENTROPY = set(FLAC_ENCODING_METHODS[2:]) # encoding methods that have entropy that needs to be encoded are fixed and lpc, use set for fast lookup

##################################################


# BOTTLENECK TYPE
##################################################

# type of bottleneck subframes
BOTTLENECK_SUBFRAME_TYPE = Tuple[int, bytes, bytes] # bottleneck subframe type is a tuple of the number of samples, encoded estimator data, and encoded residuals

# type of bottleneck frames
BOTTLENECK_FRAME_TYPE = Tuple[int, List[BOTTLENECK_SUBFRAME_TYPE]] # bottleneck frame type is a tuple of the interchannel decorrelation scheme index and list of subframes

# type of bottleneck
BOTTLENECK_TYPE = List[BOTTLENECK_FRAME_TYPE]

##################################################


# FLAC LPC ESTIMATOR FUNCTIONS
##################################################

def encode_estimator_data(subframe_data: np.array) -> Tuple[bytes, int]:
    """
    Encode the estimator data for a subframe of data using FLAC LPC implementation.
    
    Parameters
    ----------
    subframe_data : np.array
        Subframe of data to encode.

    Returns
    -------
    Tuple[bytes, int]
        The encoded estimator bits and estimator method index (see FLAC_ENCODING_METHODS).
    """

    # assertions
    assert len(subframe_data) > 0, "Subframe data must not be empty."
    assert len(subframe_data.shape) == 1, "Subframe data must be 1D."
    assert subframe_data.dtype == np.int32, "Subframe data must be int32."

    # create temporary file
    with tempfile.NamedTemporaryFile(suffix = ".bin", delete = False) as tmp_file:
        subframe_data_filepath = tmp_file.name
        subframe_data.tofile(subframe_data_filepath)
        
    # try to encode subframe to get estimator bits
    try:

        # verify file was created successfully
        if not exists(subframe_data_filepath):
            raise RuntimeError(f"Failed to create temporary file: {subframe_data_filepath}")
        
        # encode subframe data to get estimator bits using script
        result = subprocess.run(
            args = ["python3", FLAC_LPC_ENCODE_SCRIPT_FILEPATH, subframe_data_filepath],
            check = True,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
        )
        
    # except exception, raise error
    except subprocess.CalledProcessError as e:
        stderr_text = e.stderr.decode("utf-8", errors = "ignore") if e.stderr else "No error message"
        raise RuntimeError(f"FLAC encoder failed (exit {e.returncode}): {stderr_text}")

    # at last, cleanup
    finally:
    
        # always cleanup, even if there was an error
        if exists(subframe_data_filepath):
            try:
                remove(subframe_data_filepath)
            except OSError:
                pass # ignore cleanup errors

    # parse result for encoded estimator data
    encoded_estimator_data = result.stdout

    # get estimator method index
    estimator_method_index = result.stderr.decode("utf-8", errors = "ignore")
    estimator_method_index = next(filter(lambda x: x.startswith("Estimator Method Index: "), estimator_method_index.split("\n")))
    estimator_method_index = int(estimator_method_index.split(":")[-1].strip())

    # check if estimator method index is valid
    if estimator_method_index < 0 or estimator_method_index >= len(FLAC_ENCODING_METHODS):
        raise ValueError(f"Invalid estimator method index: {estimator_method_index}")

    return encoded_estimator_data, estimator_method_index


def decode_estimator_data(encoded_estimator_data: bytes, n_samples: int) -> Tuple[np.array, int, int]:
    """
    Decode the estimator data for a subframe of data using FLAC LPC implementation.
    
    Parameters
    ----------
    encoded_estimator_data : bytes
        Encoded estimator data.
    n_samples : int
        Number of samples to decode.

    Returns
    -------
    Tuple[np.array, int, int]
        The decoded subframe data, estimator method index (see FLAC_ENCODING_METHODS), and number of warmup samples (non-zero for Fixed and LPC).
    """

    # create temporary file
    with tempfile.NamedTemporaryFile(suffix = ".bin", delete = False) as tmp_file:
        encoded_estimator_data_filepath = tmp_file.name
        tmp_file.write(encoded_estimator_data)
        
    # try to encode subframe to get estimator bits
    try:

        # verify file was created successfully
        if not exists(encoded_estimator_data_filepath):
            raise RuntimeError(f"Failed to create temporary file: {encoded_estimator_data_filepath}")
        
        # encode subframe data to get estimator bits using script
        result = subprocess.run(
            args = ["python3", FLAC_LPC_DECODE_SCRIPT_FILEPATH, encoded_estimator_data_filepath, str(n_samples)],
            check = True,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
        )
        
    # except exception, raise error
    except subprocess.CalledProcessError as e:
        stderr_text = e.stderr.decode("utf-8", errors = "ignore") if e.stderr else "No error message"
        raise RuntimeError(f"FLAC encoder failed (exit {e.returncode}): {stderr_text}")

    # at last, cleanup
    finally:
    
        # always cleanup, even if there was an error
        if exists(encoded_estimator_data_filepath):
            try:
                remove(encoded_estimator_data_filepath)
            except OSError:
                pass # ignore cleanup errors

    # parse result for subframe data
    subframe_data = np.frombuffer(result.stdout, dtype = np.int32)
    
    # get estimator method index and number of warmup samples
    stderr_text = result.stderr.decode("utf-8", errors = "ignore").split("\n")
    estimator_method_index = next(filter(lambda x: x.startswith("Estimator Method Index: "), stderr_text))
    estimator_method_index = int(estimator_method_index.split(":")[-1].strip())
    n_warmup_samples = next(filter(lambda x: x.startswith("Warmup samples length: "), stderr_text))
    n_warmup_samples = int(n_warmup_samples.split(":")[-1].strip())

    # check if estimator method index is valid
    if estimator_method_index < 0 or estimator_method_index >= len(FLAC_ENCODING_METHODS):
        raise ValueError(f"Invalid estimator method index: {estimator_method_index}")

    # assertions
    assert len(subframe_data) > 0, "Subframe data must not be empty."
    assert len(subframe_data.shape) == 1, "Subframe data must be 1D."
    assert n_warmup_samples >= 0 and n_warmup_samples <= len(subframe_data), "Number of warmup samples must be between 0 and the number of samples."
    assert subframe_data.dtype == np.int32, "Subframe data must be int32."

    return subframe_data, estimator_method_index, n_warmup_samples

    
def encode_subframe(subframe_data: np.array, entropy_coder: EntropyCoder) -> BOTTLENECK_SUBFRAME_TYPE:
    """
    Encode a single subframe of data using FLAC LPC.
    
    Parameters
    ----------
    subframe_data : np.array
        Subframe of data to encode.
    entropy_coder : EntropyCoder
        The entropy coder to use.
        
    Returns
    -------
    BOTTLENECK_SUBFRAME_TYPE
        Encoded subframe as (n_samples, encoded_estimator_data, encoded_residuals)
    """
    
    # check for empty arrays, this should not happen in normal usage
    if len(subframe_data) == 0:
        return 0, bytes(), bytes()
    
    # encode estimator data
    encoded_estimator_data, estimator_method_index = encode_estimator_data(subframe_data = subframe_data)

    # encode residuals for Fixed and LPC
    if FLAC_ENCODING_METHODS[estimator_method_index] in FLAC_ENCODING_METHODS_WITH_ENTROPY:
        approximate_subframe_data, reconstructed_estimator_method_index, n_warmup_samples = decode_estimator_data(encoded_estimator_data = encoded_estimator_data, n_samples = len(subframe_data)) # decode estimator data to get reconstructed subframe data and number of warmup samples
        assert reconstructed_estimator_method_index == estimator_method_index, "Reconstructed estimator method index must match encoded estimator method index."
        approximate_subframe_data = approximate_subframe_data[:len(subframe_data)] # truncate to original length (CRITICAL: FLAC decoder may return padded data)
        assert np.array_equal(subframe_data[:n_warmup_samples], approximate_subframe_data[:n_warmup_samples]), "Warmup samples must match."
        residuals = subframe_data[n_warmup_samples:] - approximate_subframe_data[n_warmup_samples:] # calculate residuals
        encoded_residuals = entropy_coder.encode(residuals)

    # no residuals for Verbatim and Constant
    else:
        encoded_residuals = bytes()
    
    # return bottleneck subframe
    return len(subframe_data), encoded_estimator_data, encoded_residuals


def decode_subframe(bottleneck_subframe: BOTTLENECK_SUBFRAME_TYPE, entropy_coder: EntropyCoder) -> np.array:
    """
    Decode a single subframe from FLAC LPC encoding.
    
    Parameters
    ----------
    bottleneck_subframe : BOTTLENECK_SUBFRAME_TYPE
        Encoded subframe as (n_samples, encoded_estimator_data, encoded_residuals)
    entropy_coder : EntropyCoder
        The entropy coder to use.
        
    Returns
    -------
    np.array
        Decoded subframe
    """

    # unpack bottleneck subframe
    n_samples, encoded_estimator_data, encoded_residuals = bottleneck_subframe

    # check for zero samples, this should not happen in normal usage
    if n_samples == 0:
        return np.array([], dtype = np.int32)

    # decode estimator data to get reconstructed subframe data and number of warmup samples
    subframe_data, estimator_method_index, n_warmup_samples = decode_estimator_data(encoded_estimator_data = encoded_estimator_data, n_samples = n_samples)
    
    # decode residuals for Fixed and LPC
    if FLAC_ENCODING_METHODS[estimator_method_index] in FLAC_ENCODING_METHODS_WITH_ENTROPY:
        num_residual_samples = n_samples - n_warmup_samples
        residuals = entropy_coder.decode(stream = encoded_residuals, num_samples = num_residual_samples)
        subframe_data = np.concatenate((subframe_data[:n_warmup_samples], subframe_data[n_warmup_samples:] + residuals), axis = 0)

    # truncate to the original length (CRITICAL: FLAC decoder may return padded data)
    subframe_data = subframe_data[:n_samples]

    # ensure output is int32
    subframe_data = subframe_data.astype(np.int32)

    # return subframe data
    return subframe_data


def encode_frame(frame_data: np.array, entropy_coder: EntropyCoder, interchannel_decorrelation: bool = INTERCHANNEL_DECORRELATION_DEFAULT) -> BOTTLENECK_FRAME_TYPE:
    """
    Encode a single frame of data using FLAC LPC.
    
    Parameters
    ----------
    frame_data : np.array
        Frame of data to encode. Shape: (n_samples,) for mono, (n_samples, 2) for stereo.
    entropy_coder : EntropyCoder
        The entropy coder to use.
    interchannel_decorrelation : bool, default = INTERCHANNEL_DECORRELATION_DEFAULT
        Whether to try different interchannel decorrelation schemes.
        
    Returns
    -------
    BOTTLENECK_FRAME_TYPE
        Encoded frame as (interchannel_decorrelation_scheme_index, list_of_subframes)
    """

    # handle mono case
    if len(frame_data.shape) == 1:
        bottleneck_subframe = encode_subframe(subframe_data = frame_data, entropy_coder = entropy_coder)
        return 0, [bottleneck_subframe]
    
    # handle stereo case
    left_channel = frame_data[:, 0]
    right_channel = frame_data[:, 1]
    
    # If interchannel decorrelation is off, just use left/right scheme
    if not interchannel_decorrelation:
        bottleneck_subframe1 = encode_subframe(subframe_data = left_channel, entropy_coder = entropy_coder)
        bottleneck_subframe2 = encode_subframe(subframe_data = right_channel, entropy_coder = entropy_coder)
        return 0, [bottleneck_subframe1, bottleneck_subframe2]
    
    # try all interchannel decorrelation schemes and pick the best
    best_size = float("inf")
    best_bottleneck_frame = (0, [])
    for interchannel_decorrelation_scheme_index, interchannel_decorrelation_scheme_func in enumerate(INTERCHANNEL_DECORRELATION_SCHEMES_MAP):

        # apply the scheme
        channel1_transformed, channel2_transformed = interchannel_decorrelation_scheme_func(left = left_channel, right = right_channel)
        
        # ensure proper data types
        channel1_transformed = np.array(channel1_transformed, dtype = np.int32)
        channel2_transformed = np.array(channel2_transformed, dtype = np.int32)
        
        # encode both subframes
        bottleneck_subframe1 = encode_subframe(subframe_data = channel1_transformed, entropy_coder = entropy_coder)
        bottleneck_subframe2 = encode_subframe(subframe_data = channel2_transformed, entropy_coder = entropy_coder)
        
        # create bottleneck frame
        bottleneck_frame = (interchannel_decorrelation_scheme_index, [bottleneck_subframe1, bottleneck_subframe2])
        
        # calculate size
        size = get_compressed_frame_size(bottleneck_frame = bottleneck_frame)
        
        # update best if this is better
        if size < best_size:
            best_size = size
            best_bottleneck_frame = bottleneck_frame
    
    return best_bottleneck_frame


def decode_frame(bottleneck_frame: BOTTLENECK_FRAME_TYPE, entropy_coder: EntropyCoder) -> np.array:
    """
    Decode a single frame from FLAC LPC encoding.
    
    Parameters
    ----------
    bottleneck_frame : BOTTLENECK_FRAME_TYPE
        Encoded frame as (interchannel_decorrelation_scheme_index, list_of_subframes)
    entropy_coder : EntropyCoder
        The entropy coder to use.
        
    Returns
    -------
    np.array
        Decoded frame
    """

    # unpack bottleneck frame
    interchannel_decorrelation_scheme_index, subframes = bottleneck_frame
    
    # handle mono case
    if len(subframes) == 1:
        mono_frame = decode_subframe(bottleneck_subframe = subframes[0], entropy_coder = entropy_coder)
        return mono_frame
    
    # handle stereo case
    channel1_decoded = decode_subframe(bottleneck_subframe = subframes[0], entropy_coder = entropy_coder)
    channel2_decoded = decode_subframe(bottleneck_subframe = subframes[1], entropy_coder = entropy_coder)
    
    # reverse the interchannel decorrelation
    reverse_func = REVERSE_INTERCHANNEL_DECORRELATION_SCHEMES_MAP[interchannel_decorrelation_scheme_index]
    left_channel, right_channel = reverse_func(channel1 = channel1_decoded, channel2 = channel2_decoded)
    
    # combine channels
    stereo_frame = np.stack((left_channel, right_channel), axis = -1)
    
    return stereo_frame


def get_compressed_subframe_size(bottleneck_subframe: BOTTLENECK_SUBFRAME_TYPE) -> int:
    """
    Get the size of a compressed subframe in bytes.

    Parameters
    ----------
    bottleneck_subframe : BOTTLENECK_SUBFRAME_TYPE
        The compressed subframe as (n_samples, encoded_estimator_data, encoded_residuals)
        
    Returns
    -------
    int
        The size of the compressed subframe in bytes
    """

    # unpack bottleneck subframe
    n_samples, encoded_estimator_data, encoded_residuals = bottleneck_subframe

    # add size for storing number of samples
    total_size = utils.MAXIMUM_BLOCK_SIZE_ASSUMPTION_BYTES

    # add size for encoded estimator data, do not need to add size for encoding method index since it is encoded into the estimator data
    total_size += len(encoded_estimator_data)

    # add size for encoded residuals
    total_size += len(encoded_residuals)

    return total_size


def get_compressed_frame_size(bottleneck_frame: BOTTLENECK_FRAME_TYPE) -> int:
    """
    Get the size of a compressed frame in bytes.
    
    Parameters
    ----------
    bottleneck_frame : BOTTLENECK_FRAME_TYPE
        The compressed frame as (interchannel_decorrelation_scheme_index, list_of_subframes)
        
    Returns
    -------
    int
        The size of the compressed frame in bytes
    """

    # initialize total size
    interchannel_decorrelation_scheme_index, subframes = bottleneck_frame
    total_size = 1 # we can store the interchannel decorrelation scheme index as one byte
    
    # add size for each subframe
    for bottleneck_subframe in subframes:
        total_size += get_compressed_subframe_size(bottleneck_subframe = bottleneck_subframe)
    
    return total_size


def encode_frame_worker(frame_data: np.array, entropy_coder: EntropyCoder, interchannel_decorrelation: bool) -> BOTTLENECK_FRAME_TYPE:
    """
    Worker function for multiprocessing frame encoding.
    """
    return encode_frame(frame_data = frame_data, entropy_coder = entropy_coder, interchannel_decorrelation = interchannel_decorrelation)


def decode_frame_worker(bottleneck_frame: BOTTLENECK_FRAME_TYPE, entropy_coder: EntropyCoder) -> np.array:
    """
    Worker function for multiprocessing frame decoding.
    """
    return decode_frame(bottleneck_frame = bottleneck_frame, entropy_coder = entropy_coder)

##################################################


# LOSSLESS COMPRESSOR INTERFACE
##################################################

class FlacLPC(LosslessCompressor):
    """
    FLAC LPC Compressor.
    """

    def __init__(self, entropy_coder: EntropyCoder, block_size: int = BLOCK_SIZE_DEFAULT, interchannel_decorrelation: bool = INTERCHANNEL_DECORRELATION_DEFAULT, jobs: int = JOBS_DEFAULT):
        """
        Initialize the FLAC LPC Compressor.

        Parameters
        ----------
        entropy_coder : EntropyCoder
            The entropy coder to use.
        block_size : int, default = BLOCK_SIZE_DEFAULT
            The block size to use for encoding.
        interchannel_decorrelation : bool, default = INTERCHANNEL_DECORRELATION_DEFAULT
            Whether to try different interchannel decorrelation schemes.
        jobs : int, default = JOBS_DEFAULT
            The number of jobs to use for multiprocessing.
        """
        self.entropy_coder = entropy_coder
        self.block_size = block_size
        assert self.block_size > 0 and self.block_size <= utils.MAXIMUM_BLOCK_SIZE_ASSUMPTION, f"Block size must be positive and less than or equal to {utils.MAXIMUM_BLOCK_SIZE_ASSUMPTION}."
        self.interchannel_decorrelation = interchannel_decorrelation
        self.jobs = jobs
        assert self.jobs > 0 and self.jobs <= multiprocessing.cpu_count(), f"Number of jobs must be positive and less than or equal to {multiprocessing.cpu_count()}."

        # check if the python LPC encoder and decoder scripts exist
        if not exists(FLAC_LPC_ENCODE_SCRIPT_FILEPATH):
            raise RuntimeError(f"FLAC LPC encoder script not found: {FLAC_LPC_ENCODE_SCRIPT_FILEPATH}")
        elif not exists(FLAC_LPC_DECODE_SCRIPT_FILEPATH):
            raise RuntimeError(f"FLAC LPC decoder script not found: {FLAC_LPC_DECODE_SCRIPT_FILEPATH}") 

    def encode(self, data: np.array) -> BOTTLENECK_TYPE:
        """
        Encode the original data into the bottleneck.

        Parameters
        ----------
        data : np.array
            The data to encode. Shape: (n_samples,) for mono, (n_samples, 2) for stereo.

        Returns
        -------
        BOTTLENECK_TYPE
            The bottleneck.
        """
        
        # ensure input is valid
        assert len(data.shape) == 1 or len(data.shape) == 2, "Data must be 1D or 2D."
        if len(data.shape) == 2:
            assert data.shape[1] == 2, "Data must be 2D with 2 channels."
        assert data.dtype == np.int32, "Data must be int32."
        
        # split data into frames
        frames = partition_data_into_frames(data = data, block_size = self.block_size)
        
        # use multiprocessing to encode frames in parallel
        with multiprocessing.Pool(processes = self.jobs) as pool:
            worker_func = functools.partial(
                encode_frame_worker,
                entropy_coder = self.entropy_coder,
                interchannel_decorrelation = self.interchannel_decorrelation,
            )
            bottleneck = pool.map(func = worker_func, iterable = frames)
        
        return bottleneck

    def decode(self, bottleneck: BOTTLENECK_TYPE) -> np.array:
        """
        Decode the bottleneck into the original data.

        Parameters
        ----------
        bottleneck : BOTTLENECK_TYPE
            The bottleneck to decode.

        Returns
        -------
        np.array
            The decoded original data.
        """

        # use multiprocessing to decode frames in parallel
        with multiprocessing.Pool(processes = self.jobs) as pool:
            worker_func = functools.partial(
                decode_frame_worker,
                entropy_coder = self.entropy_coder,
            )
            decoded_frames = pool.map(func = worker_func, iterable = bottleneck)
        
        # concatenate all decoded frames
        reconstructed_data = np.concatenate(decoded_frames, axis = 0)

        # ensure output is valid
        assert len(reconstructed_data.shape) == 1 or len(reconstructed_data.shape) == 2, "Reconstructed data must be 1D or 2D."
        if len(reconstructed_data.shape) == 2:
            assert reconstructed_data.shape[1] == 2, "Reconstructed data must be 2D with 2 channels."
        assert reconstructed_data.dtype == np.int32, "Reconstructed data must be int32."
        
        return reconstructed_data

    def get_compressed_size(self, bottleneck: BOTTLENECK_TYPE) -> int:
        """
        Get the size of the bottleneck in bytes.

        Parameters
        ----------
        bottleneck : BOTTLENECK_TYPE
            The bottleneck.

        Returns 
        -------
        int
            The size of the bottleneck in bytes.
        """

        # use multiprocessing to get compressed frame sizes in parallel
        with multiprocessing.Pool(processes = self.jobs) as pool:
            compressed_frame_sizes = pool.map(func = get_compressed_frame_size, iterable = bottleneck)

        # calculate total size as the sum of the compressed frame sizes
        total_size = sum(compressed_frame_sizes)

        return total_size
        
##################################################
