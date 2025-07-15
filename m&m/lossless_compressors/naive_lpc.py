# README
# Phillip Long
# July 12, 2025

# Naive LPC Compressor.

# IMPORTS
##################################################

import numpy as np
from typing import List, Tuple
import scipy.signal
import warnings
import multiprocessing
import functools

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, f"{dirname(dirname(realpath(__file__)))}/entropy_coders")
sys.path.insert(0, dirname(dirname(dirname(realpath(__file__)))))

from lossless_compressors import LosslessCompressor, partition_data_into_frames, INTERCHANNEL_DECORRELATION_DEFAULT, INTERCHANNEL_DECORRELATION_SCHEMES_MAP, REVERSE_INTERCHANNEL_DECORRELATION_SCHEMES_MAP, JOBS_DEFAULT
from entropy_coders import EntropyCoder
import utils

##################################################


# CONSTANTS
##################################################

ORDER_DEFAULT = 9 # default LPC order
LPC_COEFFICIENTS_DTYPE = np.int8 # default LPC coefficients data type for quantization

##################################################


# BOTTLENECK TYPE
##################################################

# type of bottleneck subframes
BOTTLENECK_SUBFRAME_TYPE = Tuple[int, np.array, np.array, bytes] # bottleneck subframe type is a tuple of the number of samples, warmup samples, LPC coefficients, and encoded residuals

# type of bottleneck frames
BOTTLENECK_FRAME_TYPE = Tuple[int, List[BOTTLENECK_SUBFRAME_TYPE]] # bottleneck frame type is a tuple of the interchannel decorrelation scheme index and list of subframes

# type of bottleneck
BOTTLENECK_TYPE = List[BOTTLENECK_FRAME_TYPE]

##################################################


# HELPER FUNCTIONS FOR LPC
##################################################

def levinson_durbin(r: np.array, order: int) -> Tuple[np.array, float, np.array]:
    """
    Levinson-Durbin recursion to solve Toeplitz systems for linear predictive coding.
    
    Parameters
    ----------
    r : np.array
        autocorrelation sequence of length >= order + 1
    order : int
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


def lpc_autocorrelation_method(y: np.array, order: int) -> np.array:
    """
    Compute linear prediction coefficients using autocorrelation method for guaranteed stability.
    """
    y = y.astype(np.float64)
    r = np.correlate(a = y, v = y, mode = "full")
    r = r[(len(y) - 1):(len(y) - 1 + order + 1)] # take autocorrelations from lag 0 to lag order
    a, e, k = levinson_durbin(r = r, order = order)
    return a


def lpc_predict_samples(lpc_coefficients: np.array, warmup_samples: np.array, n_predicted_samples: int) -> np.array:
    """
    Predict samples using LPC coefficients and warmup samples.
    """
    order = len(lpc_coefficients) # infer order from length of LPC coefficients
    predicted_samples = np.zeros(shape = n_predicted_samples, dtype = np.float32) # initialize predicted samples
    for i in range(n_predicted_samples):
        if i < order: # use some warmup samples and some predicted samples
            previous_samples = np.concatenate((warmup_samples[-(order - i):], predicted_samples[:i]), axis = 0)
        else: # use only previous predicted samples
            previous_samples = predicted_samples[(i - order):i]
        predicted_samples[i] = np.dot(lpc_coefficients, previous_samples[::-1])
    return predicted_samples

##################################################


# NAIVE LPC ESTIMATOR FUNCTIONS
##################################################

def encode_subframe(subframe_data: np.array, entropy_coder: EntropyCoder, order: int = ORDER_DEFAULT) -> BOTTLENECK_SUBFRAME_TYPE:
    """
    Encode a single subframe of data using LPC.
    
    Parameters
    ----------
    subframe_data : np.array
        Subframe of data to encode.
    entropy_coder : EntropyCoder
        The entropy coder to use.
    order : int, default = ORDER_DEFAULT
        The LPC order to use for encoding.
        
    Returns
    -------
    BOTTLENECK_SUBFRAME_TYPE
        Encoded subframe as (n_samples, warmup_samples, lpc_coefficients, encoded_residuals)
    """

    # handle short subframes that are shorter than the LPC order
    if len(subframe_data) <= order: # for very short subframes, store entire subframe as warmup with no prediction
        warmup_samples = subframe_data.copy()
        lpc_coefficients = np.array([], dtype = np.float32)
        encoded_residuals = bytes()
        return len(subframe_data), warmup_samples, lpc_coefficients, encoded_residuals

    # convert subframe to float for LPC coefficient computation
    subframe_float = subframe_data.astype(np.float32)

    # compute LPC coefficients using autocorrelation method
    lpc_coefficients = lpc_autocorrelation_method(y = subframe_float, order = order)
    if not np.all(np.abs(np.roots(lpc_coefficients)) < 1):
        warnings.warn(message = "Linear prediction coefficients are unstable!", category = RuntimeWarning)
    lpc_coefficients = lpc_coefficients[1:] # remove first coefficient which is 1.0
    lpc_coefficients = np.round(lpc_coefficients).astype(LPC_COEFFICIENTS_DTYPE) # quantize
    
    # store first `order` samples as warmup samples for LPC
    warmup_samples = subframe_data[:order].copy()
    
    # predict samples from index `order` onwards using previous reconstructed samples
    samples_to_predict = subframe_data[order:]
    n_predicted_samples = len(samples_to_predict)
    predicted_samples = lpc_predict_samples(lpc_coefficients = lpc_coefficients, warmup_samples = warmup_samples, n_predicted_samples = n_predicted_samples)
    predicted_samples = np.round(predicted_samples).astype(subframe_data.dtype) # quantize predicted samples
    
    # encode residuals
    residuals = samples_to_predict - predicted_samples
    encoded_residuals = entropy_coder.encode(nums = residuals)
    
    return len(subframe_data), warmup_samples, lpc_coefficients, encoded_residuals


def decode_subframe(bottleneck_subframe: BOTTLENECK_SUBFRAME_TYPE, entropy_coder: EntropyCoder) -> np.array:
    """
    Decode a single subframe from LPC encoding.
    
    Parameters
    ----------
    bottleneck_subframe : BOTTLENECK_SUBFRAME_TYPE
        Encoded subframe as (n_samples, warmup_samples, lpc_coefficients, encoded_residuals)
    entropy_coder : EntropyCoder
        The entropy coder to use.
        
    Returns
    -------
    np.array
        Decoded subframe
    """

    # unpack bottleneck subframe
    n_samples, warmup_samples, lpc_coefficients, encoded_residuals = bottleneck_subframe

    # handle case where entire subframe was stored as warmup (short subframes with length less than or equal to the LPC order)
    if n_samples == len(warmup_samples):
        return warmup_samples.copy()
    
    # decode residuals
    n_predicted_samples = n_samples - len(warmup_samples)
    residuals = entropy_coder.decode(stream = encoded_residuals, num_samples = n_predicted_samples)

    # predict samples from index `order` onwards using previous reconstructed samples
    predicted_samples = lpc_predict_samples(lpc_coefficients = lpc_coefficients, warmup_samples = warmup_samples, n_predicted_samples = n_predicted_samples)
    predicted_samples = np.round(predicted_samples).astype(np.int32) # quantize predicted samples

    # reconstruct subframe
    reconstructed_subframe = np.concatenate((warmup_samples, predicted_samples + residuals), axis = 0)
    
    return reconstructed_subframe


def encode_frame(frame_data: np.array, entropy_coder: EntropyCoder, order: int = ORDER_DEFAULT, interchannel_decorrelation: bool = INTERCHANNEL_DECORRELATION_DEFAULT) -> BOTTLENECK_FRAME_TYPE:
    """
    Encode a single frame of data using LPC.
    
    Parameters
    ----------
    frame_data : np.array
        Frame of data to encode. Shape: (n_samples,) for mono, (n_samples, 2) for stereo.
    entropy_coder : EntropyCoder
        The entropy coder to use.
    order : int, default = ORDER_DEFAULT
        The LPC order to use for encoding.
    interchannel_decorrelation : bool, default = INTERCHANNEL_DECORRELATION_DEFAULT
        Whether to try different interchannel decorrelation schemes.
        
    Returns
    -------
    BOTTLENECK_FRAME_TYPE
        Encoded frame as (interchannel_decorrelation_scheme_index, list_of_subframes)
    """

    # handle mono case
    if len(frame_data.shape) == 1:
        bottleneck_subframe = encode_subframe(subframe_data = frame_data, entropy_coder = entropy_coder, order = order)
        return 0, [bottleneck_subframe]
    
    # handle stereo case
    left_channel = frame_data[:, 0]
    right_channel = frame_data[:, 1]
    
    # If interchannel decorrelation is off, just use right/left scheme
    if not interchannel_decorrelation:
        bottleneck_subframe1 = encode_subframe(subframe_data = left_channel, entropy_coder = entropy_coder, order = order)
        bottleneck_subframe2 = encode_subframe(subframe_data = right_channel, entropy_coder = entropy_coder, order = order)
        return 0, [bottleneck_subframe1, bottleneck_subframe2]
    
    # try all interchannel decorrelation schemes and pick the best
    best_interchannel_decorrelation_scheme_index = 0
    best_size = float("inf")
    best_bottleneck_frame = None
    for interchannel_decorrelation_scheme_index, interchannel_decorrelation_scheme_func in enumerate(INTERCHANNEL_DECORRELATION_SCHEMES_MAP):

        # apply the scheme
        channel1_transformed, channel2_transformed = interchannel_decorrelation_scheme_func(left = left_channel, right = right_channel)
        
        # encode both subframes
        bottleneck_subframe1 = encode_subframe(subframe_data = channel1_transformed, entropy_coder = entropy_coder, order = order)
        bottleneck_subframe2 = encode_subframe(subframe_data = channel2_transformed, entropy_coder = entropy_coder, order = order)
        
        # create bottleneck frame
        bottleneck_frame = (interchannel_decorrelation_scheme_index, [bottleneck_subframe1, bottleneck_subframe2])
        
        # calculate size
        size = get_compressed_frame_size(bottleneck_frame = bottleneck_frame)
        
        # update best if this is better
        if size < best_size:
            best_interchannel_decorrelation_scheme_index = interchannel_decorrelation_scheme_index
            best_size = size
            best_bottleneck_frame = bottleneck_frame
    
    return best_bottleneck_frame


def decode_frame(bottleneck_frame: BOTTLENECK_FRAME_TYPE, entropy_coder: EntropyCoder) -> np.array:
    """
    Decode a single frame from LPC encoding.
    
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
        The compressed subframe as (n_samples, warmup_samples, lpc_coefficients, encoded_residuals)
        
    Returns
    -------
    int
        The size of the compressed subframe in bytes
    """

    # unpack bottleneck subframe
    n_samples, warmup_samples, lpc_coefficients, encoded_residuals = bottleneck_subframe

    # add size for storing number of samples
    total_size = utils.MAXIMUM_BLOCK_SIZE_ASSUMPTION_BYTES

    # add size for warmup samples
    total_size += warmup_samples.nbytes

    # add size for LPC coefficients
    total_size += lpc_coefficients.nbytes

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


def encode_frame_worker(frame_data: np.array, entropy_coder: EntropyCoder, order: int, interchannel_decorrelation: bool) -> BOTTLENECK_FRAME_TYPE:
    """
    Worker function for multiprocessing frame encoding.
    """
    return encode_frame(frame_data = frame_data, entropy_coder = entropy_coder, order = order, interchannel_decorrelation = interchannel_decorrelation)


def decode_frame_worker(bottleneck_frame: BOTTLENECK_FRAME_TYPE, entropy_coder: EntropyCoder) -> np.array:
    """
    Worker function for multiprocessing frame decoding.
    """
    return decode_frame(bottleneck_frame = bottleneck_frame, entropy_coder = entropy_coder)

##################################################


# LOSSLESS COMPRESSOR INTERFACE
##################################################

class NaiveLPC(LosslessCompressor):
    """
    Naive LPC Compressor.
    """

    def __init__(self, entropy_coder: EntropyCoder, order: int = ORDER_DEFAULT, interchannel_decorrelation: bool = INTERCHANNEL_DECORRELATION_DEFAULT, jobs: int = JOBS_DEFAULT):
        """
        Initialize the Naive LPC Compressor.

        Parameters
        ----------
        entropy_coder : EntropyCoder
            The entropy coder to use.
        order : int, default = ORDER_DEFAULT
            The LPC order to use for encoding, defaults to ORDER_DEFAULT.
        interchannel_decorrelation : bool, default = INTERCHANNEL_DECORRELATION_DEFAULT
            Whether to decorrelate channels.
        jobs : int, default = JOBS_DEFAULT
            The number of jobs to use for multiprocessing.
        """
        self.entropy_coder = entropy_coder
        self.order = order
        assert self.order > 0, "LPC order must be positive."
        self.interchannel_decorrelation = interchannel_decorrelation
        self.jobs = jobs
        
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
                order = self.order,
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