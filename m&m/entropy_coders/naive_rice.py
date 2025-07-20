# README
# Phillip Long
# July 6, 2025

# Naive Rice Coder using C helpers for performance.

# IMPORTS
##################################################

import numpy as np
import subprocess
import tempfile
from os import remove
from os.path import dirname, realpath, exists

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(dirname(realpath(__file__)))))

from entropy_coders import EntropyCoder
import utils

##################################################

# CONSTANTS
##################################################

K_DEFAULT = 1 # default rice parameter
NAIVE_RICE_HELPERS_DIR = f"{dirname(realpath(__file__))}/naive_rice_helpers" # directory that contains the naive rice helpers
NAIVE_RICE_ENCODE_SCRIPT_FILEPATH = f"{NAIVE_RICE_HELPERS_DIR}/naive_rice_encode.py" # filepath to naive rice encode script
NAIVE_RICE_DECODE_SCRIPT_FILEPATH = f"{NAIVE_RICE_HELPERS_DIR}/naive_rice_decode.py" # filepath to naive rice decode script

##################################################

# NAIVE RICE ENTROPY CODING FUNCTIONS
##################################################

def encode(nums: np.array, k: int = K_DEFAULT) -> bytes:
    """
    Encode the data using C helper for performance.

    Parameters
    ----------
    nums : np.array
        The data to encode.
    k : int, default = K_DEFAULT
        The Rice parameter k, defaults to K_DEFAULT.

    Returns
    -------
    bytes
        The encoded data.
    """

    # ensure nums is a numpy array of correct data type
    nums = np.array(nums, dtype = np.int32)
    
    # check for empty arrays
    if len(nums) == 0:
        return bytes()
    
    # use individual temporary file for multiprocessing safety
    with tempfile.NamedTemporaryFile(suffix = ".bin", delete = False) as tmp_file:
        nums_filepath = tmp_file.name
        nums.tofile(nums_filepath)
    
    # try to encode nums to stream
    try:

        # verify file was created successfully
        if not exists(nums_filepath):
            raise RuntimeError(f"Failed to create temporary file: {nums_filepath}")

        # encode nums to stream using C helper script
        result = subprocess.run(
            args = ["python3", NAIVE_RICE_ENCODE_SCRIPT_FILEPATH, nums_filepath, str(k)],
            check = True,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
        )
        
    # except exception, raise error
    except subprocess.CalledProcessError as e:
        stderr_text = e.stderr.decode("utf-8", errors = "ignore") if e.stderr else "No error message"
        raise RuntimeError(f"Naive Rice encoder failed (exit {e.returncode}): {stderr_text}")
    
    # always cleanup, even if there was an error
    finally:
        if exists(nums_filepath):
            try:
                remove(nums_filepath)
            except OSError:
                pass # ignore cleanup errors

    # get stream from result
    stream = result.stdout

    # return stream
    return stream

def decode(stream: bytes, num_samples: int, k: int = K_DEFAULT) -> np.array:
    """
    Decode the data using C helper for performance.

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
    np.array
        The decoded data.
    """
    
    # check for zero samples
    if num_samples == 0:
        return np.array([], dtype = np.int32)
    
    # use individual temporary file for multiprocessing safety
    with tempfile.NamedTemporaryFile(suffix = ".bin", delete = False) as tmp_file:
        stream_filepath = tmp_file.name
        tmp_file.write(stream)
    
    # try to decode stream to nums
    try:

        # verify file was created successfully
        if not exists(stream_filepath):
            raise RuntimeError(f"Failed to create temporary stream file: {stream_filepath}")

        # decode stream to nums using C helper script
        result = subprocess.run(
            args = ["python3", NAIVE_RICE_DECODE_SCRIPT_FILEPATH, stream_filepath, str(num_samples), str(k)],
            check = True,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
        )
        nums = np.frombuffer(result.stdout, dtype = np.int32)
        
    # except exception, raise error
    except subprocess.CalledProcessError as e:
        stderr_text = e.stderr.decode("utf-8", errors = "ignore") if e.stderr else "No error message"
        raise RuntimeError(f"Naive Rice decoder failed (exit {e.returncode}): {stderr_text}")

    # always cleanup, even if there was an error
    finally: 
        if exists(stream_filepath):
            try:
                remove(stream_filepath)
            except OSError:
                pass # ignore cleanup errors

    # return nums
    return nums

##################################################

# ENTROPY CODER INTERFACE
##################################################

class NaiveRiceCoder(EntropyCoder):
    """
    Naive Rice Coder using C helpers for performance.
    """

    def __init__(self, k: int = K_DEFAULT):
        """
        Initialize the Naive Rice Coder.

        Parameters
        ----------
        k : int, default = K_DEFAULT
            The Rice parameter k, defaults to K_DEFAULT.
        """
        self.k = k
        assert self.k > 0, "Rice parameter k must be positive."
        
        # check if the helper scripts exist
        if not exists(NAIVE_RICE_ENCODE_SCRIPT_FILEPATH):
            raise RuntimeError(f"Naive Rice encoder script not found: {NAIVE_RICE_ENCODE_SCRIPT_FILEPATH}")
        elif not exists(NAIVE_RICE_DECODE_SCRIPT_FILEPATH):
            raise RuntimeError(f"Naive Rice decoder script not found: {NAIVE_RICE_DECODE_SCRIPT_FILEPATH}")
    
    def encode(self, nums: np.array) -> bytes:
        """
        Encode the data.

        Parameters
        ----------
        nums : np.array
            The data to encode.

        Returns
        -------
        bytes
            The encoded data.
        """
        return encode(
            nums = nums,
            k = self.k,
        )

    def decode(self, stream: bytes, num_samples: int) -> np.array:
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
        np.array
            The decoded data.
        """
        return decode(
            stream = stream,
            num_samples = num_samples,
            k = self.k,
        )

##################################################