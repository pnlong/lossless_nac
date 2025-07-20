# README
# Phillip Long
# July 6, 2025

# FLAC Entropy Coder.

# IMPORTS
##################################################

import numpy as np
import subprocess
import tempfile
from os import remove
from os.path import dirname, realpath, exists
from typing import List

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(dirname(realpath(__file__)))))

from entropy_coders import EntropyCoder
import utils

# Import the encoder and decoder modules directly
try:
    from flac_rice_helpers.flac_rice_encode import encode_residuals as flac_encode_residuals
    from flac_rice_helpers.flac_rice_encode import decode_residuals as flac_decode_residuals
    DIRECT_IMPORT_AVAILABLE = True
except ImportError:
    DIRECT_IMPORT_AVAILABLE = False

##################################################


# CONSTANTS
##################################################

FLAC_RICE_HELPERS_DIR = f"{dirname(realpath(__file__))}/flac_rice_helpers" # directory that contains the FLAC Rice encode and decode scripts
FLAC_RICE_ENCODE_SCRIPT_FILEPATH = f"{FLAC_RICE_HELPERS_DIR}/flac_rice_encode.py" # filepath to FLAC Rice encode script
FLAC_RICE_DECODE_SCRIPT_FILEPATH = f"{FLAC_RICE_HELPERS_DIR}/flac_rice_decode.py" # filepath to FLAC Rice decode script

##################################################


# MAIN FLAC ENTROPY CODING FUNCTIONS
##################################################

def encode(nums: np.array) -> bytes:
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

    # ensure nums is a numpy array of correct data type
    nums = np.array(nums, dtype = np.int32)
    
    # check for empty arrays - this should not happen in normal usage
    if len(nums) == 0:
        return bytes()
    
    # Use direct import if available (much faster - single subprocess instead of double)
    if DIRECT_IMPORT_AVAILABLE:
        try:
            return flac_encode_residuals(nums.tolist())
        except Exception as e:
            # Fall back to subprocess method if direct import fails
            pass
    
    # Fallback to original subprocess method
    # use individual temporary file instead of shared directory (more robust for multiprocessing)
    with tempfile.NamedTemporaryFile(suffix = ".bin", delete = False) as tmp_file:
        nums_filepath = tmp_file.name
        nums.tofile(nums_filepath)
    
    # try to encode nums to stream
    try:

        # verify file was created successfully
        if not exists(nums_filepath):
            raise RuntimeError(f"Failed to create temporary file: {nums_filepath}")

        # encode nums to stream using script
        result = subprocess.run(
            args = ["python3", FLAC_RICE_ENCODE_SCRIPT_FILEPATH, nums_filepath],
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
        if exists(nums_filepath):
            try:
                remove(nums_filepath)
            except OSError:
                pass # ignore cleanup errors

    # get stream from result
    stream = result.stdout

    # return stream
    return stream

def decode(stream: bytes, num_samples: int) -> np.array:
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
    
    # check for zero samples - this should not happen in normal usage
    if num_samples == 0:
        return np.array([], dtype = np.int32)
    
    # Use direct import if available (much faster - no subprocess for Python script)
    if DIRECT_IMPORT_AVAILABLE:
        try:
            decoded_list = flac_decode_residuals(stream, num_samples)
            return np.array(decoded_list, dtype=np.int32)
        except Exception as e:
            # Fall back to subprocess method if direct import fails
            pass
    
    # Fallback to original subprocess method
    # use individual temporary file instead of shared directory (more robust for multiprocessing)
    with tempfile.NamedTemporaryFile(suffix = ".bin", delete = False) as tmp_file:
        stream_filepath = tmp_file.name
        tmp_file.write(stream)
    
    # try to decode stream to nums
    try:

        # verify file was created successfully
        if not exists(stream_filepath):
            raise RuntimeError(f"Failed to create temporary stream file: {stream_filepath}")

        # decode stream to nums
        result = subprocess.run(
            args = ["python3", FLAC_RICE_DECODE_SCRIPT_FILEPATH, stream_filepath, str(num_samples)],
            check = True,
            stdout = subprocess.PIPE,
            stderr = subprocess.PIPE,
        )
        nums = np.frombuffer(result.stdout, dtype = np.int32)
        
    # except exception, raise error
    except subprocess.CalledProcessError as e:
        stderr_text = e.stderr.decode("utf-8", errors = "ignore") if e.stderr else "No error message"
        raise RuntimeError(f"FLAC decoder failed (exit {e.returncode}): {stderr_text}")

    # at last, cleanup
    finally:
        # always cleanup, even if there was an error
        if exists(stream_filepath):
            try:
                remove(stream_filepath)
            except OSError:
                pass # ignore cleanup errors

    # return nums
    return nums

def batch_encode(nums_list: List[np.array]) -> List[bytes]:
    """
    Encode multiple data arrays with reduced Python overhead.
    
    Uses direct function calls when possible to eliminate subprocess overhead.
    
    Parameters
    ----------
    nums_list : List[np.array]
        List of data arrays to encode.
        
    Returns
    -------
    List[bytes]
        List of encoded data corresponding to each input array.
    """
    
    if not nums_list:
        return []
    
    results = []
    
    # Use direct import if available (much faster)
    if DIRECT_IMPORT_AVAILABLE:
        for nums in nums_list:
            if len(nums) == 0:
                results.append(bytes())
            else:
                try:
                    # Convert numpy array to list and encode directly
                    encoded = flac_encode_residuals(nums.astype(np.int32).tolist())
                    results.append(encoded)
                except Exception as e:
                    # Fall back to subprocess method for this array
                    results.append(encode(nums))
    else:
        # Fallback: batch the individual encodes with optimizations
        # Pre-compile helper once if needed (instead of checking each time)
        first_valid_nums = next((nums for nums in nums_list if len(nums) > 0), None)
        if first_valid_nums is not None:
            # Trigger any one-time compilation by encoding a small array
            try:
                _ = encode(np.array([0], dtype=np.int32))
            except:
                pass  # Ignore compilation errors for now
        
        # Now encode all arrays (helper should be compiled)
        for nums in nums_list:
            if len(nums) == 0:
                results.append(bytes())
            else:
                results.append(encode(nums))
    
    return results

##################################################


# ENTROPY CODER INTERFACE
##################################################

class FlacRiceCoder(EntropyCoder):
    """
    FLAC Rice Coder.
    """

    def __init__(self):
        """
        Initialize the FLAC Rice Coder.
        """
        # check if the python entropy encoder and decoder scripts exist
        if not exists(FLAC_RICE_ENCODE_SCRIPT_FILEPATH):
            raise RuntimeError(f"FLAC entropy encoder script not found: {FLAC_RICE_ENCODE_SCRIPT_FILEPATH}")
        elif not exists(FLAC_RICE_DECODE_SCRIPT_FILEPATH):
            raise RuntimeError(f"FLAC entropy decoder script not found: {FLAC_RICE_DECODE_SCRIPT_FILEPATH}") 

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
        )

    def batch_encode(self, nums_list: List[np.array]) -> List[bytes]:
        """
        Encode multiple data arrays with reduced Python overhead.
        
        Parameters
        ----------
        nums_list : List[np.array]
            List of data arrays to encode.
            
        Returns
        -------
        List[bytes]
            List of encoded data corresponding to each input array.
        """
        return batch_encode(nums_list)
        
##################################################