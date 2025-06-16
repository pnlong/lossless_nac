# README
# Phillip Long
# June 15, 2025

# Log information about residuals using different lossy estimators.

# IMPORTS
##################################################

import numpy as np
import pandas as pd
from typing import Dict, Any
from os.path import exists, basename
from os import mkdir, makedirs
import hashlib

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))

import utils
import rice

##################################################


# CONSTANTS
##################################################

# read in already logged output filepaths
ALREADY_LOGGED_OUTPUT_FILEPATHS = set(pd.read_csv(filepath_or_buffer = utils.LOGGING_FOR_ZACH_FILEPATH, sep = ",", header = 0, index_col = False, usecols = ["residuals_path"])["residuals_path"]) if exists(utils.LOGGING_FOR_ZACH_FILEPATH) else set()

##################################################


# FUNCTION FOR WRITING DATA
##################################################

def log_for_zach(
        residuals: np.array, # the residuals between the original and approximated waveform
        residuals_rice: bytes, # the residuals post-rice-encoding as a bytes object
        duration: float, # duration of original waveform
        lossy_estimator: str, # the lossy estimator used (LPC, DAC, etc.)
        parameters: Dict[str, Any], # parameters for lossy estimator
        path: str = None, # path to original waveform numpy pickle file, and if not supplied, then no statistics are logged
    ):
    """
    Log data on residuals for different lossy estimators.
    """

    # reference ALREADY_LOGGED_OUTPUT_FILEPATHS
    global ALREADY_LOGGED_OUTPUT_FILEPATHS

    # create output directory if doesn't exist already
    if not exists(utils.LOGGING_FOR_ZACH_DIR):
        makedirs(utils.LOGGING_FOR_ZACH_DIR)

    # write column names if they don't already exist
    if not exists(utils.LOGGING_FOR_ZACH_FILEPATH):
        pd.DataFrame(
            columns = utils.LOGGING_FOR_ZACH_COLUMN_NAMES,
        ).to_csv(
            path_or_buf = utils.LOGGING_FOR_ZACH_FILEPATH, sep = ",", na_rep = utils.NA_STRING, header = True, index = False, mode = "w",
        )

    # run logging in the non-default case
    if path is not None:

        # determine parameters string
        parameters_string = "-".join((f"{parameter}:{value}" for parameter, value in parameters.items()))
        parameters_string_hash = hashlib.sha256(parameters_string.encode("utf-8")).hexdigest() # get hash of parameters string

        # create output directory
        output_dir = f"{utils.LOGGING_FOR_ZACH_DIR}/{lossy_estimator}_{parameters_string_hash}"
        if not exists(output_dir):
            mkdir(output_dir)

        # write residuals to output filepath
        output_filepath = f"{output_dir}/{basename(path)}" # residuals numpy pickle object filepath
        if output_filepath in ALREADY_LOGGED_OUTPUT_FILEPATHS:
            return # stop execution here if the output filepath has already been logged
        np.save(file = output_filepath, arr = residuals)

        # write data
        pd.DataFrame(data = [{
            "lossy_estimator": lossy_estimator,
            "parameters": parameters_string,
            "parameters_hash": parameters_string_hash,
            "residuals_path": output_filepath,
            "original_path": path,
            "reconstruction_error": np.mean(residuals), # the average residual
            "compression_ratio": len(residuals_rice) / residuals.nbytes, # number of bytes for rice-coded residuals divided by number of bytes for numpy residuals
            "bits_per_second": (len(residuals_rice) * 8) / duration, # bits per second
            "rice_parameter": rice.K,
        }], columns = utils.LOGGING_FOR_ZACH_COLUMN_NAMES).to_csv(
            path_or_buf = utils.LOGGING_FOR_ZACH_FILEPATH, sep = ",", na_rep = utils.NA_STRING, header = False, index = False, mode = "a",
        )
        ALREADY_LOGGED_OUTPUT_FILEPATHS.add(output_filepath)

    # return nothing
    return

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # WRITE COLUMN NAMES
    ##################################################

    # write column names by passing certain arguments
    log_for_zach(
        residuals = None, residuals_rice = None, duration = None, lossy_estimator = None, parameters = None,
        path = None, # will just write the column names
    )

    ##################################################

##################################################