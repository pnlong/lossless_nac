# README
# Phillip Long
# August 16, 2025

# Convert MUSDB18 NPY files to WAV files.

# IMPORTS
##################################################

import pandas as pd
from os.path import exists
from os import makedirs
from shutil import rmtree
import argparse
import multiprocessing
import numpy as np
import scipy.io.wavfile
from tqdm import tqdm
from math import log10, ceil

##################################################


# CONSTANTS
##################################################

# filepaths
INPUT_FILEPATH = "/deepfreeze/pnlong/lnac/test_data/musdb18_preprocessed-44100/data.csv"
OUTPUT_DIR = "/deepfreeze/pnlong/lnac/sashimi/data/musdb18"

##################################################


# FUNCTIONS
##################################################

def convert_npy_to_wav(
    input_path: str,
    output_path: str,
    sample_rate: int,
):
    """
    Convert a .npy file to a WAV file.

    Parameters
    ----------
    input_path : str
        Path to the .npy file.
    output_path : str
        Path to the output WAV file.
    sample_rate : int
        Sample rate of the WAV file.

    Returns
    -------
    None
    """

    # read in NPY file
    data = np.load(input_path)

    # write NPY file as a WAV file
    scipy.io.wavfile.write(filename = output_path, rate = sample_rate, data = data)

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # SETUP
    ##################################################

    # parse arguments
    def parse_args(args = None, namespace = None):
        """
        Parse command-line arguments for converting residuals to WAV files.
        
        Parameters
        ----------
        args : list, optional
            List of argument strings to parse, by default None (uses sys.argv)
        namespace : argparse.Namespace, optional
            Namespace object to populate with parsed arguments, by default None
            
        Returns
        -------
        argparse.Namespace
            Parsed arguments containing paths and options for expression text extraction
            
        Raises
        ------
        FileNotFoundError
            If the specified PDMX file does not exist
        """
        parser = argparse.ArgumentParser(prog = "Convert Residuals to WAV Files", description = "Convert residuals to WAV files.") # create argument parser
        parser.add_argument("--input_filepath", type = str, default = INPUT_FILEPATH, help = "Path to input file.")
        parser.add_argument("--output_dir", type = str, default = OUTPUT_DIR, help = "Path to output directory.")
        parser.add_argument("--jobs", type = int, default = int(multiprocessing.cpu_count() / 4), help = "Number of jobs to run in parallel.")
        parser.add_argument("--reset", action = "store_true", help = "Reset the output directory.")
        args = parser.parse_args(args = args, namespace = namespace) # parse arguments
        if not exists(args.input_filepath):
            raise FileNotFoundError(f"Input file not found: {args.input_filepath}")
        return args # return parsed arguments
    args = parse_args()

    # read in input csv
    print("Reading in input data...")
    dataset = pd.read_csv(filepath_or_buffer = args.input_filepath, sep = ",", header = 0, index_col = False)
    dataset = dataset[["path", "sample_rate"]] # get only the path and sample rate columns, as that is all we care about
    assert len(dataset) > 0, "Dataset is empty."
    fixed_width = ceil(log10(len(dataset)))
    print(f"Completed reading in input data.")

    # create output directory
    print("Creating output directory...")
    if exists(args.output_dir) and args.reset:
        rmtree(args.output_dir)
    if not exists(args.output_dir):
        makedirs(args.output_dir, exist_ok = True)
    print("Created output directory.")

    ##################################################


    # POPULATE OUTPUT SUBDIRECTORIES
    ##################################################

    # convert files
    def convert_helper(i: int):
        """
        Use the information in a row of the dataset to convert a residual to a WAV file, as well as the original input file.

        Parameters
        ----------
        i : int
            Index of the row of the dataset to convert.

        Returns
        -------
        None
        """

        # get row
        path, sample_rate = dataset.loc[i]
        output_path = f"{args.output_dir}/out{i:0{fixed_width}}.wav"
        
        # convert input to WAV file
        convert_npy_to_wav(input_path = path, output_path = output_path, sample_rate = sample_rate)

    # use multiprocessing
    with multiprocessing.Pool(processes = args.jobs) as pool:
        _ = list(tqdm(iterable = pool.imap_unordered(
            func = convert_helper,
            iterable = dataset.index,
            chunksize = 1,
        ), desc = "Converting NPY Files to WAV Files", total = len(dataset)))
    
    ##################################################

##################################################