# README
# Phillip Long
# August 16, 2025

# Convert residuals to WAV files.

# IMPORTS
##################################################

import pandas as pd
from os.path import exists, basename
from os import makedirs, mkdir
from shutil import rmtree
import argparse
import multiprocessing
import numpy as np
import scipy.io.wavfile
from tqdm import tqdm
import random

##################################################


# CONSTANTS
##################################################

# filepaths
RESIDUAL_DIR = "/deepfreeze/user_shares/pnlong/lnac/logging_for_zach/flac/data"
INPUT_FILEPATH = "/deepfreeze/pnlong/lnac/test_data/musdb18_preprocessed-44100/data.csv"
OUTPUT_DIR = "/deepfreeze/pnlong/lnac/listen_to_residuals"
N_SONGS = 4 # number of songs to convert, of which there are 150 in MUSDB18; for each song, convert all four stems and the mix

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

# convert residuals to wav files
def convert_residuals_to_wav(
    residual_path: str,
    output_path: str,
    sample_rate: int,
):
    """
    Convert a residual to a WAV file.

    Parameters
    ----------
    residual_path : str
        Path to the residual file.
    output_path : str
        Path to the output WAV file.
    sample_rate : int
        Sample rate of the WAV file.

    Returns
    -------
    None
    """

    # read in residual
    residual = np.load(residual_path).reshape(2, -1).astype(np.int16)

    # write residual as a WAV file
    if len(residual.shape) == 2 and residual.shape[-1] != 2:
        residual = residual.T
    scipy.io.wavfile.write(filename = output_path, rate = sample_rate, data = residual)

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
        parser.add_argument("--residual_dir", type = str, default = RESIDUAL_DIR, help = "Path to residuals directory.")
        parser.add_argument("--input_filepath", type = str, default = INPUT_FILEPATH, help = "Path to input file.")
        parser.add_argument("--output_dir", type = str, default = OUTPUT_DIR, help = "Path to output directory.")
        parser.add_argument("--n_songs", type = int, default = N_SONGS, help = "Number of songs to convert.")
        parser.add_argument("--jobs", type = int, default = int(multiprocessing.cpu_count() / 4), help = "Number of jobs to run in parallel.")
        parser.add_argument("--reset", action = "store_true", help = "Reset the output directory.")
        args = parser.parse_args(args = args, namespace = namespace) # parse arguments
        if not exists(args.residual_dir):
            raise FileNotFoundError(f"Residuals directory not found: {args.residual_dir}")
        elif not exists(args.input_filepath):
            raise FileNotFoundError(f"Input file not found: {args.input_filepath}")
        return args # return parsed arguments
    args = parse_args()

    # set random seed
    random.seed(0)

    # read in input csv
    print("Reading in input data...")
    dataset = pd.read_csv(filepath_or_buffer = args.input_filepath, sep = ",", header = 0, index_col = False)
    dataset = dataset.rename(columns = {"path": "input_path"})
    dataset["residual_path"] = list(map(lambda path: f"{args.residual_dir}/{basename(path)}", dataset["input_path"]))
    dataset = dataset[["input_path", "residual_path", "sample_rate"]] # get only the path and sample rate columns, as that is all we care about
    print(f"Completed reading in input data.")

    # sample songs
    dataset["song"] = list(map(lambda path: basename(path).split(".")[0], dataset["input_path"]))
    sampled_songs = random.sample(population = dataset["song"].to_list(), k = args.n_songs)
    dataset = dataset[dataset["song"].isin(sampled_songs)].reset_index(drop = True)
    print(f"Sampled {len(sampled_songs)} songs, which yields {len(dataset)} stems and mixes.")
    del sampled_songs

    # create output directory
    print("Creating output directory...")
    if exists(args.output_dir) and args.reset:
        rmtree(args.output_dir)
    if not exists(args.output_dir):
        makedirs(args.output_dir, exist_ok = True)
    dataset["output_dir"] = list(map(lambda song_name: f"{args.output_dir}/{song_name}", dataset["song"]))
    for output_subdir in dataset["output_dir"].unique(): # the unique is redundant
        if not exists(output_subdir):
            mkdir(output_subdir)
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
        row = dataset.loc[i]
        input_path = row["input_path"]
        residual_path = row["residual_path"]
        sample_rate = row["sample_rate"]
        output_dir = row["output_dir"]
        input_index = input_path.split(".")[-2]
        
        # convert input to WAV file
        convert_npy_to_wav(input_path = input_path, output_path = f"{output_dir}/input.{input_index}.wav", sample_rate = sample_rate)
        
        # convert residual to WAV file
        convert_residuals_to_wav(residual_path = residual_path, output_path = f"{output_dir}/residual.{input_index}.wav", sample_rate = sample_rate)

    # use multiprocessing
    with multiprocessing.Pool(processes = args.jobs) as pool:
        _ = list(tqdm(iterable = pool.imap_unordered(
            func = convert_helper,
            iterable = dataset.index,
            chunksize = 1,
        ), desc = "Converting Residuals to WAV Files", total = len(dataset)))
    
    ##################################################

##################################################