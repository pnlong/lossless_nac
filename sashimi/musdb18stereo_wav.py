# README
# Phillip Long
# August 16, 2025

# Convert MUSDB18 NPY files to WAV files.

# IMPORTS
##################################################

import pandas as pd
from os.path import exists
from os import makedirs, rename, listdir
from shutil import rmtree
import argparse
import multiprocessing
import numpy as np
import scipy.io.wavfile
from tqdm import tqdm
from math import ceil, log10

##################################################


# CONSTANTS
##################################################

# filepaths
INPUT_FILEPATH = "/deepfreeze/pnlong/lnac/test_data/musdb18_preprocessed-44100/data.csv"
OUTPUT_DIR = "/deepfreeze/pnlong/lnac/sashimi/data/musdb18stereo"

# clip length
CLIP_LENGTH = 60 * 44100 # make each clip 60 seconds
PAD_CLIPS_TO_FIXED_LENGTH = False # pad clips so they are all clip length

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
        parser.add_argument("--clip_length", type = int, default = None, help = "Length of each clip in seconds.")
        parser.add_argument("--pad_clips_to_fixed_length", action = "store_true", help = "Pad clips so they are all clip length.")
        parser.add_argument("--rename_simple", action = "store_true", help = "Rename files to be simple, e.g. out0.wav, out1.wav, etc.")
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
    print(f"Completed reading in input data.")

    # create output directory
    print("Creating output directory...")
    if exists(args.output_dir) and args.reset:
        print(f"Removing old output directory {args.output_dir} because --reset was called...")
        rmtree(args.output_dir)
        print(f"Removed old output directory {args.output_dir}.")
    if not exists(args.output_dir):
        makedirs(args.output_dir, exist_ok = True)
    print("Created output directory.")

    # determine fixed width for output file names
    dataset_fixed_width = ceil(log10(len(dataset)))

    # if clip length is not provided, don't partition into clips
    use_clips = args.clip_length is not None

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

        # get data
        path, sample_rate = dataset.loc[i]
        data = np.load(path) # loads in shape (n_samples, 2)
        n_samples, n_channels = data.shape
        assert n_channels == 2, "Data is not stereo."
        data = data.astype(np.int16) # ensure type is correct

        # if using clips, partition into clips of stereo files
        if use_clips:
            for clip_idx, start_index in enumerate(range(0, n_samples, args.clip_length)): # separate data into clips
                end_index = min(start_index + args.clip_length, n_samples) # get end index
                n_samples_in_clip = end_index - start_index # get number of samples in clip
                clip = data[start_index:end_index] # get clip
                if args.pad_clips_to_fixed_length and n_samples_in_clip < args.clip_length: # pad if necessary
                    clip = np.pad(array = clip, pad_width = [(0, args.clip_length - n_samples_in_clip), (0, 0)], mode = "constant") # end pad with zeros
                filename = f"{args.output_dir}/out{i:0{dataset_fixed_width}}.{clip_idx}.wav" # generate unique filename based on dataset index, and clip
                scipy.io.wavfile.write( # write WAV file
                    filename = filename,
                    rate = sample_rate,
                    data = clip,
                )

        # otherwise, just write the data as a stereo file
        else:
            filename = f"{args.output_dir}/out{i:0{dataset_fixed_width}}.wav" # generate unique filename based on dataset index
            scipy.io.wavfile.write( # write WAV file
                filename = filename,
                rate = sample_rate,
                data = data,
            )

    # use multiprocessing
    print("Converting NPY Files to WAV Files...")
    with multiprocessing.Pool(processes = args.jobs) as pool:
        _ = list(tqdm(iterable = pool.imap_unordered(
            func = convert_helper,
            iterable = dataset.index,
            chunksize = 1,
        ), desc = "Converting NPY Files to WAV Files", total = len(dataset)))
    print("Completed converting NPY Files to WAV Files.")

    # rename files if desired
    if args.rename_simple:
        print("Renaming Files to Simple Names because --rename_simple was called...")
        counter = 0
        bases = listdir(args.output_dir)
        fixed_width = ceil(log10(len(bases)))
        for base in tqdm(iterable = bases, desc = "Renaming Files to Simple Names", total = len(bases)):
            input_path = f"{args.output_dir}/{base}"
            output_path = f"{args.output_dir}/out{counter:0{fixed_width}}.wav"
            rename(src = input_path, dst = output_path)
            counter += 1
        print("Completed renaming files to simple names.")

    ##################################################

##################################################