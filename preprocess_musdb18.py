# README
# Phillip Long
# June 9, 2025

# Preprocess the MusDB18 dataset, which we use as a testbed, to our liking.

# IMPORTS
##################################################

import numpy as np
import pandas as pd
import stempeg
import librosa
import argparse
import multiprocessing
from typing import List
from tqdm import tqdm
from os import makedirs, mkdir, listdir
from os.path import basename, exists, isdir
from shutil import rmtree
from glob import iglob

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

import utils

##################################################


# HELPER METHOD
##################################################

def get_mixes_only_mask(paths: pd.Series) -> List[bool]:
    """
    Given the list of paths to the preprocessed MUSDB18 dataset, return a boolean mask array 
    where only the audio files that are mixes (not just stems) are True.
    """
    return list(map(lambda path: path.split(".")[-2] == "0", paths))

def get_test_only_mask(paths: pd.Series, musdb18_dir: str = utils.MUSDB18_DIR) -> List[bool]:
    """
    Given the list of paths to the preprocessed MUSDB18 dataset, return a boolean mask array 
    where only the audio files that are stems are from the test set are True.
    """
    test_stems = listdir(f"{musdb18_dir}/test") # get test stems
    test_stems = set(map(lambda stem: stem[:-len(".stem.mp4")], test_stems)) # get test stems without extension
    assert len(test_stems) == 50, f"Expected 50 test stems, got {len(test_stems)}."
    return list(map(lambda path: basename(path)[:-len(".stem.0.npy")] in test_stems, paths))

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # SETUP
    ##################################################

    # read in arguments
    def parse_args(args = None, namespace = None):
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(prog = "Preprocess", description = "Preprocess the MusDB18 Dataset as pickled numpy (NPY) files.") # create argument parser
        parser.add_argument("--musdb18_dir", type = str, default = utils.MUSDB18_DIR, help = "Absolute filepath to the MusDB18 directory.")
        parser.add_argument("--output_dir", type = str, default = utils.MUSDB18_PREPROCESSED_DIR, help = "Absolute filepath to the output directory.")
        parser.add_argument("--sample_rate", type = int, default = utils.SAMPLE_RATE, help = "Sample rate of processed audio. Resamples input audio data if the original sample rate does not match.")
        parser.add_argument("--bit_depth", type = int, default = 16, choices = (16, 24, 32), help = "Fixed-point bit depth of audio.")
        parser.add_argument("--reset", action = "store_true", help = "Re-preprocess MusDB18.")
        parser.add_argument("-j", "--jobs", type = int, default = int(multiprocessing.cpu_count() / 4), help = "Number of workers for multiprocessing.")
        args = parser.parse_args(args = args, namespace = namespace) # parse arguments
        return args # return parsed arguments
    args = parse_args()

    # ensure arguments are valid
    if not exists(args.musdb18_dir):
        raise RuntimeError(f"--musdb18_dir argument does not exist: {args.musdb18_dir}")
    elif not isdir(args.musdb18_dir):
        raise RuntimeError(f"--musdb18_dir argument is not a directory: {args.musdb18_dir}")
    if args.sample_rate <= 0:
        raise RuntimeError(f"--sample_rate must be a positive integer: {args.sample_rate}.")
    
    # the output directory will end with the sample rate, so that we can have different versions with different sample rates
    args.output_dir += f"-{args.sample_rate}"
    
    # create output directory if it does not yet exist
    if not exists(args.output_dir) or args.reset:
        if exists(args.output_dir) and args.reset:
            rmtree(args.output_dir)
        makedirs(args.output_dir, exist_ok = False)
    output_dir = f"{args.output_dir}/data" # directory to store data
    if not exists(output_dir) or args.reset:
        mkdir(output_dir)
    output_filepath = f"{args.output_dir}/data.csv" # output data table
    if not exists(output_filepath) or args.reset: # write column names
        pd.DataFrame(columns = utils.STEMS_TO_AUDIO_COLUMN_NAMES).to_csv(path_or_buf = output_filepath, sep = ",", na_rep = utils.NA_STRING, header = True, index = False, mode = "w")
        already_completed_paths = set() # no paths have been already completed
    else: # determine already completed paths
        already_completed_paths = set(pd.read_csv(filepath_or_buffer = output_filepath, sep = ",", header = 0, index_col = False, usecols = ["original_path"])["original_path"]) # read in already completed paths

    # determine bit depth
    audio_data_type = np.int16 if args.bit_depth == 16 else np.int32

    ##################################################


    # PREPROCESSING
    ##################################################

    # preprocessing function
    def preprocess(path: str):
        """
        Preprocess MusDB18 file given the input Native Instruments stems format (MP4) absolute filepath.
        """

        # save time by avoiding unnecessary calculations
        if path in already_completed_paths and not args.reset:
            return # return nothing, stop execution here

        # load in mp4
        stems, sample_rate = stempeg.read_stems(filename = path)
        n_stems = len(stems)

        # resample if necessary
        if sample_rate != args.sample_rate:
            stems = librosa.resample(y = stems, orig_sr = sample_rate, target_sr = args.sample_rate, axis = 1)
            sample_rate = args.sample_rate

        # convert audio from floating to fixed point
        stems = utils.convert_waveform_floating_to_fixed(waveform = stems, output_dtype = audio_data_type)

        # determine stem output paths
        stem_paths = [output_dir + "/" + basename(path)[:-len("mp4")] + f"{i}.npy" for i in range(n_stems)]

        # save stems as pickled numpy arrays
        for i, stem_path in enumerate(stem_paths):
            np.save(file = stem_path, arr = stems[i])

        # append to output file
        pd.DataFrame(data = dict(zip(
            utils.STEMS_TO_AUDIO_COLUMN_NAMES,
            (stem_paths, utils.rep(x = sample_rate, times = n_stems), utils.rep(x = path, times = n_stems), list(range(n_stems))),
        ))).to_csv(path_or_buf = output_filepath, sep = ",", na_rep = utils.NA_STRING, header = False, index = False, mode = "a")

        # return nothing
        return
    
    # get musdb18 paths
    paths = [f"{args.musdb18_dir}/{base}" for base in iglob("**/*.mp4", root_dir = args.musdb18_dir, recursive = True)]

    # use multiprocessing to preprocess musdb18 paths
    with multiprocessing.Pool(processes = args.jobs) as pool:
        _ = list(tqdm(iterable = pool.imap_unordered(
                func = preprocess,
                iterable = paths,
                chunksize = utils.CHUNK_SIZE,
            ),
            desc = "Preprocessing",
            total = len(paths)))

    ##################################################

##################################################
