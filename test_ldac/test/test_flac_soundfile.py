# README
# Phillip Long
# Verify compression rate of FLAC files using soundfile.

# IMPORTS
##################################################

import numpy as np
import pandas as pd
import argparse
import multiprocessing
from tqdm import tqdm
from os.path import exists, basename, dirname, getsize
from os import makedirs, remove
import tempfile
import os
import time
import soundfile as sf

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(dirname(dirname(realpath(__file__))))) # for preprocess_musdb18
sys.path.insert(0, dirname(dirname(realpath(__file__))))

from constants import INPUT_FILEPATH, OUTPUT_DIR, NA_STRING
from preprocess_musdb18 import get_mixes_only_mask, get_test_only_mask

##################################################


# CONSTANTS
##################################################

FLAC_PATH = f"{dirname(dirname(dirname(realpath(__file__))))}/flac/src/flac/flac"
OUTPUT_COLUMNS = ["path", "size_original", "size_compressed", "compression_rate", "duration_audio", "duration_encoding", "compression_speed"]

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # SETUP
    ##################################################

    # read in arguments
    def parse_args(args = None, namespace = None):
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(prog = "Evaluate", description = "Evaluate FLAC") # create argument parser
        parser.add_argument("--input_filepath", type = str, default = INPUT_FILEPATH, help = "Absolute filepath to CSV file describing the preprocessed MusDB18 dataset (see `preprocess_musdb18.py`).")
        parser.add_argument("--output_dir", type = str, default = OUTPUT_DIR, help = "Absolute filepath to the output directory.")
        parser.add_argument("--mixes_only", action = "store_true", help = "Compute statistics for only mixes in MUSDB18, not all stems.")
        parser.add_argument("--reset", action = "store_true", help = "Re-evaluate files.")
        parser.add_argument("-j", "--jobs", type = int, default = int(multiprocessing.cpu_count() / 4), help = "Number of workers for multiprocessing.")
        args = parser.parse_args(args = args, namespace = namespace) # parse arguments
        if not exists(args.input_filepath): # ensure input_filepath exists
            raise RuntimeError(f"--input_filepath argument does not exist: {args.input_filepath}")
        return args # return parsed arguments
    args = parse_args()
    
    # create output directory if necessary
    output_filepath = f"{args.output_dir}/test_flac_soundfile.csv"
    if not exists(args.output_dir):
        makedirs(args.output_dir, exist_ok = True)
    
    # write output columns if necessary
    if not exists(output_filepath) or args.reset:
        pd.DataFrame(columns = OUTPUT_COLUMNS).to_csv(path_or_buf = output_filepath, sep = ",", na_rep = NA_STRING, header = True, index = False, mode = "w") # write column names
        already_completed_paths = set() # no paths have been already completed
    else: # determine already completed paths
        results = pd.read_csv(filepath_or_buffer = output_filepath, sep = ",", header = 0, index_col = False)
        already_completed_paths = set(results["path"])
        del results # free up memory

    ##################################################


    # EVALUATE ESTIMATION BITRATE
    ##################################################

    # read in paths to evaluate
    sample_rate_by_path = pd.read_csv(filepath_or_buffer = args.input_filepath, sep = ",", header = 0, index_col = False, usecols = ["path", "sample_rate"])
    sample_rate_by_path = sample_rate_by_path[get_test_only_mask(paths = sample_rate_by_path["path"])] # filter to just the test set
    sample_rate_by_path = sample_rate_by_path.set_index(keys = "path", drop = True)["sample_rate"].to_dict() # dictionary where keys are paths and values are sample rates of those paths
    paths = list(sample_rate_by_path.keys()) # get paths to NPY audio files
    paths = list(filter(lambda path: path not in already_completed_paths, paths)) # filter out paths that have already been evaluated

    # only run if there are paths to evaluate
    if len(paths) > 0:

        # helper function for determining compression rate
        def evaluate(path: str):
            """
            Determine compression rate given the absolute filepath to an input audio file (stored as a pickled numpy object, NPY).
            Expects the input audio to be of shape (n_samples, n_channels) for multi-channel audio or (n_samples,) for mono audio.
            """

            # save time by avoiding unnecessary calculations
            if path in already_completed_paths and not args.reset:
                return # return nothing, stop execution here
            path_prefix = basename(path)[:-len(".npy")] # get prefix of path

            # create temporary files for this specific evaluation
            flac_fd, flac_filepath = tempfile.mkstemp(suffix = ".flac", prefix = f"flac_eval_{path_prefix}_")
            
            # wrap in try statement to catch errors
            try:

                # close file descriptors since we only need the file paths
                os.close(flac_fd)

                # first save the numpy array as a WAV file
                waveform = np.load(file = path)
                size_original = waveform.nbytes # compute size in bytes of original waveform
                # waveform = waveform.astype(np.int32) # ensure waveform is stored as int32 for FLAC
                sample_rate = sample_rate_by_path[path]
                duration_audio = len(waveform) / sample_rate

                # encode WAV file as FLAC
                start_time = time.perf_counter()
                sf.write(file = flac_filepath, data = waveform, samplerate = sample_rate, format = "FLAC")
                duration_encoding = time.perf_counter() - start_time # measure speed of compression
                size_compressed = getsize(flac_filepath)
                del waveform, start_time # free up memory

                # compute other final statistics
                compression_rate = size_original / size_compressed
                compression_speed = duration_audio / duration_encoding

                # output
                pd.DataFrame(data = [{
                    "path": path,
                    "size_original": size_original,
                    "size_compressed": size_compressed,
                    "compression_rate": compression_rate,
                    "duration_audio": duration_audio,
                    "duration_encoding": duration_encoding,
                    "compression_speed": compression_speed,
                }]).to_csv(path_or_buf = output_filepath, sep = ",", na_rep = NA_STRING, header = False, index = False, mode = "a")

            finally:
                # clean up temporary files
                try:
                    if exists(flac_filepath):
                        remove(flac_filepath)
                except OSError:
                    pass  # ignore cleanup errors

            # return nothing
            return

        # use multiprocessing
        with multiprocessing.Pool(processes = args.jobs) as pool:
            _ = list(tqdm(iterable = pool.imap_unordered(
                    func = evaluate,
                    iterable = paths,
                    chunksize = 1
                ),
                desc = "Evaluating",
                total = len(paths)))
        
    # free up memory
    del already_completed_paths, paths, sample_rate_by_path

    ##################################################


    # FINAL STATISTICS
    ##################################################

    print("=" * 60)

    # read in results (just the compression rate column, we don't really care about anything else)
    results = pd.read_csv(filepath_or_buffer = output_filepath, sep = ",", header = 0, index_col = False)
    if args.mixes_only: # filter for only mixes
        results = results[get_mixes_only_mask(paths = results["path"])]
    compression_rates = results["compression_rate"].to_numpy() * 100 # convert to percentages
    compression_speeds = results["compression_speed"].to_numpy()

    # output statistics on compression rate
    print(f"Mean Compression Rate: {np.mean(compression_rates):.2f}%")
    print(f"Median Compression Rate: {np.median(compression_rates):.2f}%")
    print(f"Standard Deviation of Compression Rates: {np.std(compression_rates):.2f}%")
    print(f"Best Compression Rate: {np.max(compression_rates):.2f}%")
    print(f"Worst Compression Rate: {np.min(compression_rates):.2f}%")
    print("-" * 60)

    # output statistics on compression speed
    print(f"Mean Compression Speed: {np.mean(compression_speeds):.2f}")
    print(f"Median Compression Speed: {np.median(compression_speeds):.2f}")
    print(f"Standard Deviation of Compression Speeds: {np.std(compression_speeds):.2f}")
    print(f"Best Compression Speed: {np.max(compression_speeds):.2f}")
    print(f"Worst Compression Speed: {np.min(compression_speeds):.2f}")
    print("-" * 60)

    ##################################################

##################################################