# README
# Phillip Long
# July 7, 2025

# Tests the FLAC entropy coder on real residual data from MUSDB18.

# IMPORTS
##################################################

import pandas as pd
import numpy as np
import multiprocessing
from tqdm import tqdm
import argparse
from os.path import exists, basename, getsize
from os import listdir, makedirs, mkdir, remove
import tempfile
import scipy.io.wavfile
import subprocess

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(dirname(realpath(__file__)))))

from entropy_coders_factory import factory
import utils
from preprocess_musdb18 import get_mixes_only_mask

##################################################


# CONSTANTS
##################################################

# filepaths
INPUT_FILEPATH = "/deepfreeze/pnlong/lnac/eval/flac/test.csv"
WAVEFORMS_DIR = "/deepfreeze/pnlong/lnac/eval/flac/data/wav"
RESIDUALS_DIR = "/deepfreeze/user_shares/pnlong/lnac/logging_for_zach/flac/data"
OUTPUT_DIR = f"{dirname(WAVEFORMS_DIR)}/test_flac_entropy_coder"
FLAC_PATH = f"{dirname(dirname(dirname(realpath(__file__))))}/flac/src/flac/flac"

# output columns
OUTPUT_COLUMNS = ["flac_path", "residual_path", "size_compressed", "size_residuals", "percent_difference"]

# we want to only evaluate a single block because the entropy coder expects residuals at the size of a block, not the entire file
FIRST_N_SAMPLES = 2048

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # SETUP
    ##################################################

    # read in arguments
    def parse_args(args = None, namespace = None):
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(prog = "Test", description = "Test FLAC Entropy Coder.") # create argument parser
        parser.add_argument("--waveforms_dir", type = str, default = WAVEFORMS_DIR, help = "Absolute filepath to directory with WAV files.")
        parser.add_argument("--residuals_dir", type = str, default = RESIDUALS_DIR, help = "Absolute filepath to directory for residuals.")
        parser.add_argument("--output_dir", type = str, default = OUTPUT_DIR, help = "Absolute filepath to directory for output.")
        parser.add_argument("--first_n_samples", type = int, default = FIRST_N_SAMPLES, help = "Number of samples to evaluate on.")
        parser.add_argument("--flac_path", type = str, default = FLAC_PATH, help = "Absolute filepath to the FLAC CLI.")
        parser.add_argument("--mixes_only", action = "store_true", help = "Compute statistics for only mixes in MUSDB18, not all stems.")
        parser.add_argument("--reset", action = "store_true", help = "Re-evaluate files.")
        parser.add_argument("-j", "--jobs", type = int, default = int(multiprocessing.cpu_count() / 4), help = "Number of workers for multiprocessing.")
        args = parser.parse_args(args = args, namespace = namespace) # parse arguments
        if not exists(args.waveforms_dir): # ensure waveforms_dir exists
            raise RuntimeError(f"--waveforms_dir argument does not exist: {args.waveforms_dir}")
        elif len(listdir(args.waveforms_dir)) == 0: # ensure residuals_dir is not empty
            raise RuntimeError(f"--waveforms_dir argument is empty: {args.waveforms_dir}")
        elif not exists(args.residuals_dir): # ensure residuals_dir exists
            raise RuntimeError(f"--residuals_dir argument does not exist: {args.residuals_dir}")
        elif len(listdir(args.residuals_dir)) == 0: # ensure residuals_dir is not empty
            raise RuntimeError(f"--residuals_dir argument is empty: {args.residuals_dir}")
        elif not exists(args.flac_path): # ensure flac_path exists
            raise RuntimeError(f"--flac_path argument does not exist: {args.flac_path}")
        elif args.first_n_samples <= 0: # ensure first_n_samples is positive
            raise RuntimeError(f"--first_n_samples argument is not positive: {args.first_n_samples}")
        return args # return parsed arguments
    args = parse_args()

    # ensure output directory exists
    if not exists(args.output_dir):
        makedirs(args.output_dir, exist_ok = True)
    flac_dir = f"{args.output_dir}/flac"
    if not exists(flac_dir):
        mkdir(flac_dir)

    # write output columns if necessary
    output_filepath = f"{args.output_dir}/test_flac_entropy_coder.csv"
    if not exists(output_filepath) or args.reset: # write column names
        pd.DataFrame(columns = OUTPUT_COLUMNS).to_csv(path_or_buf = output_filepath, sep = ",", na_rep = utils.NA_STRING, header = True, index = False, mode = "w")
        already_completed_residual_paths = set() # no paths have been already completed
    else: # determine already completed paths
        results = pd.read_csv(filepath_or_buffer = output_filepath, sep = ",", header = 0, index_col = False)
        already_completed_residual_paths = set(results["residual_path"])
        del results # free up memory

    # get residual paths to evaluate
    residual_paths = map(lambda base: f"{args.residuals_dir}/{base}", listdir(args.residuals_dir)) # get absolute paths
    residual_paths = list(filter(lambda residual_path: residual_path not in already_completed_residual_paths, residual_paths)) # filter out already completed paths

    ##################################################


    # RUN ENTROPY ENCODER
    ##################################################

    # create entropy coder
    entropy_coder = factory("flac_rice")

    # temporary directory for WAV files
    with tempfile.TemporaryDirectory() as temp_dir:

        # helper function to evaluate a single path
        def evaluate(residual_path: str):
            """Evaluate a single path."""

            # get waveform and flac path from residual path
            residual_path_base = basename(residual_path)[:-len(".npy")]
            waveform_path = f"{args.waveforms_dir}/{residual_path_base}.wav"
            flac_path = f"{flac_dir}/{residual_path_base}.flac"

            # load in waveform and truncate to first_n_samples
            sample_rate, waveform = scipy.io.wavfile.read(filename = waveform_path)
            waveform = waveform[:args.first_n_samples] # truncate
            truncated_waveform_path = f"{temp_dir}/{residual_path_base}.wav"
            scipy.io.wavfile.write(filename = truncated_waveform_path, rate = sample_rate, data = waveform)

            # encode truncated waveform as flac
            result = subprocess.run(
                args = [args.flac_path, "--force", "-o", flac_path, truncated_waveform_path],
                check = True,
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
            ) # encode WAV file as FLAC

            size_compressed = getsize(flac_path)
            remove(truncated_waveform_path)

            # get residuals and encode them
            residuals = np.load(residual_path).astype(np.int32)
            residuals = residuals[:args.first_n_samples] # truncate
            residuals_rice = entropy_coder.encode(nums = residuals)
            size_residuals = len(residuals_rice)

            # ensure residuals are encoded correctly (for debugging)
            # residuals_roundtrip = entropy_coder.decode(stream = residuals_rice, num_samples = len(residuals))
            # assert np.array_equal(residuals, residuals_roundtrip), "Residuals are not encoded correctly!"

            # calculate percent difference
            percent_difference = 100 * ((size_compressed - size_residuals) / size_compressed)

            # write to results
            pd.DataFrame(data = [dict(zip(
                OUTPUT_COLUMNS,
                [flac_path, residual_path, size_compressed, size_residuals, percent_difference]
            ))]).to_csv(path_or_buf = output_filepath, sep = ",", na_rep = utils.NA_STRING, header = False, index = False, mode = "a")

            # return nothing
            return
        
        # use multiprocessing
        with multiprocessing.Pool(processes = args.jobs) as pool:
            _ = list(tqdm(iterable = pool.imap_unordered(
                    func = evaluate,
                    iterable = residual_paths,
                    chunksize = utils.CHUNK_SIZE,
                ),
                desc = "Testing",
                total = len(residual_paths)))
        print(utils.MAJOR_SEPARATOR_LINE)

    ##################################################


    # RESULTS
    ##################################################

    # read in results
    results = pd.read_csv(filepath_or_buffer = output_filepath, sep = ",", header = 0, index_col = False)
    if args.mixes_only: # filter for only mixes
        results = results[get_mixes_only_mask(paths = results["residual_path"])]

    # print results
    print("FLAC Entropy Coder Results (Percent Difference = 100 * ((FLAC SIZE - ENTROPY CODED RESIDUALS SIZE) / FLAC SIZE)):")
    percent_differences = results["percent_difference"].to_numpy()
    print(f"Mean Percent Difference: {np.mean(percent_differences):.2f}%")
    print(f"Median Percent Difference: {np.median(percent_differences):.2f}%")
    print(f"Standard Deviation of Percent Difference: {np.std(percent_differences):.2f}%")
    print(f"Maximum Percent Difference: {np.max(percent_differences):.2f}%")
    print(f"Minimum Percent Difference: {np.min(percent_differences):.2f}%")
    print(utils.MAJOR_SEPARATOR_LINE)

    ##################################################

##################################################