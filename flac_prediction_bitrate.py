# README
# Phillip Long
# Test what percent of encoded bits in FLAC files are involved in encoding waveform estimations.

# IMPORTS
##################################################

import numpy as np
import pandas as pd
import argparse
import multiprocessing
import scipy.io.wavfile
from tqdm import tqdm
from os.path import exists, basename, dirname, getsize
from os import makedirs, listdir, remove
import subprocess
import tempfile
import os

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

import utils
from test_lossless_compressors.test_flac import FLAC_PATH
from preprocess_musdb18 import get_mixes_only_mask

##################################################


# CONSTANTS
##################################################

# input directory with FLAC files
INPUT_DIR = f"{utils.EVAL_DIR}/flac/data/flac"

# output filepath for table
OUTPUT_FILEPATH = f"{utils.EVAL_DIR}/flac/{basename(__file__)[:-len('.py')]}.csv"

# output columns for comparison to other methods
FLAC_ESTIMATION_METHODS = ["verbatim", "constant", "fixed", "lpc"]
OUTPUT_COLUMNS = (["path", "duration", "size_total", "size_compressed"] + 
                  ["size_estimation"] + [f"size_{estimation_method}" for estimation_method in FLAC_ESTIMATION_METHODS] + ["size_entropy"] +
                  ["n_subframes"] + [f"n_subframes_{estimation_method}" for estimation_method in FLAC_ESTIMATION_METHODS] + 
                  ["bitrate_total", "bitrate_estimation", "estimation_proportion", "entropy_proportion", "lossless_compression_rate", "lossy_compression_rate"])

# output directory for residuals
RESIDUALS_OUTPUT_DIR = f"{utils.LOGGING_FOR_ZACH_DIR}/flac"
RESIDUALS_OUTPUT_COLUMNS = ["path", "original_path", "mean_squared_error"]

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # SETUP
    ##################################################

    # read in arguments
    def parse_args(args = None, namespace = None):
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(prog = "Evaluate", description = "Evaluate FLAC to Determine Estimation Bitrate") # create argument parser
        parser.add_argument("--input_dir", type = str, default = INPUT_DIR, help = "Absolute filepath to directory containing FLAC files to evaluate.")
        parser.add_argument("--output_filepath", type = str, default = OUTPUT_FILEPATH, help = "Absolute filepath to the output CSV file.")
        parser.add_argument("--residuals_output_dir", type = str, default = RESIDUALS_OUTPUT_DIR, help = "Absolute filepath to directory for residuals.")
        parser.add_argument("--flac_path", type = str, default = FLAC_PATH, help = "Absolute filepath to the FLAC CLI.")
        parser.add_argument("--mixes_only", action = "store_true", help = "Compute statistics for only mixes in MUSDB18, not all stems.")
        parser.add_argument("--reset", action = "store_true", help = "Re-evaluate files.")
        parser.add_argument("-j", "--jobs", type = int, default = int(multiprocessing.cpu_count() / 4), help = "Number of workers for multiprocessing.")
        args = parser.parse_args(args = args, namespace = namespace) # parse arguments
        if not exists(args.input_dir): # ensure input_dir exists
            raise RuntimeError(f"--input_dir argument does not exist: {args.input_dir}")
        elif len(listdir(args.input_dir)) == 0: # ensure input_dir is not empty
            raise RuntimeError(f"--input_dir argument is empty: {args.input_dir}")
        return args # return parsed arguments
    args = parse_args()
    
    # create output directory if necessary
    output_dir = dirname(args.output_filepath)
    if not exists(output_dir):
        makedirs(output_dir, exist_ok = True)
    residuals_data_dir = f"{args.residuals_output_dir}/data"
    if not exists(residuals_data_dir):
        makedirs(residuals_data_dir, exist_ok = True)
    residuals_output_filepath = f"{args.residuals_output_dir}/data.csv"
    
    # write output columns if necessary
    if not exists(args.output_filepath) or args.reset: # write column names
        pd.DataFrame(columns = OUTPUT_COLUMNS).to_csv(path_or_buf = args.output_filepath, sep = ",", na_rep = utils.NA_STRING, header = True, index = False, mode = "w")
        already_completed_paths = set() # no paths have been already completed
    else: # determine already completed paths
        results = pd.read_csv(filepath_or_buffer = args.output_filepath, sep = ",", header = 0, index_col = False)
        already_completed_paths = set(results["path"])
        del results # free up memory
    if not exists(residuals_output_filepath) or args.reset:
        pd.DataFrame(columns = RESIDUALS_OUTPUT_COLUMNS).to_csv(path_or_buf = residuals_output_filepath, sep = ",", na_rep = utils.NA_STRING, header = True, index = False, mode = "w")

    # get paths to evaluate
    paths = map(lambda base: f"{args.input_dir}/{base}", listdir(args.input_dir)) # get absolute paths
    paths = list(filter(lambda path: path.endswith(".flac"), paths)) # ensure just FLAC files
    if len(paths) == 0:
        raise RuntimeError(f"--input_dir argument does not contain any FLAC files to evaluate: {args.input_dir}")

    ##################################################


    # EVALUATE ESTIMATION BITRATE
    ##################################################

    # helper function for determining bitrate of encoded estimation bits
    def evaluate(path: str):
        """
        Evaluate bitrate of encoded estimation bits.
        """

        # save time by avoiding unnecessary calculations
        if path in already_completed_paths and not args.reset:
            return # return nothing, stop execution here
        path_prefix = basename(path)[:-len(".flac")] # get prefix of path
        
        # get total number of bits for FLAC file
        size_compressed = getsize(path) * 8

        # create temporary files for this specific evaluation
        wav_fd, wav_filepath = tempfile.mkstemp(suffix=".wav", prefix=f"flac_eval_{path_prefix}_")
        residuals_fd, residuals_filepath = tempfile.mkstemp(suffix=".bin", prefix=f"flac_residuals_{path_prefix}_")
        
        # wrap in try statement to catch errors
        try:

            # close file descriptors since we only need the file paths
            os.close(wav_fd)
            os.close(residuals_fd)
            
            # get statistics from FLAC decoder
            result = subprocess.run(args = [args.flac_path, "--force", "--decode", "--log-residuals", residuals_filepath, "-o", wav_filepath, path], check = True, stdout = subprocess.PIPE, stderr = subprocess.DEVNULL) # encode WAV file as FLAC
            result = result.stdout.decode("utf-8") # read stdout as a string
            result = next(filter(lambda line: line.startswith("Prediction Statistics: "), result.split("\n"))) # get the line that includes estimation bits
            statistics = eval(result[len("Prediction Statistics: "):]) # statistics
            size_estimation = statistics["prediction_bits"]
            size_entropy = statistics["entropy_bits"]
            del result # free up memory

            # get duration of decoded waveform
            sample_rate, waveform = scipy.io.wavfile.read(filename = wav_filepath)
            duration = len(waveform) / sample_rate # duration of waveform in seconds
            del waveform, sample_rate # free up memory
            size_total = getsize(wav_filepath) * 8

            # statistics
            bitrate_total = size_compressed / duration # overall bitrate of the FLAC file
            bitrate_estimation = size_estimation / duration # bitrate of the estimation bits in the FLAC file
            estimation_proportion = size_estimation / size_compressed # proportion of bits that are estimation bits
            entropy_proportion = size_entropy / size_compressed # proportion of bits that are entropy bits
            lossless_compression_rate = utils.get_compression_rate(size_original = size_total, size_compressed = size_compressed)
            lossy_compression_rate = utils.get_compression_rate(size_original = size_total, size_compressed = size_estimation)

            # output
            pd.DataFrame(data = [dict(zip(
                OUTPUT_COLUMNS, 
                [path, duration, size_total, size_compressed] + [statistics[key] for key in ("prediction_bits", "verbatim_bits", "constant_bits", "fixed_bits", "lpc_bits", "entropy_bits", "n_subframes", "verbatim_n_subframes", "constant_n_subframes", "fixed_n_subframes", "lpc_n_subframes")] + [bitrate_total, bitrate_estimation, estimation_proportion, entropy_proportion, lossless_compression_rate, lossy_compression_rate]
            ))]).to_csv(path_or_buf = args.output_filepath, sep = ",", na_rep = utils.NA_STRING, header = False, index = False, mode = "a")

            # read in residuals
            residuals = np.fromfile(file = residuals_filepath, dtype = np.int32)
            mean_squared_error = np.mean(residuals ** 2)

            # write residuals
            output_residuals_filepath = f"{residuals_data_dir}/{path_prefix}.npy"
            np.save(file = output_residuals_filepath, arr = residuals)
            pd.DataFrame(data = [dict(zip(
                RESIDUALS_OUTPUT_COLUMNS,
                [output_residuals_filepath, path, mean_squared_error]
            ))]).to_csv(path_or_buf = residuals_output_filepath, sep = ",", na_rep = utils.NA_STRING, header = False, index = False, mode = "a")

        finally:
            # clean up temporary files
            try:
                if exists(wav_filepath):
                    remove(wav_filepath)
                if exists(residuals_filepath):
                    remove(residuals_filepath)
            except OSError:
                pass  # ignore cleanup errors

        # return nothing
        return

    # use multiprocessing
    with multiprocessing.Pool(processes = args.jobs) as pool:
        _ = list(tqdm(iterable = pool.imap_unordered(
                func = evaluate,
                iterable = paths,
                chunksize = utils.CHUNK_SIZE,
            ),
            desc = "Evaluating",
            total = len(paths)))
    print(utils.MAJOR_SEPARATOR_LINE)
        
    # free up memory
    del already_completed_paths, paths

    ##################################################


    # FINAL STATISTICS
    ##################################################

    # read in results (just the compression rate column, we don't really care about anything else)
    results = pd.read_csv(filepath_or_buffer = args.output_filepath, sep = ",", header = 0, index_col = False)
    if args.mixes_only: # filter for only mixes
        results = results[get_mixes_only_mask(paths = results["path"])]

    # output statistics on overall bitrate
    bitrates_total = results["bitrate_total"].to_numpy()
    print(f"Mean Bitrate: {np.mean(bitrates_total):.2f} bps")
    print(f"Median Bitrate: {np.median(bitrates_total):.2f} bps")
    print(f"Standard Deviation of Bitrates: {np.std(bitrates_total):.2f} bps")
    print(f"Maximum Bitrate: {np.max(bitrates_total):.2f} bps")
    print(f"Minimum Bitrate: {np.min(bitrates_total):.2f} bps")
    print(utils.MINOR_SEPARATOR_LINE)

    # output statistics on estimation bitrate
    bitrates_estimation = results["bitrate_estimation"].to_numpy()
    print(f"Mean Estimation Bitrate: {np.mean(bitrates_estimation):.2f} bps")
    print(f"Median Estimation Bitrate: {np.median(bitrates_estimation):.2f} bps")
    print(f"Standard Deviation of Estimation Bitrates: {np.std(bitrates_estimation):.2f} bps")
    print(f"Maximum Estimation Bitrate: {np.max(bitrates_estimation):.2f} bps")
    print(f"Minimum Estimation Bitrate: {np.min(bitrates_estimation):.2f} bps")
    print(utils.MINOR_SEPARATOR_LINE)

    # output statistics on subframe rate
    subframe_rates = (results["n_subframes"] / results["duration"]).to_numpy()
    print(f"Mean Subframe Rate: {np.mean(subframe_rates):.2f} fps")
    print(f"Median Subframe Rate: {np.median(subframe_rates):.2f} fps")
    print(f"Standard Deviation of Subframe Rates: {np.std(subframe_rates):.2f} fps")
    print(f"Maximum Subframe Rate: {np.max(subframe_rates):.2f} fps")
    print(f"Minimum Subframe Rate: {np.min(subframe_rates):.2f} fps")
    print(utils.MINOR_SEPARATOR_LINE)

    # output statistics on estimation percentage
    estimation_percentages = results["estimation_proportion"].to_numpy() * 100 # convert to percentage
    print(f"Mean Estimation Percentage: {np.mean(estimation_percentages):.2f}%")
    print(f"Median Estimation Percentage: {np.median(estimation_percentages):.2f}%")
    print(f"Standard Deviation of Estimation Percentages: {np.std(estimation_percentages):.2f}%")
    print(f"Minimum Estimation Percentage: {np.min(estimation_percentages):.2f}%")
    print(f"Maximum Estimation Percentage: {np.max(estimation_percentages):.2f}%")
    print(utils.MINOR_SEPARATOR_LINE)

    # output statistics on entropy percentage
    entropy_percentages = results["entropy_proportion"].to_numpy() * 100 # convert to percentage
    print(f"Mean Entropy Percentage: {np.mean(entropy_percentages):.2f}%")
    print(f"Median Entropy Percentage: {np.median(entropy_percentages):.2f}%")
    print(f"Standard Deviation of Entropy Percentages: {np.std(entropy_percentages):.2f}%")
    print(f"Minimum Entropy Percentage: {np.min(entropy_percentages):.2f}%")
    print(f"Maximum Entropy Percentage: {np.max(entropy_percentages):.2f}%")
    print(utils.MINOR_SEPARATOR_LINE)

    # output statistics on lossless compression rate
    lossless_compression_rates = results["lossless_compression_rate"].to_numpy() * 100 # convert to percentage
    print(f"Mean Lossless Compression Rate: {np.mean(lossless_compression_rates):.2f}%")
    print(f"Median Lossless Compression Rate: {np.median(lossless_compression_rates):.2f}%")
    print(f"Standard Deviation of Lossless Compression Rates: {np.std(lossless_compression_rates):.2f}%")
    print(f"Minimum Lossless Compression Rate: {np.min(lossless_compression_rates):.2f}%")
    print(f"Maximum Lossless Compression Rate: {np.max(lossless_compression_rates):.2f}%")
    print(utils.MINOR_SEPARATOR_LINE)

    # output statistics on lossy compression rate
    lossy_compression_rates = results["lossy_compression_rate"].to_numpy() * 100 # convert to percentage
    print(f"Mean Lossy Compression Rate: {np.mean(lossy_compression_rates):.2f}%")
    print(f"Median Lossy Compression Rate: {np.median(lossy_compression_rates):.2f}%")
    print(f"Standard Deviation of Lossy Compression Rates: {np.std(lossy_compression_rates):.2f}%")
    print(f"Minimum Lossy Compression Rate: {np.min(lossy_compression_rates):.2f}%")
    print(f"Maximum Lossy Compression Rate: {np.max(lossy_compression_rates):.2f}%")
    print(utils.MINOR_SEPARATOR_LINE)

    # output statistics on different predictor method subframe percentages
    print(f"Mean Verbatim Subframe Percentage: {100 * np.mean(results['n_subframes_verbatim'] / results['n_subframes']):.2f}%")
    print(f"Mean Constant Subframe Percentage: {100 * np.mean(results['n_subframes_constant'] / results['n_subframes']):.2f}%")
    print(f"Mean Fixed Subframe Percentage: {100 * np.mean(results['n_subframes_fixed'] / results['n_subframes']):.2f}%")
    print(f"Mean LPC Subframe Percentage: {100 * np.mean(results['n_subframes_lpc'] / results['n_subframes']):.2f}%")
    print(utils.MINOR_SEPARATOR_LINE)

    # output statistics on different predictor method bit percentages
    print(f"Mean Verbatim Bit Percentage: {100 * np.mean(results['size_verbatim'] / results['size_estimation']):.2f}%")
    print(f"Mean Constant Bit Percentage: {100 * np.mean(results['size_constant'] / results['size_estimation']):.2f}%")
    print(f"Mean Fixed Bit Percentage: {100 * np.mean(results['size_fixed'] / results['size_estimation']):.2f}%")
    print(f"Mean LPC Bit Percentage: {100 * np.mean(results['size_lpc'] / results['size_estimation']):.2f}%")
    print(utils.MAJOR_SEPARATOR_LINE)

    ##################################################

##################################################