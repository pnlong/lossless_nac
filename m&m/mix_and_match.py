# README
# Phillip Long
# July 15, 2025

# Mix and match a lossless compressor with an entropy coder and analyze compression rate.

# IMPORTS
##################################################

import pandas as pd
import numpy as np
import argparse
import multiprocessing
from os.path import exists
from os import mkdir, listdir
import warnings
import json
from tqdm import tqdm
import random

from os.path import dirname, realpath
import sys
sys.path.insert(0, f"{dirname(realpath(__file__))}/lossless_compressors")
sys.path.insert(0, f"{dirname(realpath(__file__))}/entropy_coders")
sys.path.insert(0, dirname(dirname(realpath(__file__))))

from lossless_compressors_factory import factory as lossless_compressor_factory
from lossless_compressors_factory import TYPES as LOSSLESS_COMPRESSOR_TYPES
from entropy_coders_factory import factory as entropy_coder_factory
from entropy_coders_factory import TYPES as ENTROPY_CODER_TYPES
import utils
from preprocess_musdb18 import get_mixes_only_mask

##################################################

# CONSTANTS
##################################################

INPUT_DIR = f"{utils.TEST_DATA_DIR}/musdb18_preprocessed-44100/data" # directory containing NPY files of audio to evaluate
OUTPUT_FILEPATH = f"{utils.EVAL_DIR}/m&m/mix_and_match.csv" # output filepath
N_SAMPLES = 50 # number of samples to evaluate on
OUTPUT_COLUMNS = ["path", "lossless_compressor", "entropy_coder", "size_original", "size_compressed", "compression_rate"] # output columns
BLOCK_SIZE_DEFAULT = 4096 # block size for the lossless compressor
N_SECONDS_DEFAULT = 20 # number of seconds to evaluate on

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # SETUP
    ##################################################

    # read in arguments
    def parse_args(args = None, namespace = None):
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(prog = "Mix and Match", description = "Mix and match a lossless compressor with an entropy coder and analyze compression rate.") # create argument parser
        parser.add_argument("--lossless_compressor", type = str, default = LOSSLESS_COMPRESSOR_TYPES[0], choices = LOSSLESS_COMPRESSOR_TYPES, help = "Lossless compressor type.")
        parser.add_argument("--entropy_coder", type = str, default = ENTROPY_CODER_TYPES[0], choices = ENTROPY_CODER_TYPES, help = "Entropy coder type.")
        parser.add_argument("--lossless_compressor_parameters", type = str, default = "{}", help = "Parameters for the lossless compressor.")
        parser.add_argument("--entropy_coder_parameters", type = str, default = "{}", help = "Parameters for the entropy coder.")
        parser.add_argument("--input_dir", type = str, default = INPUT_DIR, help = "Absolute filepath to directory containing NPY files of audio to evaluate.")
        parser.add_argument("--output_filepath", type = str, default = OUTPUT_FILEPATH, help = "Absolute filepath to the output CSV file.")
        parser.add_argument("--block_size", type = int, default = BLOCK_SIZE_DEFAULT, help = "Block size for the lossless compressor.")
        parser.add_argument("--interchannel_decorrelation", action = "store_true", help = "Whether to use interchannel decorrelation.")
        parser.add_argument("-n", "--n_samples", type = int, default = N_SAMPLES, help = "Number of samples to evaluate on.")
        parser.add_argument("--n_seconds", type = float, default = N_SECONDS_DEFAULT, help = "Number of seconds to evaluate on.")
        parser.add_argument("--mixes_only", action = "store_true", help = "Compute statistics for only mixes in MUSDB18, not all stems.")
        parser.add_argument("--reset", action = "store_true", help = "Re-evaluate files.")
        parser.add_argument("-j", "--jobs", type = int, default = int(multiprocessing.cpu_count() / 4), help = "Number of workers for multiprocessing.")
        args = parser.parse_args(args = args, namespace = namespace) # parse arguments
        if not exists(args.input_dir): # ensure input_dir exists
            raise RuntimeError(f"--input_dir argument does not exist: {args.input_dir}")
        elif len(listdir(args.input_dir)) == 0: # ensure input_dir is not empty
            raise RuntimeError(f"--input_dir argument is empty: {args.input_dir}")
        elif args.n_samples <= 0:
            raise RuntimeError(f"--n_samples argument must be greater than 0: {args.n_samples}")
        elif args.n_seconds <= 0:
            raise RuntimeError(f"--n_seconds argument must be greater than 0: {args.n_seconds}")
        args.lossless_compressor_parameters = json.loads(args.lossless_compressor_parameters)
        args.entropy_coder_parameters = json.loads(args.entropy_coder_parameters)
        return args # return parsed arguments
    args = parse_args()

    # set random seed for reproducibility
    random.seed(42)

    # get sample rate
    sample_rate = int(dirname(args.input_dir).split("/")[-1].split("-")[-1])
    n_samples_per_song = int(sample_rate * args.n_seconds)

    # create output directory if necessary
    output_dir = dirname(args.output_filepath)
    if not exists(output_dir):
        mkdir(output_dir)

    # write output columns if necessary
    if not exists(args.output_filepath): # write column names
        pd.DataFrame(columns = OUTPUT_COLUMNS).to_csv(path_or_buf = args.output_filepath, sep = ",", na_rep = utils.NA_STRING, header = True, index = False, mode = "w")
    results = pd.read_csv(filepath_or_buffer = args.output_filepath, sep = ",", header = 0, index_col = False)
    if args.reset:
        results = results[~((results["lossless_compressor"] == args.lossless_compressor) & (results["entropy_coder"] == args.entropy_coder))]
        results.to_csv(path_or_buf = args.output_filepath, sep = ",", na_rep = utils.NA_STRING, header = True, index = False, mode = "w")
        already_completed_paths = set() # no paths have been already completed
    else: # determine already completed paths
        already_completed_paths = set(results[(results["lossless_compressor"] == args.lossless_compressor) & (results["entropy_coder"] == args.entropy_coder)]["path"])
    del results # free up memory

    # get paths to evaluate
    paths = list(map(lambda base: f"{args.input_dir}/{base}", listdir(args.input_dir))) # get absolute paths
    random.shuffle(paths)
    if args.mixes_only:
        paths = [path for is_mix, path in zip(get_mixes_only_mask(pd.Series(paths)), paths) if is_mix] # filter out stems
    if args.n_samples > len(paths):
        warnings.warn(f"--n_samples argument is greater than the number of paths to evaluate: {args.n_samples} > {len(paths)}, defaulting to {len(paths)}.")
        args.n_samples = len(paths)
    paths = list(filter(lambda path: path not in already_completed_paths, paths))[:args.n_samples]

    # create entropy coder
    entropy_coder = entropy_coder_factory(type_ = args.entropy_coder, **args.entropy_coder_parameters)

    # create lossless compressor
    lossless_compressor = lossless_compressor_factory(type_ = args.lossless_compressor, entropy_coder = entropy_coder, block_size = args.block_size, interchannel_decorrelation = args.interchannel_decorrelation, **args.lossless_compressor_parameters)

    ##################################################


    # COMPUTE COMPRESSION RATE
    ##################################################

    # evaluate compression rate    
    for i, path in tqdm(iterable = enumerate(paths), desc = "Evaluating", total = len(paths)):

        # load in waveform
        waveform = np.load(file = path).astype(np.int32)
        waveform = waveform[:min(n_samples_per_song, len(waveform))] # truncate to number of samples per song
        size_original = waveform.nbytes

        # encode
        compressed = lossless_compressor.encode(waveform)
        size_compressed = lossless_compressor.get_compressed_size(compressed)
        compression_rate = size_original / size_compressed
        
        # decode to ensure losslessness
        waveform_reconstructed = lossless_compressor.decode(compressed)
        assert np.array_equal(waveform, waveform_reconstructed), "Waveform reconstruction is not lossless."

        # write results
        results = pd.DataFrame(data = [dict(zip(OUTPUT_COLUMNS, [path, args.lossless_compressor, args.entropy_coder, size_original, size_compressed, compression_rate]))])
        results.to_csv(path_or_buf = args.output_filepath, mode = "a", header = False, index = False, sep = ",", na_rep = utils.NA_STRING)

    ##################################################


    # REPORT RESULTS
    ##################################################

    # read in results
    results = pd.read_csv(filepath_or_buffer = args.output_filepath, sep = ",", header = 0, index_col = False)
    results = results[(results["lossless_compressor"] == args.lossless_compressor) & (results["entropy_coder"] == args.entropy_coder)]

    # output statistics on compression rate
    compression_rates = np.array(results["compression_rate"]) * 100 # convert to percentages
    print(f"Mean Compression Rate: {np.mean(compression_rates):.2f}%")
    print(f"Median Compression Rate: {np.median(compression_rates):.2f}%")
    print(f"Standard Deviation of Compression Rates: {np.std(compression_rates):.2f}%")
    print(f"Best Compression Rate: {np.max(compression_rates):.2f}%") # best compression rate
    print(f"Worst Compression Rate: {np.min(compression_rates):.2f}%")

    ##################################################


##################################################