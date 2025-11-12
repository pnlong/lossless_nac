# Phillip Long
# 11/6/2025
# flac_eval.py

# IMPORTS
##################################################

import argparse
import sys
from os.path import getsize, exists
import multiprocessing
from tqdm import tqdm
import logging
import numpy as np
from typing import Tuple
import tempfile
import pandas as pd
import datetime
import subprocess
import warnings

from flac_eval import get_dataset, get_dataset_choices

##################################################


# CONSTANTS
##################################################

# default output filepath
DEFAULT_OUTPUT_FILEPATH = "/home/pnlong/lnac/flac_eval_results_fixed.csv"

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # SETUP
    ##################################################

    # parse arguments
    def parse_args(args = None, namespace = None):
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(prog = "FLAC Evaluation", description = "Evalute FLAC Compression.") # create argument parser
        parser.add_argument("--dataset", type = str, required = True, choices = get_dataset_choices(), help = "Dataset to evaluate.")
        parser.add_argument("--output_filepath", type = str, default = DEFAULT_OUTPUT_FILEPATH, help = "Absolute filepath (CSV file) to append the evaluation results to.")
        parser.add_argument("--jobs", type = int, default = int(multiprocessing.cpu_count() / 4), help = "Number of workers for multiprocessing.")
        parser.add_argument("--reset", action = "store_true", help = "Reset the output file.")
        args = parser.parse_args(args = args, namespace = namespace) # parse arguments
        return args # return parsed arguments
    args = parse_args()

    # set up logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")
    console_handler = logging.StreamHandler(stream = sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # set up output file if necessary, writing column names
    if not exists(args.output_filepath) or args.reset:
        pd.DataFrame(columns = [
            "dataset",
            "total_size",
            "compressed_size",
            "overall_compression_rate",
            "mean_compression_rate",
            "median_compression_rate",
            "std_compression_rate",
            "max_compression_rate",
            "min_compression_rate",
            "datetime",
        ]).to_csv(path_or_buf = args.output_filepath, sep = ",", na_rep = "NA", header = True, index = False, mode = "w")

    # get dataset
    dataset = get_dataset(dataset_name = args.dataset)
    
    # log some information about the dataset
    dataset_name = f" {dataset.name.upper()}, {dataset.bit_depth}-bit " # add spaces on side so it looks nicer
    line_character, line_width = "=", 100
    logger.info(f"{dataset_name:{line_character}^{line_width}}") # print dataset name with equal signs
    logger.info(f"Running Command: python {' '.join(sys.argv)}")
    logger.info(f"Dataset: {dataset.get_description()}")

    ##################################################


    # HELPER FUNCTION FOR EVALUATING COMPRESSION RATE
    ##################################################

    def evaluate(index: int) -> Tuple[int, int]:
        """
        Evaluate the compression rate for the item at the given index.
        
        Parameters:
            index: int - The index of the item to evaluate.

        Returns:
            Tuple[int, int] - The raw size and compressed size (both in bytes) of the audio file.
        """

        # create temporary directory
        with tempfile.TemporaryDirectory() as tmp_dir:

            # get path
            path = dataset.paths[index]

            try:

                # if file is stored compressed
                if path.endswith(".flac"):
                    compressed_size = getsize(path)
                    wav_filepath = f"{tmp_dir}/reconstructed.wav"
                    _ = subprocess.run(args = [
                            "flac",
                            "-d",
                            "-o", wav_filepath,
                            "--force",
                            path,
                        ], 
                        check = True,
                        stdout = subprocess.DEVNULL,
                        stderr = subprocess.DEVNULL,
                    )
                    raw_size = getsize(wav_filepath)
                
                # if file is stored uncompressed
                elif path.endswith(".wav"):
                    raw_size = getsize(path)
                    flac_filepath = f"{tmp_dir}/compressed.flac"
                    _ = subprocess.run(args = [
                            "flac",
                            "-o", flac_filepath,
                            "--force",
                            path,
                        ], 
                        check = True,
                        stdout = subprocess.DEVNULL,
                        stderr = subprocess.DEVNULL,
                    )
                    compressed_size = getsize(flac_filepath)

                # otherwise, raise an error
                else:
                    raise ValueError(f"File {path} is not a WAV or FLAC file.")

            except Exception as e:
                warnings.warn(f"Error evaluating compression rate for {path}: {e}", category = RuntimeWarning)
                return 0, 0 # does not contribute to statistics

        # return raw size and compressed size
        return raw_size, compressed_size

    ##################################################


    # EVALUATE COMPRESSION RATE
    ##################################################

    # use multiprocessing to evaluate compression rate
    with multiprocessing.Pool(processes = args.jobs) as pool:
        results = list(tqdm(iterable = pool.imap_unordered(
            func = evaluate,
            iterable = range(len(dataset)),
            chunksize = 1,
        ), desc = "Evaluating", total = len(dataset)))
        raw_sizes, compressed_sizes = list(map(np.array, zip(*results)))
        mask = np.logical_and(raw_sizes != 0, compressed_sizes != 0)
        raw_sizes = raw_sizes[mask]
        compressed_sizes = compressed_sizes[mask]
        del results, mask

    # calculate statistics
    total_size = np.sum(raw_sizes)
    compressed_size = np.sum(compressed_sizes)
    overall_compression_rate = total_size / compressed_size
    compression_rates = raw_sizes / compressed_sizes
    mean_compression_rate = np.mean(compression_rates)
    median_compression_rate = np.median(compression_rates)
    std_compression_rate = np.std(compression_rates)
    max_compression_rate = np.max(compression_rates)
    min_compression_rate = np.min(compression_rates)

    # output evaluation results
    logger.info(f"Total Size: {total_size} bytes")
    logger.info(f"Compressed Size: {compressed_size} bytes")
    logger.info(f"Overall Compression Rate: {overall_compression_rate:.2f}x ({1 / overall_compression_rate:.2%})")
    logger.info(f"Mean Compression Rate: {mean_compression_rate:.2f}x ({1 / mean_compression_rate:.2%})")
    logger.info(f"Median Compression Rate: {median_compression_rate:.2f}x ({1 / median_compression_rate:.2%})")
    logger.info(f"Standard Deviation of Compression Rate: {std_compression_rate:.2f}x ({1 / std_compression_rate:.2%})")
    logger.info(f"Maximum Compression Rate: {max_compression_rate:.2f}x ({1 / max_compression_rate:.2%})")
    logger.info(f"Minimum Compression Rate: {min_compression_rate:.2f}x ({1 / min_compression_rate:.2%})")
    logger.info("") # log empty line

    # write evaluation results to output file
    pd.DataFrame(data = [{
        "dataset": dataset.name,
        "total_size": total_size,
        "compressed_size": compressed_size,
        "overall_compression_rate": overall_compression_rate,
        "mean_compression_rate": mean_compression_rate,
        "median_compression_rate": median_compression_rate,
        "std_compression_rate": std_compression_rate,
        "max_compression_rate": max_compression_rate,
        "min_compression_rate": min_compression_rate,
        "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), # current datetime
    }]).to_csv(path_or_buf = args.output_filepath, sep = ",", na_rep = "NA", header = False, index = False, mode = "a")

    ##################################################

##################################################
