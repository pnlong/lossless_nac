# README
# Phillip Long
# June 21, 2025

# Test compression rate of IFLAC encoder. We use the MusDB18 dataset as a testbed.

# IMPORTS
##################################################

import numpy as np
import pandas as pd
import argparse
import multiprocessing
from tqdm import tqdm
from os.path import exists, dirname
from os import makedirs
import time

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

import utils
from lossless_compressors import iflac

##################################################


# CONSTANTS
##################################################

OUTPUT_COLUMNS = utils.TEST_COMPRESSION_COLUMN_NAMES + ["block_size", "interchannel_decorrelate"]

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # SETUP
    ##################################################

    # read in arguments
    def parse_args(args = None, namespace = None):
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(prog = "Evaluate", description = "Evaluate IFLAC Implementation") # create argument parser
        parser.add_argument("--input_filepath", type = str, default = f"{utils.MUSDB18_PREPROCESSED_DIR}-44100/data.csv", help = "Absolute filepath to CSV file describing the preprocessed MusDB18 dataset (see `preprocess_musdb18.py`).")
        parser.add_argument("--output_dir", type = str, default = f"{utils.EVAL_DIR}/iflac", help = "Absolute filepath to the output directory.")
        parser.add_argument("--block_size", type = int, default = utils.BLOCK_SIZE, help = "Block size.")
        parser.add_argument("--no_interchannel_decorrelate", action = "store_true", help = "Turn off interchannel-decorrelation.")
        parser.add_argument("--reset", action = "store_true", help = "Re-evaluate files.")
        parser.add_argument("-j", "--jobs", type = int, default = int(multiprocessing.cpu_count() / 4), help = "Number of workers for multiprocessing.")
        args = parser.parse_args(args = args, namespace = namespace) # parse arguments
        args.interchannel_decorrelate = not args.no_interchannel_decorrelate # infer interchannel decorrelation
        if not exists(args.input_filepath): # ensure input_filepath exists
            raise RuntimeError(f"--input_filepath argument does not exist: {args.input_filepath}")
        return args # return parsed arguments
    args = parse_args()

    # create output directory if necessary
    if not exists(args.output_dir):
        makedirs(args.output_dir, exist_ok = True)
    output_filepath = f"{args.output_dir}/test.csv"
    
    # write output columns if necessary
    if not exists(output_filepath) or args.reset: # write column names
        pd.DataFrame(columns = OUTPUT_COLUMNS).to_csv(path_or_buf = output_filepath, sep = ",", na_rep = utils.NA_STRING, header = True, index = False, mode = "w")
        already_completed_paths = set() # no paths have been already completed
    else: # determine already completed paths
        results = pd.read_csv(filepath_or_buffer = output_filepath, sep = ",", header = 0, index_col = False)
        results = results[(results["block_size"] == args.block_size) & (results["interchannel_decorrelate"] == args.interchannel_decorrelate)]
        already_completed_paths = set(results["path"])
        del results # free up memory

    ##################################################


    # DETERMINE COMPRESSION RATE
    ##################################################

    # read in paths to evaluate
    sample_rate_by_path = pd.read_csv(filepath_or_buffer = args.input_filepath, sep = ",", header = 0, index_col = False, usecols = utils.INPUT_COLUMN_NAMES)
    sample_rate_by_path = sample_rate_by_path.set_index(keys = "path", drop = True)["sample_rate"].to_dict() # dictionary where keys are paths and values are sample rates of those paths
    paths = list(sample_rate_by_path.keys()) # get paths to NPY audio files

    # helper function for determining compression rate
    def evaluate(path: str):
        """
        Determine compression rate given the absolute filepath to an input audio file (stored as a pickled numpy object, NPY).
        Expects the input audio to be of shape (n_samples, n_channels) for multi-channel audio or (n_samples,) for mono audio.
        """

        # save time by avoiding unnecessary calculations
        if path in already_completed_paths and not args.reset:
            return # return nothing, stop execution here
        
        # load in waveform
        waveform = np.load(file = path)
        sample_rate = sample_rate_by_path[path]
        
        # assertions
        assert sample_rate == utils.SAMPLE_RATE, f"{path} audio has a sample rate of {sample_rate:,} Hz, but must have a sample rate of {utils.SAMPLE_RATE:,} Hz." # ensure sample rate is correct
        assert waveform.ndim <= 2, f"Input audio must be of shape (n_samples, n_channels) for multi-channel audio or (n_samples,) for mono audio, but {path} has shape {tuple(waveform.shape)}."
        if waveform.ndim == 2:
            assert waveform.shape[-1] <= 2, f"Multichannel-audio must have either one or two channels, but {path} has {waveform.shape[-1]} channels."
        assert any(waveform.dtype == dtype for dtype in utils.VALID_AUDIO_DTYPES), f"Audio must be stored as a numpy signed integer data type, but found {waveform.dtype}."

        # encode and decode
        duration_audio = len(waveform) / sample_rate
        start_time = time.perf_counter()
        bottleneck = iflac.encode(
            waveform = waveform, block_size = args.block_size, interchannel_decorrelate = args.interchannel_decorrelate,
            log_for_zach_kwargs = {"duration": duration_audio, "lossless_compressor": "iflac", "parameters": {"block_size": args.block_size, "interchannel_decorrelate": args.interchannel_decorrelate}, "path": path}, # arguments to log for zach
        ) # compute compressed bottleneck
        duration_encoding = time.perf_counter() - start_time # measure speed of compression
        round_trip = iflac.decode(bottleneck = bottleneck, interchannel_decorrelate = args.interchannel_decorrelate) # reconstruct waveform from bottleneck to ensure losslessness
        assert np.array_equal(waveform, round_trip), "Original and reconstructed waveforms do not match. The encoding is lossy."
        del round_trip, start_time # free up memory

        # compute size in bytes of original waveform
        size_original = utils.get_waveform_size(waveform = waveform)

        # compute size in bytes of compressed bottleneck
        size_compressed = iflac.get_bottleneck_size(bottleneck = bottleneck)

        # compute other final statistics
        compression_rate = utils.get_compression_rate(size_original = size_original, size_compressed = size_compressed)
        compression_speed = utils.get_compression_speed(duration_audio = duration_audio, duration_encoding = duration_encoding) # speed is inversely related to duration

        # output
        pd.DataFrame(data = [dict(zip(
            OUTPUT_COLUMNS, 
            (path, size_original, size_compressed, compression_rate, duration_audio, duration_encoding, compression_speed, args.block_size, args.interchannel_decorrelate)
        ))]).to_csv(path_or_buf = output_filepath, sep = ",", na_rep = utils.NA_STRING, header = False, index = False, mode = "a")

        # return nothing
        return

    # use multiprocessing
    postfix = {
        "Block Size": f"{args.block_size}",
        "Interchannel Decorrelate": str(args.interchannel_decorrelate),
    }
    with multiprocessing.Pool(processes = args.jobs) as pool:
        _ = list(tqdm(iterable = pool.imap_unordered(
                func = evaluate,
                iterable = paths,
                chunksize = utils.CHUNK_SIZE,
            ),
            desc = "Evaluating",
            total = len(paths),
            postfix = postfix))
        
    # free up memory
    del already_completed_paths, paths, sample_rate_by_path, postfix
        
    ##################################################
        
    # FINAL STATISTICS
    ##################################################

    # read in results (just the compression rate column, we don't really care about anything else)
    results = pd.read_csv(filepath_or_buffer = output_filepath, sep = ",", header = 0, index_col = False)
    results = results[(results["block_size"] == args.block_size) & (results["interchannel_decorrelate"] == args.interchannel_decorrelate)]
    compression_rates = results["compression_rate"].to_numpy() * 100 # convert to percentages
    compression_speeds = results["compression_speed"].to_numpy()

    # output statistics on compression rate
    print(f"Mean Compression Rate: {np.mean(compression_rates):.2f}%")
    print(f"Median Compression Rate: {np.median(compression_rates):.2f}%")
    print(f"Standard Deviation of Compression Rates: {np.std(compression_rates):.2f}%")
    print(f"Best Compression Rate: {np.min(compression_rates):.2f}%")
    print(f"Worst Compression Rate: {np.max(compression_rates):.2f}%")

    # output statistics on compression speed
    print(f"Mean Compression Speed: {np.mean(compression_speeds):.2f}")
    print(f"Median Compression Speed: {np.median(compression_speeds):.2f}")
    print(f"Standard Deviation of Compression Speeds: {np.std(compression_speeds):.2f}")
    print(f"Best Compression Speed: {np.max(compression_speeds):.2f}")
    print(f"Worst Compression Speed: {np.min(compression_speeds):.2f}")

    ##################################################

##################################################