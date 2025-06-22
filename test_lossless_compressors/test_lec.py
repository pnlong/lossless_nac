# README
# Phillip Long
# June 12, 2025

# Test compression rate of naive-LEC encoder. We use the MusDB18 dataset as a testbed.

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
import torch
import warnings

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))
sys.path.insert(0, f"{dirname(dirname(realpath(__file__)))}/encodec") # import encodec package

import utils
from lossless_compressors import lec
import encodec

# ignore deprecation warning from pytorch
warnings.filterwarnings(action = "ignore", message = "torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm")

##################################################


# CONSTANTS
##################################################

OUTPUT_COLUMNS = utils.TEST_COMPRESSION_COLUMN_NAMES + ["block_size", "target_bandwidth", "gpu"]

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # SETUP
    ##################################################

    # read in arguments
    def parse_args(args = None, namespace = None):
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(prog = "Evaluate", description = "Evaluate Naive-LEC Implementation") # create argument parser
        parser.add_argument("--input_filepath", type = str, default = f"{utils.MUSDB18_PREPROCESSED_DIR}-48000/data.csv", help = "Absolute filepath to CSV file describing the preprocessed MusDB18 dataset (see `preprocess_musdb18.py`).")
        parser.add_argument("--output_dir", type = str, default = f"{utils.EVAL_DIR}/lec", help = "Absolute filepath to the output directory.")
        parser.add_argument("--target_bandwidth", type = float, default = lec.TARGET_BANDWIDTH, choices = lec.POSSIBLE_ENCODEC_TARGET_BANDWIDTHS, help = "Target bandwidth for EnCodec model. The number of codebooks used will be determined by the bandwidth selected (see https://github.com/facebookresearch/encodec#:~:text=The%20number%20of%20codebooks%20used%20will%20be%20determined%20bythe%20bandwidth%20selected.).")
        parser.add_argument("--block_size", type = int, default = utils.BLOCK_SIZE, help = "Block size.") # int(model.sample_rate * 0.99) # the 48 kHz encodec model processes audio in one-second chunks with 1% overlap
        parser.add_argument("--reset", action = "store_true", help = "Re-evaluate files.")
        parser.add_argument("-g", "--gpu", type = int, default = -1, help = "GPU (-1 for CPU).")
        parser.add_argument("-j", "--jobs", type = int, default = int(multiprocessing.cpu_count() / 4), help = "Number of workers for multiprocessing.")
        args = parser.parse_args(args = args, namespace = namespace) # parse arguments
        return args # return parsed arguments
    args = parse_args()

    # ensure input_filepath exists
    if not exists(args.input_filepath):
        raise RuntimeError(f"--input_filepath argument does not exist: {args.input_filepath}")
    
    # create output directory if necessary
    if not exists(args.output_dir):
        makedirs(args.output_dir, exist_ok = True)
    output_filepath = f"{args.output_dir}/test.csv"

    # load descript audio codec
    using_gpu = torch.cuda.is_available() and args.gpu != -1
    device = torch.device(f"cuda:{abs(args.gpu)}" if using_gpu else "cpu")
    model = encodec.EncodecModel.encodec_model_48khz().to(device)
    model.set_target_bandwidth(bandwidth = args.target_bandwidth)
    model.eval() # turn on evaluate mode
    
    # write output columns if necessary
    if not exists(output_filepath) or args.reset: # write column names
        pd.DataFrame(columns = OUTPUT_COLUMNS).to_csv(path_or_buf = output_filepath, sep = ",", na_rep = utils.NA_STRING, header = True, index = False, mode = "w")
        already_completed_paths = set() # no paths have been already completed
    else: # determine already completed paths
        already_completed_paths = pd.read_csv(filepath_or_buffer = output_filepath, sep = ",", header = 0, index_col = False)
        already_completed_paths = already_completed_paths[(already_completed_paths["block_size"] == args.block_size) & (already_completed_paths["target_bandwidth"] == args.target_bandwidth) & (already_completed_paths["gpu"] == using_gpu)]
        already_completed_paths = set(already_completed_paths["path"])

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
        assert sample_rate == model.sample_rate, f"{path} audio has a sample rate of {sample_rate:,} Hz, but must have a sample rate of {model.sample_rate:,} Hz to be compatible with the Descript Audio Codec." # ensure sample rate is correct
        assert waveform.ndim <= 2, f"Input audio must be of shape (n_samples, n_channels) for multi-channel audio or (n_samples,) for mono audio, but {path} has shape {tuple(waveform.shape)}."
        if waveform.ndim == 2:
            assert waveform.shape[-1] == 2, f"Multichannel-audio must have two channels, but {path} has {waveform.shape[-1]} channels."
        assert any(waveform.dtype == dtype for dtype in utils.VALID_AUDIO_DTYPES), f"Audio must be stored as a numpy signed integer data type, but found {waveform.dtype}."

        # encode and decode
        with torch.no_grad():
            duration_audio = len(waveform) / sample_rate
            start_time = time.perf_counter()
            bottleneck = lec.encode(
                waveform = waveform, sample_rate = sample_rate, model = model, device = device, block_size = args.block_size,
                log_for_zach_kwargs = {"duration": duration_audio, "lossless_compressor": "lec", "parameters": {"block_size": args.block_size, "target_bandwidth": args.target_bandwidth, "gpu": using_gpu}, "path": path}, # arguments to log for zach
            ) # compute compressed bottleneck
            duration_encoding = time.perf_counter() - start_time # measure speed of compression
            round_trip = lec.decode(bottleneck = bottleneck, model = model, device = device) # reconstruct waveform from bottleneck to ensure losslessness
            assert np.array_equal(waveform, round_trip), "Original and reconstructed waveforms do not match. The encoding is lossy."
            del round_trip, start_time # free up memory

        # compute size in bytes of original waveform
        size_original = utils.get_waveform_size(waveform = waveform)

        # compute size in bytes of compressed bottleneck
        size_compressed = lec.get_bottleneck_size(bottleneck = bottleneck)

        # compute other final statistics
        compression_rate = utils.get_compression_rate(size_original = size_original, size_compressed = size_compressed)
        compression_speed = utils.get_compression_speed(duration_audio = duration_audio, duration_encoding = duration_encoding) # speed is inversely related to duration

        # output
        pd.DataFrame(data = [dict(zip(
            OUTPUT_COLUMNS, 
            (path, size_original, size_compressed, compression_rate, duration_audio, duration_encoding, compression_speed, args.block_size, args.target_bandwidth, using_gpu)
        ))]).to_csv(path_or_buf = output_filepath, sep = ",", na_rep = utils.NA_STRING, header = False, index = False, mode = "a")

        # return nothing
        return

    # evaluate over testbed
    postfix = {
        "Block Size": f"{args.block_size}",
        "Target Bandwidth": f"{args.target_bandwidth}",
        "Using GPU": str(using_gpu),
    }
    if using_gpu: # cannot use multiprocessing with GPU
        for path in tqdm(iterable = paths, desc = "Evaluating", total = len(paths), postfix = postfix):
            _ = evaluate(path = path)
    else: # we can use multiprocessing if not using GPU
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
    del already_completed_paths, paths, sample_rate_by_path, using_gpu, postfix
        
    ##################################################
        
    # FINAL STATISTICS
    ##################################################

    # read in results (just the compression rate column, we don't really care about anything else)
    results = pd.read_csv(filepath_or_buffer = output_filepath, sep = ",", header = 0, index_col = False)
    results = results[(results["block_size"] == args.block_size) & (results["target_bandwidth"] == args.target_bandwidth) & (results["gpu"] == using_gpu)]
    compression_rates = results["compression_rate"].to_numpy() * 100 # convert to percentages
    compression_speeds = results["compression_speed"].to_numpy()

    # output statistics on compression rate
    print(f"Mean Compression Rate: {np.mean(compression_rates):.2f}%")
    print(f"Median Compression Rate: {np.median(compression_rates):.2f}%")
    print(f"Standard Deviation of Compression Rates: {np.std(compression_rates):.2f}%")
    print(f"Best Compression Rate: {np.min(compression_rates):.2f}%")
    print(f"Worst Compression Rate: {np.max(compression_rates):.2f}%")

    # output statistics on compression speed
    print(f"Mean Compression Speed: {np.mean(compression_speeds):.2f}%")
    print(f"Median Compression Speed: {np.median(compression_speeds):.2f}%")
    print(f"Standard Deviation of Compression Speeds: {np.std(compression_speeds):.2f}%")
    print(f"Best Compression Speed: {np.max(compression_speeds):.2f}%")
    print(f"Worst Compression Speed: {np.min(compression_speeds):.2f}%")

    ##################################################

##################################################