# README
# Phillip Long
# July 21, 2025

# Test compression rate of Blocked LDAC. We use the MusDB18 dataset as a testbed.

# IMPORTS
##################################################

import numpy as np
import pandas as pd
import argparse
import multiprocessing
from tqdm import tqdm
from os.path import exists, dirname, basename, getsize
from os import makedirs
import time
import torch
import warnings
import json
import tempfile

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(dirname(dirname(realpath(__file__))))) # for preprocess_musdb18
sys.path.insert(0, f"{dirname(dirname(dirname(realpath(__file__))))}/dac") # import dac package
sys.path.insert(0, dirname(dirname(realpath(__file__)))) # needs to be after so that blocked_lossless_compressors is correctly imported

from blocked_lossless_compressors import ldac
from blocked_lossless_compressors import ldac_compressor
from entropy_coders.factory import get_entropy_coder
from entropy_coders.serialize import serialize
from constants import INPUT_FILEPATH, OUTPUT_DIR, NA_STRING
from preprocess_musdb18 import get_mixes_only_mask, get_test_only_mask
import dac

# ignore deprecation warning from pytorch
warnings.filterwarnings(action = "ignore", message = "torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm")

##################################################


# CONSTANTS
##################################################

OUTPUT_COLUMNS = ["path", "size_original", "size_compressed", "compression_rate", "duration_audio", "duration_encoding", "compression_speed", "model_path", "entropy_coder", "codebook_level", "audio_scale", "block_size", "batch_size", "using_gpu", "total_bits", "metadata_bits", "estimator_bits", "entropy_bits"]

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # SETUP
    ##################################################

    # read in arguments
    def parse_args(args = None, namespace = None):
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(prog = "Evaluate", description = "Evaluate LDAC Implementation") # create argument parser
        parser.add_argument("--input_filepath", type = str, default = INPUT_FILEPATH, help = "Absolute filepath to CSV file describing the preprocessed MusDB18 dataset (see `preprocess_musdb18.py`).")
        parser.add_argument("--output_dir", type = str, default = OUTPUT_DIR, help = "Absolute filepath to the output directory.")
        parser.add_argument("-mp", "--model_path", type = str, default = ldac_compressor.DAC_PATH, help = "Absolute filepath to the Descript Audio Codec model weights.")
        parser.add_argument("--entropy_coder", type = str, default = "adaptive_rice", help = "Entropy coder to use. Please enter in the following format: '<entropy_coder_type>-{\"param\": val, ..., \"param\": val}'")
        parser.add_argument("--codebook_level", type = int, default = 0, help = "Codebook level for DAC model. Use 0 for Adaptive DAC.")
        parser.add_argument("--audio_scale", type = float, default = None, help = "Audio scale. If None, determine the optimal audio scale for each waveform.")
        parser.add_argument("--block_size", type = int, default = ldac_compressor.BLOCK_SIZE_DEFAULT, help = "Block size.")
        parser.add_argument("--batch_size", type = int, default = ldac_compressor.BATCH_SIZE_DEFAULT, help = "Batch size.")
        parser.add_argument("--mixes_only", action = "store_true", help = "Compute statistics for only mixes in MUSDB18, not all stems.")
        parser.add_argument("--reset", action = "store_true", help = "Re-evaluate files.")
        parser.add_argument("-g", "--gpu", type = int, default = -1, help = "GPU (-1 for CPU).")
        parser.add_argument("-j", "--jobs", type = int, default = int(multiprocessing.cpu_count() / 4), help = "Number of workers for multiprocessing.")
        args = parser.parse_args(args = args, namespace = namespace) # parse arguments
        if not exists(args.input_filepath): # ensure input_filepath exists
            raise RuntimeError(f"--input_filepath argument does not exist: {args.input_filepath}")
        elif not exists(args.model_path):
            raise RuntimeError(f"--model_path argument does not exist: {args.model_path}")
        return args # return parsed arguments
    args = parse_args()

    # create output directory if necessary
    if not exists(args.output_dir):
        makedirs(args.output_dir, exist_ok = True)
    output_filepath = f"{args.output_dir}/test_blocked.csv"

    # load descript audio codec
    using_gpu = (torch.cuda.is_available() and args.gpu > -1)
    device = torch.device(f"cuda:{args.gpu}" if using_gpu else "cpu")
    model = dac.DAC.load(location = args.model_path).to(device)
    model.eval() # turn on evaluate mode

    # parse entropy coder
    entropy_coder_type, entropy_coder_kwargs = args.entropy_coder.split("-")
    entropy_coder_kwargs = json.loads(entropy_coder_kwargs) # parse entropy coder kwargs
    entropy_coder = get_entropy_coder(type_ = entropy_coder_type, **entropy_coder_kwargs)
    serialized_entropy_coder = serialize(entropy_coder = entropy_coder)
    
    # write output columns if necessary
    if not exists(output_filepath): # write column names
        pd.DataFrame(columns = OUTPUT_COLUMNS).to_csv(path_or_buf = output_filepath, sep = ",", na_rep = NA_STRING, header = True, index = False, mode = "w")
    results = pd.read_csv(filepath_or_buffer = output_filepath, sep = ",", header = 0, index_col = False)
    results_mask = (
        (results["model_path"] == args.model_path) & 
        (results["entropy_coder"] == serialized_entropy_coder) & 
        (results["codebook_level"] == args.codebook_level) & 
        (results["audio_scale"] == args.audio_scale) & 
        (results["block_size"] == args.block_size) & 
        (results["batch_size"] == args.batch_size) & 
        (results["using_gpu"] == using_gpu))
    if args.reset:
        results = results[~results_mask]
        results.to_csv(path_or_buf = output_filepath, sep = ",", na_rep = NA_STRING, header = True, index = False, mode = "w")
        already_completed_paths = set() # no paths have been already completed
    else: # determine already completed paths
        results = results[results_mask]
        already_completed_paths = set(results["path"])
    del results, results_mask # free up memory

    ##################################################


    # DETERMINE COMPRESSION RATE
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
            
            # load in waveform
            waveform = np.load(file = path)
            size_original = waveform.nbytes # compute size in bytes of original waveform
            waveform = waveform.astype(np.int32) # ensure waveform is stored as int32 for ldac
            sample_rate = sample_rate_by_path[path]

            # create temporary LDAC file
            with tempfile.NamedTemporaryFile(suffix = ".ldac", delete = True) as f:

                # output filepath
                path_compressed = f.name
            
                # encode and decode
                with torch.no_grad():
                    duration_audio = len(waveform) / sample_rate
                    start_time = time.perf_counter()
                    statistics = ldac.encode_to_file(
                        path = path_compressed,
                        data = waveform,
                        entropy_coder = entropy_coder,
                        model = model,
                        sample_rate = sample_rate,
                        codebook_level = args.codebook_level,
                        audio_scale = args.audio_scale,
                        block_size = args.block_size,
                        batch_size = args.batch_size,
                        return_statistics = True,
                    )
                    duration_encoding = time.perf_counter() - start_time # measure speed of compression
                    round_trip = ldac.decode_from_file(
                        path = path_compressed,
                        model = model,
                    ) # reconstruct waveform from bottleneck to ensure losslessness
                    assert np.array_equal(waveform, round_trip), "Original and reconstructed waveforms do not match. The encoding is lossy."
                    del round_trip, start_time # free up memory                

                # compute size in bytes of compressed bottleneck
                size_compressed = getsize(path_compressed)

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
                    "model_path": args.model_path,
                    "entropy_coder": serialized_entropy_coder,
                    "codebook_level": args.codebook_level,
                    "audio_scale": args.audio_scale,
                    "block_size": args.block_size,
                    "batch_size": args.batch_size,
                    "using_gpu": using_gpu,
                    "total_bits": statistics["total_bits"],
                    "metadata_bits": statistics["metadata_bits"],
                    "estimator_bits": statistics["estimator_bits"],
                    "entropy_bits": statistics["entropy_bits"],
                }]).to_csv(path_or_buf = output_filepath, sep = ",", na_rep = NA_STRING, header = False, index = False, mode = "a")

            return

        # evaluate over testbed
        if using_gpu: # cannot use multiprocessing with GPU
            for path in tqdm(iterable = paths, desc = "Evaluating", total = len(paths)):
                _ = evaluate(path = path)
        else: # we can use multiprocessing if not using GPU
            with multiprocessing.Pool(processes = args.jobs) as pool:
                _ = list(tqdm(iterable = pool.imap_unordered(
                        func = evaluate,
                        iterable = paths,
                        chunksize = 1,
                    ),
                    desc = "Evaluating",
                    total = len(paths),
                ))
        
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
    results = results[
        (results["model_path"] == args.model_path) & 
        (results["entropy_coder"] == serialized_entropy_coder) & 
        (results["codebook_level"] == args.codebook_level) & 
        (results["audio_scale"] == args.audio_scale) & 
        (results["block_size"] == args.block_size) & 
        (results["batch_size"] == args.batch_size) & 
        (results["using_gpu"] == using_gpu)
    ]
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

    # output statistics on bitrate
    bitrates = (results["size_compressed"] * 8) / results["duration_audio"]
    print(f"Mean Bitrate: {np.mean(bitrates):.2f} bps")
    print(f"Median Bitrate: {np.median(bitrates):.2f} bps")
    print(f"Standard Deviation of Bitrates: {np.std(bitrates):.2f} bps")
    print(f"Best Bitrate: {np.max(bitrates):.2f} bps")
    print(f"Worst Bitrate: {np.min(bitrates):.2f} bps")
    print("-" * 60)

    # output statistics on lossy bitrate
    lossy_bitrates = results["estimator_bits"] / results["duration_audio"]
    print(f"Mean Lossy Bitrate: {np.mean(lossy_bitrates):.2f} bps")
    print(f"Median Lossy Bitrate: {np.median(lossy_bitrates):.2f} bps")
    print(f"Standard Deviation of Lossy Bitrates: {np.std(lossy_bitrates):.2f} bps")
    print(f"Best Lossy Bitrate: {np.max(lossy_bitrates):.2f} bps")
    print(f"Worst Lossy Bitrate: {np.min(lossy_bitrates):.2f} bps")
    print("-" * 60)

    # output statistics on different bit types
    print(f"Mean Metadata Bits Proportion: {100 * np.mean(results['metadata_bits'] / results['total_bits']):.2f}%")
    print(f"Mean Estimator Bits Proportion: {100 * np.mean(results['estimator_bits'] / results['total_bits']):.2f}%")
    print(f"Mean Entropy Bits Proportion: {100 * np.mean(results['entropy_bits'] / results['total_bits']):.2f}%")
    print("-" * 60)

    ##################################################

##################################################