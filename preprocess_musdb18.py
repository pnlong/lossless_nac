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
from os import makedirs, mkdir, listdir, close
from os.path import basename, exists, isdir
from shutil import rmtree
from glob import iglob
import os
import soundfile as sf
import tempfile

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

    def _discover_paths():
        """Discover paths: prefer train/test subdirs that contain WAV files; else fall back to **/*.mp4."""
        wav_subdirs = []
        for split in ("train", "test"):
            split_dir = os.path.join(args.musdb18_dir, split)
            if not isdir(split_dir):
                continue
            for name in sorted(listdir(split_dir)):
                subdir = os.path.join(split_dir, name)
                if not isdir(subdir):
                    continue
                wavs = list(iglob("*.wav", root_dir=subdir))
                if wavs:
                    wav_subdirs.append(subdir)
        if wav_subdirs:
            return wav_subdirs
        # fallback: mp4 files anywhere under musdb18_dir
        return [os.path.join(args.musdb18_dir, base) for base in iglob("**/*.mp4", root_dir=args.musdb18_dir, recursive=True)]

    # preprocessing function
    def preprocess(path: str):
        """
        Preprocess one MusDB18 track. path is either:
        - a directory containing WAV files (one per stem), or
        - an MP4 file (Native Instruments stems format).
        """

        # save time by avoiding unnecessary calculations
        if path in already_completed_paths and not args.reset:
            return

        if isdir(path):
            # load stems from WAV files in the directory (sorted by filename)
            wav_files = sorted(iglob("*.wav", root_dir=path))
            if not wav_files:
                return
            stems_list = []
            sample_rate = None
            for f in wav_files:
                fp = os.path.join(path, f)
                wav, sr = sf.read(fp, dtype=np.float64, always_2d=True)
                if sample_rate is None:
                    sample_rate = sr
                elif sr != sample_rate:
                    wav = librosa.resample(y=wav.T, orig_sr=sr, target_sr=sample_rate).T
                # keep (n_samples, n_channels) per stem; stack to (n_stems, n_samples, n_channels) to match stempeg layout
                stems_list.append(wav)
            stems = np.stack(stems_list, axis=0)
            del stems_list
        else:
            # load from single MP4
            stems, sample_rate = stempeg.read_stems(filename=path)
            n_stems = len(stems)

        n_stems = len(stems)

        # resample if necessary
        if sample_rate != args.sample_rate:
            stems = librosa.resample(y=stems, orig_sr=sample_rate, target_sr=args.sample_rate, axis=1)
            sample_rate = args.sample_rate

        # determine stem output paths (path id: dir name or mp4 basename without extension)
        path_id = basename(path.rstrip(os.sep)) if isdir(path) else basename(path)[:-len(".mp4")]
        stem_prefixes = [f"{output_dir}/{path_id}.{i}" for i in range(n_stems)]
        stem_paths = [f"{p}.npy" for p in stem_prefixes]

        # save stems as pickled numpy arrays
        for i, stem_prefix, stem_path in zip(range(n_stems), stem_prefixes, stem_paths):
            wav_fd, wav_filepath = tempfile.mkstemp(suffix=".wav", prefix=f"wav_eval_{basename(stem_prefix)}.")
            close(wav_fd)
            sf.write(file=wav_filepath, data=stems[i], samplerate=sample_rate, format="WAV", subtype=f"PCM_{args.bit_depth}")

            waveform, _ = sf.read(file=wav_filepath, dtype=audio_data_type)
            np.save(file=stem_path, arr=waveform)
            del waveform

        pd.DataFrame(data=dict(zip(
            utils.STEMS_TO_AUDIO_COLUMN_NAMES,
            (stem_paths, utils.rep(x=sample_rate, times=n_stems), utils.rep(x=path, times=n_stems), list(range(n_stems))),
        ))).to_csv(path_or_buf=output_filepath, sep=",", na_rep=utils.NA_STRING, header=False, index=False, mode="a")
        return

    # get musdb18 paths (WAV subdirs under train/test, or **/*.mp4)
    paths = _discover_paths()

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
