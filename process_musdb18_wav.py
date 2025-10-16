# README
# Phillip Long
# August 16, 2025

# Convert MUSDB18 NPY files to WAV files.

# IMPORTS
##################################################

import pandas as pd
from os.path import exists, basename, dirname
from typing import List
from os import makedirs, mkdir, listdir
from shutil import rmtree
import argparse
import multiprocessing
import numpy as np
import scipy.io.wavfile
from tqdm import tqdm
from random import sample

##################################################


# CONSTANTS
##################################################

# filepaths
INPUT_FILEPATH = "/deepfreeze/pnlong/lnac/test_data/musdb18_preprocessed-44100/data.csv"
OUTPUT_DIR = "/deepfreeze/pnlong/lnac/sashimi/data/musdb18mono"

# clip length
CLIP_LENGTH = 60 * 44100 # make each clip 60 seconds
PAD_CLIPS_TO_FIXED_LENGTH = False # pad clips so they are all clip length

# train partition proportion
TRAIN_PARTITION_PROPORTION = 0.8

# channel names
CHANNEL_NAMES = ("left", "right")

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # SETUP
    ##################################################

    # parse arguments
    def parse_args(args = None, namespace = None):
        """
        Parse command-line arguments for converting residuals to WAV files.
        
        Parameters
        ----------
        args : list, optional
            List of argument strings to parse, by default None (uses sys.argv)
        namespace : argparse.Namespace, optional
            Namespace object to populate with parsed arguments, by default None
            
        Returns
        -------
        argparse.Namespace
            Parsed arguments containing paths and options for expression text extraction
            
        Raises
        ------
        FileNotFoundError
            If the specified PDMX file does not exist
        """
        parser = argparse.ArgumentParser(prog = "Convert Residuals to WAV Files", description = "Convert residuals to WAV files.") # create argument parser
        parser.add_argument("--input_filepath", type = str, default = INPUT_FILEPATH, help = "Path to input file.")
        parser.add_argument("--output_dir", type = str, default = OUTPUT_DIR, help = "Path to output directory.")
        parser.add_argument("--mono", action = "store_true", help = "Convert to mono, writing a separate WAV file for each channel of each song. If not provided, the default is to write a single WAV file for each song, including both channels.")
        parser.add_argument("--clip_length", type = int, default = None, help = "Length of each clip in seconds.")
        parser.add_argument("--pad_clips_to_fixed_length", action = "store_true", help = "Pad clips so they are all clip length.")
        parser.add_argument("--jobs", type = int, default = int(multiprocessing.cpu_count() / 4), help = "Number of jobs to run in parallel.")
        parser.add_argument("--reset", action = "store_true", help = "Reset the output directory.")
        args = parser.parse_args(args = args, namespace = namespace) # parse arguments
        if not exists(args.input_filepath):
            raise FileNotFoundError(f"Input file not found: {args.input_filepath}")
        return args # return parsed arguments
    args = parse_args()

    # read in input csv
    print("Reading in input data...")
    dataset = pd.read_csv(filepath_or_buffer = args.input_filepath, sep = ",", header = 0, index_col = False)
    dataset = dataset[["path", "sample_rate"]] # get only the path and sample rate columns, as that is all we care about
    assert len(dataset) > 0, "Dataset is empty."
    print(f"Completed reading in input data.")

    # manage partitions
    get_song_name = lambda path: basename(path).split(".")[0] # get the name of the song
    get_song_stem_index = lambda path: int(basename(path).split(".")[-2]) # get the index of the song stem (0 for mix, 1-4 for stems)
    songs = list(set(map(get_song_name, dataset["path"])))
    train_songs = set(sample(population = songs, k = int(len(songs) * TRAIN_PARTITION_PROPORTION)))
    print(f"Partitioning into train and valid sets...")
    print(f"Number of songs: {len(songs)}")
    print(f"Number of train songs: {len(train_songs)}")
    print(f"Number of valid songs: {len(songs) - len(train_songs)}")
    dataset["is_train"] = dataset["path"].apply(get_song_name).apply(lambda song_name: song_name in train_songs)

    # create output directory
    print("Creating output directory...")
    if exists(args.output_dir) and args.reset:
        print(f"Removing old output directory {args.output_dir} because --reset was called...")
        rmtree(args.output_dir)
        print(f"Removed old output directory {args.output_dir}.")
    if not exists(args.output_dir):
        makedirs(args.output_dir, exist_ok = True)
    train_dir = f"{args.output_dir}/train"
    valid_dir = f"{args.output_dir}/valid"
    for directory in (train_dir, valid_dir):
        if not exists(directory):
            mkdir(directory)
    mix_mapping_path = f"{args.output_dir}/mixes.csv"
    print("Created output directory.")

    # if clip length is not provided, don't partition into clips
    use_clips = args.clip_length is not None

    ##################################################


    # POPULATE OUTPUT SUBDIRECTORIES
    ##################################################

    # convert files
    def convert_helper(i: int) -> List[str]:
        """
        Use the information in a row of the dataset to convert a residual to a WAV file, as well as the original input file.

        Parameters
        ----------
        i : int
            Index of the row of the dataset to convert.

        Returns
        -------
        List[str]
            List of file paths for all generated files
        """

        # get data
        path, sample_rate, is_train = dataset.loc[i]
        data = np.load(path) # loads in shape (n_samples, 2)
        data = data.astype(np.int16) # ensure type is correct
        n_samples, _ = data.shape

        # track generated files
        generated_files = []

        # get output path
        song_name, song_stem_index = get_song_name(path), get_song_stem_index(path)
        output_path_prefix = f"{train_dir if is_train else valid_dir}/{song_name}.{song_stem_index}"

        # if using clips, partition into clips
        if use_clips:
            for clip_idx, start_index in enumerate(range(0, n_samples, args.clip_length)): # separate channel into clips
                end_index = min(start_index + args.clip_length, n_samples) # get end index
                n_samples_in_clip = end_index - start_index # get number of samples in clip
                clip = data[start_index:end_index] # get clip
                if args.pad_clips_to_fixed_length and n_samples_in_clip < args.clip_length: # pad if necessary
                    clip = np.pad(array = clip, pad_width = [(0, args.clip_length - n_samples_in_clip), (0, 0)], mode = "constant") # end pad with zeros
                clip_output_path_prefix = f"{output_path_prefix}.{clip_idx}"
                if args.mono: # mono
                    for channel, channel_name in zip(clip.T, CHANNEL_NAMES):
                        filename = f"{clip_output_path_prefix}.{channel_name}.wav"
                        if not exists(filename) or args.reset:
                            scipy.io.wavfile.write( # write WAV file
                                filename = filename,
                                rate = sample_rate,
                                data = channel,
                            )
                        generated_files.append(filename)
                else: # stereo
                    filename = f"{clip_output_path_prefix}.wav"
                    if not exists(filename) or args.reset:
                        scipy.io.wavfile.write( # write WAV file
                            filename = filename,
                            rate = sample_rate,
                            data = clip,
                        )
                    generated_files.append(filename)

        # otherwise, just write the channel as a mono file
        else:
            if args.mono:
                for channel, channel_name in zip(data.T, CHANNEL_NAMES):
                    filename = f"{output_path_prefix}.{channel_name}.wav"
                    if not exists(filename) or args.reset:
                        scipy.io.wavfile.write( # write WAV file
                            filename = filename,
                            rate = sample_rate,
                            data = channel,
                        )
                    generated_files.append(filename)
            else:
                filename = f"{output_path_prefix}.wav"
                if not exists(filename) or args.reset:
                    scipy.io.wavfile.write( # write WAV file
                        filename = filename,
                        rate = sample_rate,
                        data = data,
                    )
                generated_files.append(filename)

        return generated_files

    # use multiprocessing
    print("Converting NPY Files to WAV Files...")
    with multiprocessing.Pool(processes = args.jobs) as pool:
        output_paths = sum(list(tqdm(iterable = pool.imap_unordered(
            func = convert_helper,
            iterable = dataset.index,
            chunksize = 1,
        ), desc = "Converting NPY Files to WAV Files", total = len(dataset))), [])
    print("Completed converting NPY Files to WAV Files.")

    # create table mapping files to mix status
    print("Creating mix mapping table...")
    mix_mapping = pd.DataFrame(data = {
        "path": list(map(lambda path: f"./{path[len(args.output_dir) + 1:]}", output_paths)), # use basename to get simple file names
        "is_train": list(map(lambda path: basename(dirname(path)) == "train", output_paths)),
        "is_mix": list(map(lambda path: basename(path).split(".")[1] == "0", output_paths)),
    })
    mix_mapping.to_csv(path_or_buf = mix_mapping_path, sep = ",", header = True, index = False)
    print(f"Created mix mapping table: {mix_mapping_path}")

    # stats
    n_files = len(mix_mapping)
    n_mixes = mix_mapping['is_mix'].sum()
    n_train_files = mix_mapping['is_train'].sum()
    n_valid_files = (~mix_mapping['is_train']).sum()
    actual_n_train_files, actual_n_valid_files = len(listdir(train_dir)), len(listdir(valid_dir))
    print(f"Total number of files: {n_files}")
    print(f"Total number of mixes: {n_mixes}")
    print(f"Total number of train files: {n_train_files}")
    print(f"  (Sanity Check) Number of files in train directory: {actual_n_train_files}")
    print(f"Total number of valid files: {n_valid_files}")
    print(f"  (Sanity Check) Number of files in valid directory: {actual_n_valid_files}")

    ##################################################

##################################################