# Phillip Long
# 11/6/2025
# flac_eval.py

# IMPORTS
##################################################

import argparse
import sys
from os.path import getsize, exists, basename, dirname
import multiprocessing
from tqdm import tqdm
import logging
import numpy as np
from typing import List, Tuple
import tempfile
import soundfile as sf
import pydub
import pandas as pd
import glob
import datetime
import io
import subprocess

##################################################


# CONSTANTS
##################################################

# default output filepath
DEFAULT_OUTPUT_FILEPATH = "/home/pnlong/lnac/flac_eval_results.csv"

# FLAC compression level
DEFAULT_FLAC_COMPRESSION_LEVEL = 5

# valid bit depths
VALID_BIT_DEPTHS = (8, 16, 24)

# raw size percent difference threshold
RAW_SIZE_PERCENT_DIFFERENCE_THRESHOLD = 0.1 # 10%

##################################################


# DATASET SPECIFIC CONSTANTS
##################################################

# MUSDB18 Mono
MUSDB18MONO_DATA_DIR = "/graft3/datasets/pnlong/lnac/sashimi/data/musdb18mono" # yggdrasil

# MUSDB18 Stereo
MUSDB18STEREO_DATA_DIR = "/graft3/datasets/pnlong/lnac/sashimi/data/musdb18stereo" # yggdrasil

# LibriSpeech
LIBRISPEECH_SPLIT = "dev-clean" # "dev-clean" or "train-clean-100"
LIBRISPEECH_DATA_DIR = f"/graft3/datasets/pnlong/lnac/sashimi/data/librispeech/LibriSpeech/{LIBRISPEECH_SPLIT}" # yggdrasil

# LJSpeech
LJSPEECH_DATA_DIR = "/graft3/datasets/pnlong/lnac/sashimi/data/ljspeech" # yggdrasil

# Epidemic Sound
EPIDEMIC_SOUND_DATA_DIR = "/graft1/datasets/kechen/epidemic/epidemic_sound" # pando

# VCTK (speech)
VCTK_DATA_DIR = "/graft2/datasets/znovack/VCTK-Corpus-0.92/wav48_silence_trimmed" # pando

# Torrent Data 16-bit
TORRENT_DATA_DATA_DIR = "/graft3/datasets/znovack/trilobyte" # yggdrasil

# Birdvox bioacoustic data
BIRDVOX_DATA_DIR = "/mnt/arrakis_data/pnlong/lnac/birdvox/unit06" # yggdrasil

# Beethoven Piano Sonatas
BEETHOVEN_DATA_DIR = "/mnt/arrakis_data/pnlong/lnac/beethoven" # yggdrasil

# YouTubeMix Audio Dataset
YOUTUBE_MIX_DATA_DIR = "/mnt/arrakis_data/pnlong/lnac/youtube_mix" # yggdrasil

# SC09 Speech Dataset
SC09_DATA_DIR = "/mnt/arrakis_data/pnlong/lnac/sc09" # yggdrasil

##################################################


# HELPER FUNCTIONS
##################################################

def load_output_results(
    filepath: str = DEFAULT_OUTPUT_FILEPATH,
) -> pd.DataFrame:
    """
    Load the output results from a file.
    
    Args:
        filepath: The path to the file.
    
    Returns:
        A pandas DataFrame containing the output results.
    """
    results = pd.read_csv(filepath_or_buffer = filepath, sep = ",", header = 0, index_col = False)
    results = results[["dataset", "bit_depth", "is_native_bit_depth", "overall_compression_rate"]]
    return results


def load_audio(
    path: str,
    expected_sample_rate: int = None,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and convert it to the target bit depth.
    
    Args:
        path: The path to the audio file.
        expected_sample_rate: The expected sample rate. If None, the sample rate will not be checked.
    
    Returns:
        The waveform and sample rate. Note that waveform will always be signed integer data type.
    """

    # read audio file
    try:
        waveform, sample_rate = sf.read(file = path, dtype = np.float32) # get the audio as a numpy array
    except Exception as e: # if soundfile fails, use alternative method, which reads the file as a pydub AudioSegment and then converts to a numpy array
        # warnings.warn(f"Error reading audio file {path} with soundfile, using alternative method: {e}", category = RuntimeWarning)
        waveform = pydub.AudioSegment.from_file(file = path, format = path.split(".")[-1])
        stream = io.BytesIO()
        waveform.export(stream, format = "FLAC") # export as corrected FLAC file
        waveform, sample_rate = sf.read(file = stream, dtype = np.float32) # get the audio as a numpy array
        del stream

    # make assertions
    if expected_sample_rate is not None:
        assert sample_rate == expected_sample_rate, f"Sample rate mismatch: {sample_rate} != {expected_sample_rate}."
    
    # return waveform and sample rate
    return waveform, sample_rate

##################################################


# DATASET BASE CLASS
##################################################

class Dataset:
    """Base class for datasets."""

    def __init__(
        self,
        name: str,
        sample_rate: int,
        bit_depth: int,
        native_bit_depth: int,
        is_mu_law: bool,
        native_is_mu_law: bool,
        is_mono: bool,
        paths: List[str],
    ):
        """Initialize the dataset."""
        self.name: str = name
        self.sample_rate: int = sample_rate
        self.bit_depth: int = bit_depth
        self.native_bit_depth: int = native_bit_depth
        self.is_mu_law: bool = is_mu_law
        self.native_is_mu_law: bool = native_is_mu_law
        self.is_mono: bool = is_mono
        self.paths: List[str] = paths
        if len(self.paths) == 0:
            raise ValueError(f"No paths found for dataset: {self.name}.")

    def __str__(self) -> str:
        """Return a string representation of the dataset."""
        return self.name

    def get_description(self) -> str:
        """Return a description of the dataset."""
        return (
            f"{self.name} dataset " + 
            "(" +
            f"{len(self)} files, " + 
            (f"{self.sample_rate} Hz" if self.sample_rate is not None else "variable sample rate") + 
            f", {self.bit_depth}-bit, " + 
            f"{'mu-law' if self.is_mu_law else 'linear'}, " + 
            f"{'mono' if self.is_mono else 'stereo'}" + 
            ")"
        )

    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        """Return the item at the given index."""
        waveform, sample_rate = load_audio(
            path = self.paths[index],
            expected_sample_rate = self.sample_rate,
        )
        return waveform, sample_rate

##################################################


# DATASETS
##################################################

# MUSDB18 Dataset Base Class
class MUSDB18Dataset(Dataset):
    """Dataset for MUSDB18."""

    def __init__(
        self,
        is_mono: bool,
        bit_depth: int = None,
        is_mu_law: bool = None,
        subset: str = None,
        partition: str = None,
    ):
        native_bit_depth: int = 16
        bit_depth = native_bit_depth if bit_depth is None else bit_depth
        native_is_mu_law: bool = False
        is_mu_law = native_is_mu_law if is_mu_law is None else is_mu_law
        paths = self._get_paths(is_mono = is_mono, subset = subset, partition = partition)
        super().__init__(
            name = "musdb18" + ("mono" if is_mono else "stereo") + (f"_{subset}" if subset is not None else "") + (f"_{partition}" if partition is not None else ""),
            sample_rate = 44100,
            bit_depth = bit_depth,
            native_bit_depth = native_bit_depth,
            is_mu_law = is_mu_law,
            native_is_mu_law = native_is_mu_law,
            is_mono = is_mono,
            paths = paths,
        )

    def _get_paths(
        self,
        is_mono: bool,
        subset: str,
        partition: str,
    ) -> List[str]:
        """Return the paths of the dataset."""
        data_dir = MUSDB18MONO_DATA_DIR if is_mono else MUSDB18STEREO_DATA_DIR
        musdb18 = pd.read_csv(filepath_or_buffer = f"{data_dir}/mixes.csv", sep = ",", header = 0, index_col = False)
        musdb18["path"] = musdb18["path"].apply(lambda path: f"{data_dir}/{path}")
        if subset == "mixes": # include only mixes, instead of everything
            musdb18 = musdb18[musdb18["is_mix"]]
        elif subset == "stems": # include only stems, instead of everything
            musdb18 = musdb18[~musdb18["is_mix"]]
        if partition == "train": # include only the "train" partition
            musdb18 = musdb18[musdb18["is_train"]]
        elif partition == "valid": # include only the "valid" partition
            musdb18 = musdb18[~musdb18["is_train"]]
        return musdb18["path"].tolist()


# MUSDB18 Mono Dataset
class MUSDB18MonoDataset(MUSDB18Dataset):
    """Dataset for MUSDB18 Mono."""

    def __init__(
        self,
        bit_depth: int = None,
        is_mu_law: bool = None,
        subset: str = None,
        partition: str = None,
    ):
        super().__init__(
            is_mono = True,
            bit_depth = bit_depth,
            is_mu_law = is_mu_law,
            subset = subset,
            partition = partition,
        )


# MUSDB18 Stereo Dataset
class MUSDB18StereoDataset(MUSDB18Dataset):
    """Dataset for MUSDB18 Stereo."""

    def __init__(
        self,
        bit_depth: int = None,
        is_mu_law: bool = None,
        subset: str = None,
        partition: str = None,
    ):
        super().__init__(
            is_mono = False,
            bit_depth = bit_depth,
            is_mu_law = is_mu_law,
            subset = subset,
            partition = partition,
        )


# LibriSpeech Dataset
class LibriSpeechDataset(Dataset):
    """Dataset for LibriSpeech."""

    def __init__(
        self,
        bit_depth: int = None,
        is_mu_law: bool = None,
    ):
        native_bit_depth: int = 16
        bit_depth = native_bit_depth if bit_depth is None else bit_depth
        native_is_mu_law: bool = False
        is_mu_law = native_is_mu_law if is_mu_law is None else is_mu_law
        paths = self._get_paths()
        super().__init__(
            name = "librispeech",
            sample_rate = 16000,
            bit_depth = bit_depth,
            native_bit_depth = native_bit_depth,
            is_mu_law = is_mu_law,
            native_is_mu_law = native_is_mu_law,
            is_mono = True,
            paths = paths,
        )

    def _get_paths(self) -> List[str]:
        """Return the paths of the dataset."""
        paths = glob.glob(f"{LIBRISPEECH_DATA_DIR}/**/*.flac", recursive = True)
        return paths


# LJSpeech Dataset
class LJSpeechDataset(Dataset):
    """Dataset for LJSpeech."""

    def __init__(
        self,
        bit_depth: int = None,
        is_mu_law: bool = None,
    ):
        native_bit_depth: int = 16
        bit_depth = native_bit_depth if bit_depth is None else bit_depth
        native_is_mu_law: bool = False
        is_mu_law = native_is_mu_law if is_mu_law is None else is_mu_law
        paths = self._get_paths()
        super().__init__(
            name = "ljspeech",
            sample_rate = 22050,
            bit_depth = bit_depth,
            native_bit_depth = native_bit_depth,
            is_mu_law = is_mu_law,
            native_is_mu_law = native_is_mu_law,
            is_mono = True,
            paths = paths,
        )

    def _get_paths(self) -> List[str]:
        """Return the paths of the dataset."""
        paths = glob.glob(f"{LJSPEECH_DATA_DIR}/**/*.wav", recursive = True)
        return paths


# Epidemic Sound Dataset
class EpidemicSoundDataset(Dataset):
    """Dataset for Epidemic Sound."""

    def __init__(
        self,
        bit_depth: int = None,
        is_mu_law: bool = None,
    ):
        native_bit_depth: int = 24
        bit_depth = native_bit_depth if bit_depth is None else bit_depth
        native_is_mu_law: bool = False
        is_mu_law = native_is_mu_law if is_mu_law is None else is_mu_law
        paths = self._get_paths()
        super().__init__(
            name = "epidemic",
            sample_rate = 48000,
            bit_depth = bit_depth,
            native_bit_depth = native_bit_depth,
            is_mu_law = is_mu_law,
            native_is_mu_law = native_is_mu_law,
            is_mono = True,
            paths = paths,
        )

    def _get_paths(self) -> List[str]:
        """Return the paths of the dataset."""
        paths = glob.glob(f"{EPIDEMIC_SOUND_DATA_DIR}/**/*.flac", recursive = True)
        return paths


# VCTK Dataset
class VCTKDataset(Dataset):
    """Dataset for VCTK."""

    def __init__(
        self,
        bit_depth: int = None,
        is_mu_law: bool = None,
    ):
        native_bit_depth: int = 16
        bit_depth = native_bit_depth if bit_depth is None else bit_depth
        native_is_mu_law: bool = False
        is_mu_law = native_is_mu_law if is_mu_law is None else is_mu_law
        paths = self._get_paths()
        super().__init__(
            name = "vctk",
            sample_rate = 48000,
            bit_depth = bit_depth,
            native_bit_depth = native_bit_depth,
            is_mu_law = is_mu_law,
            native_is_mu_law = native_is_mu_law,
            is_mono = True,
            paths = paths,
        )

    def _get_paths(self) -> List[str]:
        """Return the paths of the dataset."""
        paths = glob.glob(f"{VCTK_DATA_DIR}/**/*.flac", recursive = True)
        return paths


# Torrent Dataset Base Class
class TorrentDataset(Dataset):
    """Dataset for Torrented Audio Files."""
    
    def __init__(
        self,
        bit_depth: int,
        native_bit_depth: int,
        subset: str,
        is_mu_law: bool = None,
    ):
        native_is_mu_law: bool = False
        is_mu_law = native_is_mu_law if is_mu_law is None else is_mu_law
        paths = self._get_paths(
            native_bit_depth = native_bit_depth,
            subset = subset,
        )
        super().__init__(
            name = f"torrent{native_bit_depth}b" + (f"_{subset}" if subset is not None else ""),
            sample_rate = None,
            bit_depth = bit_depth,
            native_bit_depth = native_bit_depth,
            is_mu_law = is_mu_law,
            native_is_mu_law = native_is_mu_law,
            is_mono = False,
            paths = paths,
        )

    def _get_paths(
        self,
        native_bit_depth: int,
        subset: str,
    ) -> List[str]:
        """Return the paths of the dataset."""
        if subset == "pro":
            paths = glob.glob(f"{TORRENT_DATA_DATA_DIR}/Pro/{native_bit_depth}b/**/*.flac", recursive = True)
        elif subset == "amateur":
            paths = glob.glob(f"{TORRENT_DATA_DATA_DIR}/train/Amateur/{native_bit_depth}b/**/*.flac", recursive = True)
        elif subset == "freeload":
            paths = glob.glob(f"{TORRENT_DATA_DATA_DIR}/train/Freeload/{native_bit_depth}b/**/*.flac", recursive = True)
        else:
            paths = glob.glob(f"{TORRENT_DATA_DATA_DIR}/**/{native_bit_depth}b/**/*.flac", recursive = True)
        return paths


# Torrent Dataset 16-bit
class Torrent16BDataset(TorrentDataset):
    """Dataset for Torrented Audio Files (16-bit)."""
    
    def __init__(
        self,
        bit_depth: int = None,
        is_mu_law: bool = None,
        subset: str = None,
    ):
        native_bit_depth: int = 16
        bit_depth = native_bit_depth if bit_depth is None else bit_depth
        super().__init__(
            bit_depth = bit_depth,
            native_bit_depth = native_bit_depth,
            subset = subset,
            is_mu_law = is_mu_law,
        )


# Torrent Dataset 24-bit
class Torrent24BDataset(TorrentDataset):
    """Dataset for Torrented Audio Files (24-bit)."""

    def __init__(
        self,
        bit_depth: int = None,
        is_mu_law: bool = None,
        subset: str = None,
    ):
        native_bit_depth: int = 24
        bit_depth = native_bit_depth if bit_depth is None else bit_depth
        super().__init__(
            bit_depth = bit_depth,
            native_bit_depth = native_bit_depth,
            subset = subset,
            is_mu_law = is_mu_law,
        )


# Birdvox Dataset
class BirdvoxDataset(Dataset):
    """Dataset for Birdvox."""

    def __init__(
        self,
        bit_depth: int = None,
        is_mu_law: bool = None,
    ):
        native_bit_depth: int = 16
        bit_depth = native_bit_depth if bit_depth is None else bit_depth
        native_is_mu_law: bool = False
        is_mu_law = native_is_mu_law if is_mu_law is None else is_mu_law
        paths = self._get_paths()
        super().__init__(
            name = "birdvox",
            sample_rate = 24000,
            bit_depth = bit_depth,
            native_bit_depth = native_bit_depth,
            is_mu_law = is_mu_law,
            native_is_mu_law = native_is_mu_law,
            is_mono = True,
            paths = paths,
        )

    def _get_paths(self) -> List[str]:
        """Return the paths of the dataset."""
        paths = glob.glob(f"{BIRDVOX_DATA_DIR}/**/*.flac", recursive = True)
        paths = [path for path in paths if basename(dirname(path)) != "split_data"] # exclude split_data directory
        return paths


# Beethoven Dataset
class BeethovenDataset(Dataset):
    """Dataset for Beethoven Piano Sonatas."""

    def __init__(
        self,
        bit_depth: int = None,
        is_mu_law: bool = None,
    ):
        native_bit_depth: int = 8
        bit_depth = native_bit_depth if bit_depth is None else bit_depth
        native_is_mu_law: bool = True
        is_mu_law = native_is_mu_law if is_mu_law is None else is_mu_law
        paths = self._get_paths()
        super().__init__(
            name = "beethoven",
            sample_rate = 16000,
            bit_depth = bit_depth,
            native_bit_depth = native_bit_depth,
            is_mu_law = is_mu_law,
            native_is_mu_law = native_is_mu_law,
            is_mono = True,
            paths = paths,
        )

    def _get_paths(self) -> List[str]:
        """Return the paths of the dataset."""
        paths = glob.glob(f"{BEETHOVEN_DATA_DIR}/**/*.wav", recursive = True)
        return paths


# YouTubeMix Dataset
class YouTubeMixDataset(Dataset):
    """Dataset for YouTubeMix."""

    def __init__(
        self,
        bit_depth: int = None,
        is_mu_law: bool = None,
    ):
        native_bit_depth: int = 8
        bit_depth = native_bit_depth if bit_depth is None else bit_depth
        native_is_mu_law: bool = True
        is_mu_law = native_is_mu_law if is_mu_law is None else is_mu_law
        paths = self._get_paths()
        super().__init__(
            name = "youtube_mix",
            sample_rate = 16000,
            bit_depth = bit_depth,
            native_bit_depth = native_bit_depth,
            is_mu_law = is_mu_law,
            native_is_mu_law = native_is_mu_law,
            is_mono = True,
            paths = paths,
        )

    def _get_paths(self) -> List[str]:
        """Return the paths of the dataset."""
        paths = glob.glob(f"{YOUTUBE_MIX_DATA_DIR}/**/*.wav", recursive = True)
        return paths


# SC09 Speech Dataset
class SC09SpeechDataset(Dataset):
    """Dataset for SC09 Speech."""

    def __init__(
        self,
        bit_depth: int = None,
        is_mu_law: bool = None,
    ):
        native_bit_depth: int = 8
        bit_depth = native_bit_depth if bit_depth is None else bit_depth
        native_is_mu_law: bool = True
        is_mu_law = native_is_mu_law if is_mu_law is None else is_mu_law
        paths = self._get_paths()
        super().__init__(
            name = "sc09",
            sample_rate = 16000,
            bit_depth = bit_depth,
            native_bit_depth = native_bit_depth,
            is_mu_law = is_mu_law,
            native_is_mu_law = native_is_mu_law,
            is_mono = True,
            paths = paths,
        )

    def _get_paths(self) -> List[str]:
        """Return the paths of the dataset."""
        paths = glob.glob(f"{SC09_DATA_DIR}/**/*.wav", recursive = True)
        return paths

##################################################


# DICTIONARY OF DATASETS
##################################################

# get choices of datasets
def get_dataset_choices() -> List[str]:
    """Return the choices of datasets."""
    dataset_choices = []
    for mono_stereo in ("mono", "stereo"): # musdb18 mono or stereo
        for mixes in ("", "_mixes", "_stems"): # "" for all, "_mixes" for mixes only, "_stems" for stems only
            for partition in ("", "_train", "_valid"): # "" for all, "_train" for train, "_valid" for valid
                dataset_choices.append("musdb18" + mono_stereo + mixes + partition) # e.g. "musdb18mono_mixes_train", "musdb18stereo_stems_train", "musdb18stereo_valid", etc.
    dataset_choices.append("librispeech") # librispeech
    dataset_choices.append("ljspeech") # ljspeech
    dataset_choices.append("epidemic") # epidemicsound
    dataset_choices.append("vctk") # vctk
    for bit_depth in (16, 24):
        for torrent_subset in ("", "_pro", "_amateur", "_freeload"):
            dataset_choices.append("torrent" + str(bit_depth) + "b" + torrent_subset) # e.g. "torrent16b", "torrent16b_pro", "torrent16b_amateur", "torrent16b_freeload", "torrent24b", "torrent24b_pro", "torrent24b_amateur", "torrent24b_freeload", etc.
    dataset_choices.append("birdvox") # birdvox
    dataset_choices.append("beethoven") # beethoven
    dataset_choices.append("youtube_mix") # youtube_mix
    dataset_choices.append("sc09") # sc09
    return dataset_choices

# factory function to get dataset
def get_dataset(
    dataset_name: str,
    bit_depth: int = None,
    is_mu_law: bool = None,
) -> Dataset:
    """
    Factory function to get dataset.
    
    Parameters:
        dataset_name: str - The name of the dataset.
        bit_depth: int - The bit depth of the dataset.
        is_mu_law: bool - Whether to use mu-law encoding.

    Returns:
        Dataset - The dataset.
    """

    # default error message
    dataset = None
    error_message = f"Invalid dataset name: {dataset_name}."

    # factory ladder
    if dataset_name.startswith("musdb18mono") or dataset_name.startswith("musdb18stereo"):
        subset = None # default to all
        if "mixes" in dataset_name:
            subset = "mixes"
        elif "stems" in dataset_name:
            subset = "stems"
        partition = None # default to all
        if "train" in dataset_name:
            partition = "train"
        elif "valid" in dataset_name:
            partition = "valid"
        if dataset_name.startswith("musdb18mono"):
            dataset = MUSDB18MonoDataset(bit_depth = bit_depth, is_mu_law = is_mu_law, subset = subset, partition = partition)
        elif dataset_name.startswith("musdb18stereo"):
            dataset = MUSDB18StereoDataset(bit_depth = bit_depth, is_mu_law = is_mu_law, subset = subset, partition = partition)
    elif dataset_name == "librispeech":
        dataset = LibriSpeechDataset(bit_depth = bit_depth, is_mu_law = is_mu_law)
    elif dataset_name == "ljspeech":
        dataset = LJSpeechDataset(bit_depth = bit_depth, is_mu_law = is_mu_law)
    elif dataset_name == "epidemic":
        dataset = EpidemicSoundDataset(bit_depth = bit_depth, is_mu_law = is_mu_law)
    elif dataset_name == "vctk":
        dataset = VCTKDataset(bit_depth = bit_depth, is_mu_law = is_mu_law)
    elif dataset_name.startswith("torrent"):
        subset = None # default to all
        if "pro" in dataset_name:
            subset = "pro"
        elif "amateur" in dataset_name:
            subset = "amateur"
        elif "freeload" in dataset_name:
            subset = "freeload"
        if dataset_name.startswith("torrent16b"):
            dataset = Torrent16BDataset(bit_depth = bit_depth, is_mu_law = is_mu_law, subset = subset)
        elif dataset_name.startswith("torrent24b"):
            dataset = Torrent24BDataset(bit_depth = bit_depth, is_mu_law = is_mu_law, subset = subset)
    elif dataset_name == "birdvox":
        dataset = BirdvoxDataset(bit_depth = bit_depth, is_mu_law = is_mu_law)
    elif dataset_name == "beethoven":
        dataset = BeethovenDataset(bit_depth = bit_depth, is_mu_law = is_mu_law)
    elif dataset_name == "youtube_mix":
        dataset = YouTubeMixDataset(bit_depth = bit_depth, is_mu_law = is_mu_law)
    elif dataset_name == "sc09":
        dataset = SC09SpeechDataset(bit_depth = bit_depth, is_mu_law = is_mu_law)
    else:
        raise ValueError(error_message)

    # assert dataset is not None
    if dataset is None:
        raise ValueError(error_message)

    # assert bit depth is valid
    assert dataset.bit_depth in VALID_BIT_DEPTHS, f"Invalid bit depth: {dataset.bit_depth}. Valid bit depths are {VALID_BIT_DEPTHS}."

    # return dataset
    return dataset

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
        parser.add_argument("--bit_depth", type = int, default = None, choices = VALID_BIT_DEPTHS, help = "Bit depth of the audio files. If not provided, the bit depth is determined by the dataset.")
        parser.add_argument("--is_mu_law", type = bool, default = None, help = "Whether to use mu-law encoding. If not provided, the is_mu_law is determined by the dataset.")
        parser.add_argument("--flac_compression_level", type = int, default = DEFAULT_FLAC_COMPRESSION_LEVEL, choices = list(range(0, 9)), help = "Compression level for FLAC.")
        parser.add_argument("--output_filepath", type = str, default = DEFAULT_OUTPUT_FILEPATH, help = "Absolute filepath to output CSV file.")
        parser.add_argument("--jobs", type = int, default = int(multiprocessing.cpu_count() / 4), help = "Number of workers for multiprocessing.")
        parser.add_argument("--disable_constant_subframes", action = "store_true", help = "Disable constant subframes for FLAC.")
        parser.add_argument("--disable_fixed_subframes", action = "store_true", help = "Disable fixed subframes for FLAC.")
        parser.add_argument("--disable_verbatim_subframes", action = "store_true", help = "Disable verbatim subframes for FLAC.")
        parser.add_argument("--reset", action = "store_true", help = "Reset the output file.")
        args = parser.parse_args(args = args, namespace = namespace) # parse arguments
        assert args.flac_compression_level >= 0 and args.flac_compression_level <= 8, f"Invalid FLAC compression level: {args.flac_compression_level}. Valid compression levels are 0 to 8."
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
    output_filepath = args.output_filepath
    if not exists(output_filepath) or args.reset:
        pd.DataFrame(columns = [
            "dataset",
            "bit_depth",
            "is_native_bit_depth",
            "is_mu_law",
            "matches_native_quantization",
            "total_size",
            "compressed_size",
            "overall_compression_rate",
            "mean_compression_rate",
            "median_compression_rate",
            "std_compression_rate",
            "max_compression_rate",
            "min_compression_rate",
            "flac_compression_level",
            "disable_constant_subframes",
            "disable_fixed_subframes",
            "disable_verbatim_subframes",
            "datetime",
        ]).to_csv(path_or_buf = output_filepath, sep = ",", na_rep = "NA", header = True, index = False, mode = "w")

    # get dataset
    dataset = get_dataset(dataset_name = args.dataset, bit_depth = args.bit_depth, is_mu_law = args.is_mu_law)
    assert dataset.bit_depth in VALID_BIT_DEPTHS, f"Dataset bit depth {dataset.bit_depth} is not a valid bit depth. Valid bit depths are {VALID_BIT_DEPTHS}."
    assert dataset.is_mu_law is not None, f"Dataset is_mu_law is not set. Provided is_mu_law: {dataset.is_mu_law}."
    
    # log some information about the dataset
    dataset_name = f" {dataset.name.upper()}, {'pseudo-' if dataset.native_bit_depth != dataset.bit_depth else ''}{dataset.bit_depth}-bit, {'mu-law' if dataset.is_mu_law else 'linear'} " # add spaces on side so it looks nicer
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

            # get waveform
            waveform, sample_rate = dataset[index]
            assert waveform.dtype == np.float32, f"Waveform must be a float32 numpy array, but got {waveform.dtype}."

            # get correct PCM subtype for soundfile
            if dataset.bit_depth == 8:
                subtype = "PCM_U8"
            elif dataset.bit_depth == 16:
                subtype = "PCM_16"
            elif dataset.bit_depth == 24:
                subtype = "PCM_24"
            else:
                raise ValueError(f"Invalid bit depth: {dataset.bit_depth}. Valid bit depths are {VALID_BIT_DEPTHS}.")

            # convert waveform to mu-law encoding if necessary, assumes waveform is in the range [-1, 1] and linear-quantized
            if dataset.is_mu_law: # perform mu-law companding transformation
                mu = (2 ** dataset.bit_depth) - 1
                numerator = np.log1p(mu * np.abs(waveform + 1e-8))
                denominator = np.log1p(mu)
                waveform = np.sign(waveform) * (numerator / denominator)
                del mu, numerator, denominator

            # write original waveform to temporary file
            wav_filepath = f"{tmp_dir}/original.wav"
            sf.write(
                file = wav_filepath,
                data = waveform,
                samplerate = sample_rate,
                format = "WAV",
                subtype = subtype,
            )
            raw_size = getsize(wav_filepath)
            theoretical_raw_size = np.prod(waveform.shape) * (dataset.bit_depth // 8)
            raw_size_percent_difference = abs((raw_size - theoretical_raw_size) / theoretical_raw_size)
            assert raw_size_percent_difference < RAW_SIZE_PERCENT_DIFFERENCE_THRESHOLD, f"Raw size mismatch: % difference between raw size and theoretical raw size is {raw_size_percent_difference:.2%}, which is greater than {raw_size_percent_difference_threshold:.2%}." # check that raw size is within certain percentage of theoretical raw size

            # compress waveform to temporary file
            flac_filepath = f"{tmp_dir}/compressed.flac"
            _ = subprocess.run(args = [
                    "flac",
                    "-o", flac_filepath,
                    f"--compression-level-{args.flac_compression_level}",
                    "--force",
                ] + (
                    ["--disable-constant-subframes"] if args.disable_constant_subframes else []
                ) + (
                    ["--disable-fixed-subframes"] if args.disable_fixed_subframes else []
                ) + (
                    ["--disable-verbatim-subframes"] if args.disable_verbatim_subframes else []
                ) + [
                    wav_filepath,
                ], 
                check = True,
                stdout = subprocess.DEVNULL,
                stderr = subprocess.DEVNULL,
            )
            compressed_size = getsize(flac_filepath)

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
        del results

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
        "bit_depth": dataset.bit_depth,
        "is_native_bit_depth": dataset.native_bit_depth == dataset.bit_depth,
        "is_mu_law": dataset.is_mu_law,
        "matches_native_quantization": dataset.native_is_mu_law == dataset.is_mu_law,
        "total_size": total_size,
        "compressed_size": compressed_size,
        "overall_compression_rate": overall_compression_rate,
        "mean_compression_rate": mean_compression_rate,
        "median_compression_rate": median_compression_rate,
        "std_compression_rate": std_compression_rate,
        "max_compression_rate": max_compression_rate,
        "min_compression_rate": min_compression_rate,
        "flac_compression_level": args.flac_compression_level,
        "disable_constant_subframes": args.disable_constant_subframes,
        "disable_fixed_subframes": args.disable_fixed_subframes,
        "disable_verbatim_subframes": args.disable_verbatim_subframes,
        "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), # current datetime
    }]).to_csv(path_or_buf = output_filepath, sep = ",", na_rep = "NA", header = False, index = False, mode = "a")

    ##################################################

##################################################
