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
from typing import List, Tuple
import tempfile
import soundfile as sf
import scipy.io.wavfile
import pandas as pd
import glob
import datetime

##################################################


# CONSTANTS
##################################################

# default output filepath
DEFAULT_OUTPUT_FILEPATH = "/home/pnlong/lnac/flac_eval_results.csv"

# FLAC compression level
DEFAULT_FLAC_COMPRESSION_LEVEL = 5

# valid bit depths
VALID_BIT_DEPTHS = (8, 16, 24)

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
    bit_depth: int,
    expected_sample_rate: int = None,
) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and convert it to the target bit depth.
    
    Args:
        path: The path to the audio file.
        bit_depth: The target bit depth. See VALID_BIT_DEPTHS for valid bit depths.
        expected_sample_rate: The expected sample rate. If None, the sample rate will not be checked.
    
    Returns:
        The waveform and sample rate. Note that waveform will always be signed integer data type.
    """

    # ensure bit depth is valid
    assert bit_depth in VALID_BIT_DEPTHS, f"Invalid bit depth: {bit_depth}. Valid bit depths are {VALID_BIT_DEPTHS}."

    # read audio file
    waveform, sample_rate = sf.read(file = path, dtype = np.float32) # get the audio as a numpy array
    waveform_dtype = np.int8 if bit_depth == 8 else np.int16 if bit_depth == 16 else np.int32
    waveform = (waveform * ((2 ** (bit_depth - 1)) - 1)).astype(waveform_dtype)

    # make assertions
    if expected_sample_rate is not None:
        assert sample_rate == expected_sample_rate, f"Sample rate mismatch: {sample_rate} != {expected_sample_rate}."
    waveform_min, waveform_max = waveform.min(), waveform.max()
    expected_waveform_min, expected_waveform_max = -(2 ** (bit_depth - 1)), (2 ** (bit_depth - 1)) - 1
    assert waveform_min >= expected_waveform_min and waveform_max <= expected_waveform_max, f"Waveform must be in the range [{expected_waveform_min}, {expected_waveform_max}]. Got min {waveform_min} and max {waveform_max}."
    
    # return waveform and sample rate
    return waveform, sample_rate


def convert_bit_depth(
    waveform: np.ndarray,
    bit_depth: int,
) -> np.ndarray:
    """
    Convert the bit depth of the waveform to the target bit depth.
    
    Args:
        waveform: The waveform to convert.
        bit_depth: The target bit depth.
    
    Returns:
        The converted waveform.
    """

    # confirm that bit depth is in fact 24-bit
    if waveform.dtype == np.int32 or waveform.dtype == np.uint32:
        waveform_min, waveform_max = waveform.min(), waveform.max()
        assert waveform_min >= -(2 ** 23) and waveform_max <= (2 ** 23) - 1, f"24-bit waveform must be in the range [-(2 ** 23), (2 ** 23) - 1]. Got min {waveform_min} and max {waveform_max}."

    # go through different data types
    if waveform.dtype == np.int8:
        if bit_depth == 8:
            new_waveform = waveform # no conversion needed
        elif bit_depth == 16:
            new_waveform = waveform.astype(np.int16) * (2 ** 8) # convert to 16-bit
        elif bit_depth == 24:
            new_waveform = waveform.astype(np.int32) * (2 ** 8) # convert to 24-bit
    elif waveform.dtype == np.uint8:
        if bit_depth == 8:
            new_waveform = waveform # no conversion needed
        elif bit_depth == 16:
            new_waveform = waveform.astype(np.uint16) * (2 ** 8) # convert to 16-bit
        elif bit_depth == 24:
            new_waveform = waveform.astype(np.uint32) * (2 ** 8) # convert to 24-bit
    elif waveform.dtype == np.int16:
        if bit_depth == 8:
            new_waveform = (waveform / (2 ** 8)).astype(np.int8) # convert to 8-bit
        elif bit_depth == 16:
            new_waveform = waveform # no conversion needed
        elif bit_depth == 24:
            new_waveform = waveform.astype(np.int32) * (2 ** 8) # convert to 24-bit
    elif waveform.dtype == np.uint16:
        if bit_depth == 8:
            new_waveform = (waveform / (2 ** 8)).astype(np.uint8) # convert to 8-bit
        elif bit_depth == 16:
            new_waveform = waveform # no conversion needed
        elif bit_depth == 24:
            new_waveform = waveform.astype(np.uint32) * (2 ** 8) # convert to 24-bit
    elif waveform.dtype == np.int32: # 24-bit signed masquerading as 32-bit because numpy doesn't support 24-bit
        if bit_depth == 8:
            new_waveform = (waveform / (2 ** 16)).astype(np.int8) # convert to 8-bit
        elif bit_depth == 16:
            new_waveform = (waveform / (2 ** 8)).astype(np.int16) # convert to 16-bit
        elif bit_depth == 24:
            new_waveform = waveform # no conversion needed
    elif waveform.dtype == np.uint32:
        if bit_depth == 8:
            new_waveform = (waveform / (2 ** 16)).astype(np.uint8) # convert to 8-bit
        elif bit_depth == 16:
            new_waveform = (waveform / (2 ** 8)).astype(np.uint16) # convert to 16-bit
        elif bit_depth == 24:
            new_waveform = waveform # no conversion needed
    else:
        raise ValueError(f"Invalid waveform dtype: {waveform.dtype}. Valid dtypes are np.int8, np.uint8, np.int16, np.uint16, np.int32, and np.uint32.")
    return new_waveform

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
        is_mono: bool,
        paths: List[str],
    ):
        """Initialize the dataset."""
        self.name: str = name
        self.sample_rate: int = sample_rate
        self.bit_depth: int = bit_depth
        self.native_bit_depth: int = native_bit_depth
        self.is_mono: bool = is_mono
        self.paths: List[str] = paths

    def __str__(self) -> str:
        """Return a string representation of the dataset."""
        return self.name

    def get_description(self) -> str:
        """Return a description of the dataset."""
        return f"{self.name} dataset ({len(self)} files, {self.sample_rate} Hz, {self.bit_depth}-bit, {'mono' if self.is_mono else 'stereo'})"

    def __len__(self):
        """Return the number of items in the dataset."""
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[np.ndarray, int]:
        """Return the item at the given index."""
        waveform, sample_rate = load_audio(
            path = self.paths[index],
            bit_depth = self.native_bit_depth,
            expected_sample_rate = self.sample_rate,
        )
        waveform = convert_bit_depth(
            waveform = waveform,
            bit_depth = self.bit_depth,
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
        mixes_only: bool = False,
        partition: str = None,
    ):
        native_bit_depth: int = 16
        bit_depth = native_bit_depth if bit_depth is None else bit_depth
        paths = self._get_paths(is_mono = is_mono, mixes_only = mixes_only, partition = partition)
        super().__init__(
            name = "musdb18" + ("mono" if is_mono else "stereo") + ("_mixes" if mixes_only else "") + (f"_{partition}" if partition is not None else ""),
            sample_rate = 44100,
            bit_depth = bit_depth,
            native_bit_depth = native_bit_depth,
            is_mono = is_mono,
            paths = paths,
        )

    def _get_paths(
        self,
        is_mono: bool,
        mixes_only: bool,
        partition: str,
    ) -> List[str]:
        """Return the paths of the dataset."""
        data_dir = MUSDB18MONO_DATA_DIR if is_mono else MUSDB18STEREO_DATA_DIR
        musdb18 = pd.read_csv(filepath_or_buffer = f"{data_dir}/mixes.csv", sep = ",", header = 0, index_col = False)
        musdb18["path"] = musdb18["path"].apply(lambda path: f"{data_dir}/{path}")
        if mixes_only: # include only mixes, instead of everything
            musdb18 = musdb18[musdb18["is_mix"]]
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
        mixes_only: bool = False,
        partition: str = None,
    ):
        super().__init__(
            is_mono = True,
            bit_depth = bit_depth,
            mixes_only = mixes_only,
            partition = partition,
        )


# MUSDB18 Stereo Dataset
class MUSDB18StereoDataset(MUSDB18Dataset):
    """Dataset for MUSDB18 Stereo."""

    def __init__(
        self,
        bit_depth: int = None,
        mixes_only: bool = False,
        partition: str = None,
    ):
        super().__init__(
            is_mono = False,
            bit_depth = bit_depth,
            mixes_only = mixes_only,
            partition = partition,
        )


# LibriSpeech Dataset
class LibriSpeechDataset(Dataset):
    """Dataset for LibriSpeech."""

    def __init__(
        self,
        bit_depth: int = None,
    ):
        native_bit_depth: int = 16
        bit_depth = native_bit_depth if bit_depth is None else bit_depth
        paths = self._get_paths()
        super().__init__(
            name = "librispeech",
            sample_rate = 16000,
            bit_depth = bit_depth,
            native_bit_depth = native_bit_depth,
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
    ):
        native_bit_depth: int = 16
        bit_depth = native_bit_depth if bit_depth is None else bit_depth
        paths = self._get_paths()
        super().__init__(
            name = "ljspeech",
            sample_rate = 22050,
            bit_depth = bit_depth,
            native_bit_depth = native_bit_depth,
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
    ):
        native_bit_depth: int = 24
        bit_depth = native_bit_depth if bit_depth is None else bit_depth
        paths = self._get_paths()
        super().__init__(
            name = "epidemic",
            sample_rate = 48000,
            bit_depth = bit_depth,
            native_bit_depth = native_bit_depth,
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
    ):
        native_bit_depth: int = 16
        bit_depth = native_bit_depth if bit_depth is None else bit_depth
        paths = self._get_paths()
        super().__init__(
            name = "vctk",
            sample_rate = 48000,
            bit_depth = bit_depth,
            native_bit_depth = native_bit_depth,
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
    ):
        paths = self._get_paths(subset = subset)
        super().__init__(
            name = f"torrent{native_bit_depth}b" + (f"_{subset}" if subset is not None else ""),
            sample_rate = 48000,
            bit_depth = bit_depth,
            native_bit_depth = native_bit_depth,
            is_mono = False,
            paths = paths,
        )

    def _get_paths(self, subset: str) -> List[str]:
        """Return the paths of the dataset."""
        if subset == "pro":
            paths = glob.glob(f"{TORRENT_DATA_DATA_DIR}/Pro/{self.native_bit_depth}b/**/*.flac", recursive = True)
        elif subset == "amateur":
            paths = glob.glob(f"{TORRENT_DATA_DATA_DIR}/train/Amateur/{self.native_bit_depth}b/**/*.flac", recursive = True)
        elif subset == "freeload":
            paths = glob.glob(f"{TORRENT_DATA_DATA_DIR}/train/Freeload/{self.native_bit_depth}b/**/*.flac", recursive = True)
        else:
            paths = glob.glob(f"{TORRENT_DATA_DATA_DIR}/**/{self.native_bit_depth}b/**/*.flac", recursive = True)
        return paths


# Torrent Dataset 16-bit
class Torrent16BDataset(TorrentDataset):
    """Dataset for Torrented Audio Files (16-bit)."""
    
    def __init__(
        self,
        bit_depth: int = None,
        subset: str = None,
    ):
        native_bit_depth: int = 16
        bit_depth = native_bit_depth if bit_depth is None else bit_depth
        super().__init__(
            bit_depth = bit_depth,
            native_bit_depth = native_bit_depth,
            subset = subset,
        )


# Torrent Dataset 24-bit
class Torrent24BDataset(TorrentDataset):
    """Dataset for Torrented Audio Files (24-bit)."""

    def __init__(
        self,
        bit_depth: int = None,
        subset: str = None,
    ):
        native_bit_depth: int = 24
        bit_depth = native_bit_depth if bit_depth is None else bit_depth
        super().__init__(
            bit_depth = bit_depth,
            native_bit_depth = native_bit_depth,
            subset = subset,
        )


# Birdvox Dataset
class BirdvoxDataset(Dataset):
    """Dataset for Birdvox."""

    def __init__(
        self,
        bit_depth: int = None,
    ):
        native_bit_depth: int = 16
        bit_depth = native_bit_depth if bit_depth is None else bit_depth
        paths = self._get_paths()
        super().__init__(
            name = "birdvox",
            sample_rate = 24000,
            bit_depth = bit_depth,
            native_bit_depth = native_bit_depth,
            is_mono = True,
            paths = paths,
        )

    def _get_paths(self) -> List[str]:
        """Return the paths of the dataset."""
        paths = glob.glob(f"{BIRDVOX_DATA_DIR}/**/*.flac", recursive = True)
        return paths

##################################################


# DICTIONARY OF DATASETS
##################################################

# get choices of datasets
def get_dataset_choices() -> List[str]:
    """Return the choices of datasets."""
    dataset_choices = []
    for mono_stereo in ("mono", "stereo"): # musdb18 mono or stereo
        for mixes in ("", "_mixes"): # "" for all, "_mixes" for mixes only
            for partition in ("", "_train", "_valid"): # "" for all, "_train" for train, "_valid" for valid
                dataset_choices.append("musdb18" + mono_stereo + mixes + partition) # e.g. "musdb18mono_mixes_train", "musdb18stereo_train", "musdb18stereo_valid", etc.
    dataset_choices.append("librispeech") # librispeech
    dataset_choices.append("ljspeech") # ljspeech
    dataset_choices.append("epidemic") # epidemicsound
    dataset_choices.append("vctk") # vctk
    for bit_depth in (16, 24):
        for torrent_subset in ("", "_pro", "_amateur", "_freeload"):
            dataset_choices.append("torrent" + str(bit_depth) + "b" + torrent_subset) # e.g. "torrent16b", "torrent16b_pro", "torrent16b_amateur", "torrent16b_freeload", "torrent24b", "torrent24b_pro", "torrent24b_amateur", "torrent24b_freeload", etc.
    dataset_choices.append("birdvox") # birdvox
    return dataset_choices

# factory function to get dataset
def get_dataset(
    dataset_name: str,
    bit_depth: int = None,
) -> Dataset:
    """
    Factory function to get dataset.
    
    Parameters:
        dataset_name: str - The name of the dataset.
        bit_depth: int - The bit depth of the dataset.

    Returns:
        Dataset - The dataset.
    """

    # default error message
    dataset = None
    error_message = f"Invalid dataset name: {dataset_name}."

    # factory ladder
    if dataset_name.startswith("musdb18mono") or dataset_name.startswith("musdb18stereo"):
        mixes_only = ("mixes" in dataset_name)
        partition = None # default to all
        if "train" in dataset_name:
            partition = "train"
        elif "valid" in dataset_name:
            partition = "valid"
        if dataset_name.startswith("musdb18mono"):
            dataset = MUSDB18MonoDataset(bit_depth = bit_depth, mixes_only = mixes_only, partition = partition) if bit_depth is not None else MUSDB18MonoDataset(mixes_only = mixes_only, partition = partition) # use default bit depth if not specified
        elif dataset_name.startswith("musdb18stereo"):
            dataset = MUSDB18StereoDataset(bit_depth = bit_depth, mixes_only = mixes_only, partition = partition) if bit_depth is not None else MUSDB18StereoDataset(mixes_only = mixes_only, partition = partition) # use default bit depth if not specified
    elif dataset_name == "librispeech":
        dataset = LibriSpeechDataset(bit_depth = bit_depth) if bit_depth is not None else LibriSpeechDataset() # use default bit depth if not specified
    elif dataset_name == "ljspeech":
        dataset = LJSpeechDataset(bit_depth = bit_depth) if bit_depth is not None else LJSpeechDataset() # use default bit depth if not specified
    elif dataset_name == "epidemic":
        dataset = EpidemicSoundDataset(bit_depth = bit_depth) if bit_depth is not None else EpidemicSoundDataset() # use default bit depth if not specified
    elif dataset_name == "vctk":
        dataset = VCTKDataset(bit_depth = bit_depth) if bit_depth is not None else VCTKDataset() # use default bit depth if not specified
    elif dataset_name.startswith("torrent"):
        subset = None # default to all
        if "pro" in dataset_name:
            subset = "pro"
        elif "amateur" in dataset_name:
            subset = "amateur"
        elif "freeload" in dataset_name:
            subset = "freeload"
        if dataset_name.startswith("torrent16b"):
            dataset = Torrent16BDataset(bit_depth = bit_depth, subset = subset) if bit_depth is not None else Torrent16BDataset(subset = subset) # use default bit depth if not specified
        elif dataset_name.startswith("torrent24b"):
            dataset = Torrent24BDataset(bit_depth = bit_depth, subset = subset) if bit_depth is not None else Torrent24BDataset(subset = subset) # use default bit depth if not specified
    elif dataset_name == "birdvox":
        dataset = BirdvoxDataset(bit_depth = bit_depth) if bit_depth is not None else BirdvoxDataset() # use default bit depth if not specified
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
        parser.add_argument("--bit_depth", type = int, default = None, choices = VALID_BIT_DEPTHS, help = "Bit depth of the audio files.")
        parser.add_argument("--flac_compression_level", type = int, default = DEFAULT_FLAC_COMPRESSION_LEVEL, choices = list(range(0, 9)), help = "Compression level for FLAC.")
        parser.add_argument("--output_filepath", type = str, default = DEFAULT_OUTPUT_FILEPATH, help = "Absolute filepath (CSV file) to append the evaluation results to.")
        parser.add_argument("--jobs", type = int, default = int(multiprocessing.cpu_count() / 4), help = "Number of workers for multiprocessing.")
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
    if not exists(args.output_filepath) or args.reset:
        pd.DataFrame(columns = [
            "dataset",
            "bit_depth",
            "is_native_bit_depth",
            "total_size",
            "compressed_size",
            "overall_compression_rate",
            "mean_compression_rate",
            "median_compression_rate",
            "std_compression_rate",
            "max_compression_rate",
            "min_compression_rate",
            "flac_compression_level",
            "datetime",
        ]).to_csv(path_or_buf = args.output_filepath, sep = ",", na_rep = "NA", header = True, index = False, mode = "w")

    # get dataset
    dataset = get_dataset(dataset_name = args.dataset, bit_depth = args.bit_depth)
    
    # log some information about the dataset
    dataset_name = f" {dataset.name.upper()}, {'pseudo-' if dataset.native_bit_depth != dataset.bit_depth else ''}{dataset.bit_depth}-bit " # add spaces on side so it looks nicer
    line_character, line_width = "=", 100
    logger.info(f"{dataset_name:{line_character}^{line_width}}") # print dataset name with equal signs
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

            # convert to either 16-bit or 32-bit for soundfile, which only supports 16-bit and 32-bit
            waveform_dtype = waveform.dtype
            if waveform.itemsize == 1: # 8-bit
                waveform = waveform.astype(np.int16)
                if waveform_dtype == np.int8: # convert signed to unsigned 8-bit
                    waveform += 2 ** 7
                subtype = "PCM_U8"
            elif waveform.itemsize == 2: # 16-bit
                if waveform_dtype == np.uint16: # convert unsigned to signed 16-bit
                    waveform = (waveform.astype(np.int32) - (2 ** 15)).astype(np.int16)
                subtype = "PCM_16"
            elif waveform.itemsize == 4: # 24-bit masquerading as 32-bit because numpy doesn't support 24-bit
                if waveform_dtype == np.uint32: # convert unsigned to signed 24-bit
                    waveform = (waveform.astype(np.int32) - (2 ** 23))
                subtype = "PCM_24"

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

            # compress waveform to temporary file
            flac_filepath = f"{tmp_dir}/compressed.flac"
            sf.write(
                file = flac_filepath,
                data = waveform,
                samplerate = sample_rate,
                format = "FLAC",
                compression_level = args.flac_compression_level / 8, # soundfile uses compression level [0.0, 1.0], so convert integer compression level to float compression level (maximum FLAC compression level is 8)
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
        "total_size": total_size,
        "compressed_size": compressed_size,
        "overall_compression_rate": overall_compression_rate,
        "mean_compression_rate": mean_compression_rate,
        "median_compression_rate": median_compression_rate,
        "std_compression_rate": std_compression_rate,
        "max_compression_rate": max_compression_rate,
        "min_compression_rate": min_compression_rate,
        "flac_compression_level": args.flac_compression_level,
        "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), # current datetime
    }]).to_csv(path_or_buf = args.output_filepath, sep = ",", na_rep = "NA", header = False, index = False, mode = "a")

    ##################################################

##################################################
