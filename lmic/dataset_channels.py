#!/usr/bin/env python3
"""
For each dataset in the LMIC audio data loaders, load one track and record the number of channels.
Outputs a DataFrame (CSV) with columns: dataset, num_channels (int), is_stereo (bool), note (reason if missing).
Run from the lmic/ directory so that language_modeling_is_compression is importable.

NaN num_channels (e.g. epidemic, vctk) usually means: no files found under that dataset's path,
or an error while loading (e.g. path not mounted, permission). The "note" column records the reason.
"""

import argparse
import os
import sys
from collections.abc import Iterator

# Allow importing language_modeling_is_compression when run from repo root or lmic/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Package lives in language_modeling_is_compression/ under this dir
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import numpy as np
import pandas as pd

from language_modeling_is_compression import data_loaders_audio


def get_waveform_iterator(dataset: str) -> Iterator[np.ndarray]:
    """Return an iterator that yields one waveform (numpy array) per track for the given dataset.
    Mirrors the dataset list from GET_AUDIO_DATA_GENERATOR_FN_DICT."""
    dl = data_loaders_audio
    if dataset not in dl.GET_AUDIO_DATA_GENERATOR_FN_DICT:
        raise ValueError(f"Unknown dataset: {dataset}. Valid: {list(dl.GET_AUDIO_DATA_GENERATOR_FN_DICT.keys())}.")
    if dataset.startswith("musdb18mono"):
        part, sub = None, None
        rest = dataset.replace("musdb18mono", "").lstrip("_")
        if rest:
            for s in ("mixes", "stems"):
                if rest == s or rest.startswith(s + "_"):
                    sub = s
                    rest = rest[len(s) :].lstrip("_")
                    break
            if rest in ("train", "valid"):
                part = rest
        return dl._get_musdb18mono_dataset(partition=part, subset=sub)
    if dataset.startswith("musdb18stereo"):
        part, sub = None, None
        rest = dataset.replace("musdb18stereo", "").lstrip("_")
        if rest:
            for s in ("mixes", "stems"):
                if rest == s or rest.startswith(s + "_"):
                    sub = s
                    rest = rest[len(s) :].lstrip("_")
                    break
            if rest in ("train", "valid"):
                part = rest
        return dl._get_musdb18stereo_dataset(partition=part, subset=sub)
    if dataset == "librispeech":
        return dl._get_librispeech_dataset()
    if dataset == "ljspeech":
        return dl._get_ljspeech_dataset()
    if dataset == "epidemic":
        return dl._get_epidemic_dataset()
    if dataset == "vctk":
        return dl._get_vctk_dataset()
    if dataset.startswith("torrent16b"):
        sub = None
        if dataset != "torrent16b":
            sub = dataset.replace("torrent16b_", "")
        return dl._get_torrent_dataset(native_bit_depth=16, subset=sub)
    if dataset.startswith("torrent24b"):
        sub = None
        if dataset != "torrent24b":
            sub = dataset.replace("torrent24b_", "")
        return dl._get_torrent_dataset(native_bit_depth=24, subset=sub)
    if dataset == "birdvox":
        return dl._get_birdvox_dataset()
    if dataset == "beethoven":
        return dl._get_beethoven_dataset()
    if dataset == "youtube_mix":
        return dl._get_youtube_mix_dataset()
    if dataset == "sc09":
        return dl._get_sc09_dataset()
    raise ValueError(f"Dataset not implemented for waveform iterator: {dataset}.")


def main():
    parser = argparse.ArgumentParser(description="Report number of channels per dataset (one track per dataset).")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=os.path.join(_SCRIPT_DIR, "dataset_channels.csv"),
        help="Output CSV path",
    )
    args = parser.parse_args()

    rows = []
    for name in sorted(data_loaders_audio.GET_AUDIO_DATA_GENERATOR_FN_DICT.keys()):
        try:
            it = get_waveform_iterator(name)
            waveform = next(it)
            if waveform.ndim == 1:
                num_channels = 1
            else:
                num_channels = int(waveform.shape[1])
            rows.append({"dataset": name, "num_channels": num_channels, "note": ""})
        except StopIteration:
            rows.append({"dataset": name, "num_channels": None, "note": "no files"})
            print(f"Warning: {name}: no files in dataset path", file=sys.stderr)
        except Exception as e:
            rows.append({"dataset": name, "num_channels": None, "note": str(e)})
            print(f"Warning: {name}: {e}", file=sys.stderr)

    df = pd.DataFrame(rows)
    # Integer column (nullable so missing stay as NA)
    df["num_channels"] = df["num_channels"].astype("Int64")
    df["is_stereo"] = df["num_channels"] == 2
    df.to_csv(args.output, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
