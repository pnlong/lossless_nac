# Phillip Long
# Preprocess the MusDB18 dataset to WAV files and mixes.csv.

import argparse
import multiprocessing
import os
from os.path import basename, dirname, exists, isdir, join, relpath, splitext

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

TARGET_SAMPLE_RATE = 44100
TARGET_BIT_DEPTH = 16
STEM_IDS = ["mixture", "drums", "bass", "other", "vocals"]
MIXES_CSV_COLUMNS = ["path", "is_train", "is_mix"]
VALID_TRACK_NAMES_MONO = set("""ANiMAL - Clinic A
Angels In Amplifiers - I'm Alright
AvaLuna - Waterduct
Black Bloc - If You Want Success
Clara Berry And Wooldog - Stella
Cnoc An Tursa - Bannockburn
Cristina Vane - So Easy
Detsky Sad - Walkie Talkie
Fergessen - Nos Palpitants
Fergessen - The Wind
James May - On The Line
Jokers, Jacks & Kings - Sea Of Leaves
Lyndsey Ollard - Catching Up
Matthew Entwistle - Dont You Ever
Motor Tapes - Shore
Music Delta - Country1
Music Delta - Disco
Music Delta - Punk
PR - Happy Daze
PR - Oh No
Secret Mountains - High Horse
Skelpolu - Resurrection
Spike Mullings - Mike's Sulking
Swinging Steaks - Lost My Way
The Districts - Vermont
The So So Glos - Emergency
Tom McKenzie - Directions
Triviul - Angelsaint
We Fell From The Sky - Not You
Young Griffo - Facade""".split("\n"))
VALID_TRACK_NAMES_STEREO = set("""ANiMAL - Clinic A
Actions - One Minute Smile
Al James - Schoolboy Facination
Angela Thomas Wade - Milk Cow Blues
Angels In Amplifiers - I'm Alright
Bobby Nobody - Stitch Up
Buitraker - Revo X
Cnoc An Tursa - Bannockburn
Detsky Sad - Walkie Talkie
Dreamers Of The Ghetto - Heavy Love
Fergessen - Back From The Start
Fergessen - The Wind
Johnny Lokke - Promises & Lies
Jokers, Jacks & Kings - Sea Of Leaves
Leaf - Summerghost
Matthew Entwistle - Dont You Ever
Music Delta - Country1
Music Delta - Punk
North To Alaska - All The Same
Phre The Eon - Everybody's Falling Apart
Punkdisco - Oral Hygiene
Sambasevam Shanmugam - Kaathaadi
Skelpolu - Resurrection
Steven Clark - Bounty
The Districts - Vermont
The Mountaineering Club - Mallory
The So So Glos - Emergency
Tim Taler - Stalker
Traffic Experiment - Sirens
Triviul - Dorothy""".split("\n"))


def process_subdir(args_tuple):
    """
    Process one MusDB18 track subdir. Load each stem WAV, resample to 44100 if needed,
    write to output (stereo: one .wav per stem; mono: .left.wav and .right.wav per stem).
    Returns list of rows for mixes.csv: {"path": rel_path, "is_train": bool, "is_mix": bool}.
    """
    subdir_path, output_root, mono = args_tuple
    track_name = basename(subdir_path.rstrip(os.sep)).split(".")[0]  # rstrip so basename isn't "" when path ends with /, split to align with previous approach
    validation_set_track_names = VALID_TRACK_NAMES_MONO if mono else VALID_TRACK_NAMES_STEREO
    is_train = track_name not in validation_set_track_names
    out_split = "train" if is_train else "valid"

    try:
        wav_files = [f for f in os.listdir(subdir_path) if f.lower().endswith(".wav")]
    except OSError as e:
        return []  # skip if unreadable

    rows = []
    for wav_file in wav_files:
        stem_name = splitext(wav_file)[0]
        stem_id = STEM_IDS.index(stem_name)
        is_mix = stem_id == 0

        wav_path = join(subdir_path, wav_file)
        try:
            data, sr = sf.read(wav_path, dtype="float64", always_2d=True)
        except Exception:
            continue

        if sr != TARGET_SAMPLE_RATE:
            data = librosa.resample(
                y=data.T, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE
            ).T
            sr = TARGET_SAMPLE_RATE

        output_path_stem = join(output_root, out_split, f"{track_name}.{stem_id}")

        if mono:
            left = data[:, 0]
            right = data[:, 1]
            path_left = output_path_stem + ".left.wav"
            path_right = output_path_stem + ".right.wav"
            sf.write(path_left, left, sr, format="WAV", subtype=f"PCM_{TARGET_BIT_DEPTH}")
            sf.write(path_right, right, sr, format="WAV", subtype=f"PCM_{TARGET_BIT_DEPTH}")
            rel_left = "./" + relpath(path_left, output_root).replace(os.sep, "/")
            rel_right = "./" + relpath(path_right, output_root).replace(os.sep, "/")
            rows.append({"path": rel_left, "is_train": is_train, "is_mix": is_mix})
            rows.append({"path": rel_right, "is_train": is_train, "is_mix": is_mix})
        else:
            path = output_path_stem + ".wav"
            sf.write(path, data, sr, format="WAV", subtype=f"PCM_{TARGET_BIT_DEPTH}")
            rel = "./" + relpath(path, output_root).replace(os.sep, "/")
            rows.append({"path": rel, "is_train": is_train, "is_mix": is_mix})

    return rows


def discover_subdir_paths(musdb18_dir):
    """Return list of subdir paths under train/ and test/ (normally 150 total)."""
    subdirs = []
    for split in ("train", "test"):
        split_dir = join(musdb18_dir, split)
        if not isdir(split_dir):
            continue
        for name in sorted(os.listdir(split_dir)):
            subdir = join(split_dir, name)
            if isdir(subdir):
                subdirs.append(subdir)
    assert len(subdirs) == 150, f"Expected 150 subdirs, got {len(subdirs)}"
    return subdirs


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MusDB18 to WAV and mixes.csv."
    )
    parser.add_argument(
        "--mono",
        action="store_true",
        help="Write left and right channels as separate WAV files per stem.",
    )
    parser.add_argument(
        "--musdb18_dir",
        type=str,
        required=True,
        help="Path to MusDB18 root (with train/ and test/ subdirs).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output root; dataset will be created under output_dir/musdb18mono or musdb18stereo.",
    )
    parser.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=max(1, multiprocessing.cpu_count() // 4),
        help="Number of worker processes.",
    )
    args = parser.parse_args()

    if not exists(args.musdb18_dir) or not isdir(args.musdb18_dir):
        raise RuntimeError(f"--musdb18_dir must be an existing directory: {args.musdb18_dir}")

    dataset_name = "musdb18mono" if args.mono else "musdb18stereo"
    output_root = join(args.output_dir, dataset_name)
    os.makedirs(join(output_root, "train"), exist_ok=True)
    os.makedirs(join(output_root, "valid"), exist_ok=True)

    subdir_paths = discover_subdir_paths(args.musdb18_dir)
    mixes_csv_path = join(output_root, "mixes.csv")

    # Prepare arguments for workers: (subdir_path, output_root, mono)
    worker_args = [
        (p, output_root, args.mono) for p in subdir_paths
    ]

    if args.jobs <= 1:
        all_rows = []
        for ta in tqdm(worker_args, desc="Preprocessing"):
            all_rows.extend(process_subdir(ta))
    else:
        with multiprocessing.Pool(processes=args.jobs) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(process_subdir, worker_args, chunksize=1),
                    total=len(worker_args),
                    desc="Preprocessing",
                )
            )
        all_rows = []
        for r in results:
            all_rows.extend(r)

    pd.DataFrame(all_rows, columns=MIXES_CSV_COLUMNS).to_csv(
        mixes_csv_path, index=False, mode="w", header=True
    )
    print(f"Wrote {len(all_rows)} rows to {mixes_csv_path}")


if __name__ == "__main__":
    main()
