#!/usr/bin/env python3
"""
Overview statistics for Torrent dataset subsets (amateur, freeload, pro) at 16- and 24-bit.
Uses Torrent16BDataset and Torrent24BDataset from flac_eval.
"""

import soundfile as sf
from collections import Counter

# Import dataset classes; flac_eval uses TORRENT_DATA_DATA_DIR internally
from flac_eval import Torrent16BDataset, Torrent24BDataset


SUBSETS = ("amateur", "freeload", "pro")
BIT_DEPTHS = (16, 24)


def get_dataset(subset: str, bit_depth: int):
    """Return the Torrent dataset for the given subset and bit depth."""
    if bit_depth == 16:
        return Torrent16BDataset(subset=subset)
    elif bit_depth == 24:
        return Torrent24BDataset(subset=subset)
    raise ValueError(f"Unsupported bit_depth: {bit_depth}")


def collect_stats(paths):
    """For a list of audio paths, return (sample_rates_counter, total_duration_sec, error_count)."""
    sample_rates = []
    total_duration_sec = 0.0
    errors = 0
    for path in paths:
        try:
            info = sf.info(path)
            sample_rates.append(info.samplerate)
            total_duration_sec += info.duration
        except Exception:
            errors += 1
    return Counter(sample_rates), total_duration_sec, errors


def print_section(label, sr_counter, n, total_sec, err, in_khz=True):
    """Print (A) sample rates, (B) file count, (C) total duration for a section."""
    print(f"\n--- {label} ---")
    print(f"  (A) Sample rates (% of files):")
    if sr_counter:
        for sr, count in sorted(sr_counter.items()):
            pct = 100.0 * count / n if n else 0
            if in_khz:
                print(f"      {sr / 1000:.1f} kHz: {pct:.1f}% ({count} files)")
            else:
                print(f"      {sr} Hz: {pct:.1f}% ({count} files)")
    else:
        print("      (none or all errors)")
    if err:
        print(f"      (errors reading: {err} files)")
    print(f"  (B) File count: {n}")
    hours = total_sec / 3600.0
    print(f"  (C) Total duration: {hours:.2f} hours")


def main():
    print("Torrent data overview")
    print("=" * 60)

    # Accumulators per bit depth for combined summaries
    by_bit = {
        16: {"sr": Counter(), "files": 0, "duration_sec": 0.0, "errors": 0},
        24: {"sr": Counter(), "files": 0, "duration_sec": 0.0, "errors": 0},
    }

    for subset in SUBSETS:
        for bit_depth in BIT_DEPTHS:
            label = f"{subset} / {bit_depth}-bit"
            try:
                ds = get_dataset(subset, bit_depth)
            except ValueError as e:
                print(f"\n{label}: skipped - {e}")
                continue

            paths = ds.paths
            n = len(paths)
            sr_counter, total_sec, err = collect_stats(paths)

            print_section(label, sr_counter, n, total_sec, err, in_khz=True)

            acc = by_bit[bit_depth]
            acc["sr"].update(sr_counter)
            acc["files"] += n
            acc["duration_sec"] += total_sec
            acc["errors"] += err

    # Summaries: one for 16-bit (all subsets), one for 24-bit (all subsets)
    for bit_depth in BIT_DEPTHS:
        acc = by_bit[bit_depth]
        print_section(
            f"Summary (all subsets, {bit_depth}-bit)",
            acc["sr"],
            acc["files"],
            acc["duration_sec"],
            acc["errors"],
            in_khz=True,
        )

    print("\nDone.")


if __name__ == "__main__":
    main()
