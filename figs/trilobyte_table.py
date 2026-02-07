#!/usr/bin/env python3
"""
Generate LaTeX table from WandB runs (t5_lnac) for specified Trilobyte runs.

Uses RUNS_TO_INCLUDE dict (run_name -> dataset_name). For each run, extracts
val/bpb and max_bit_depth; computes compression rate. Loads FLAC compression
rates from flac_eval_results.csv. Outputs table with columns: Bit Depth,
Dataset, Compression Rate (x), FLAC Compression Rate (x). Uses multirow for
Bit Depth. Missing FLAC data (e.g. 24-bit) shown as dash.
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import wandb

# User populates this: run_name -> dataset name for display
RUNS_TO_INCLUDE: Dict[str, str] = {
    "musdb18 mono 8-8": "MusDB18 Mono (All)",
    "bilobyte musdb18 (val mixes)": "MusDB18 Stereo (Mixes)",
    "bilobyte torr (16bit, no pro split)": "Torrent 16-bit (Amateur, Freeload)",
    "trilobyte torr (24 bit only, no pro split)": "Torrent 24-bit (Amateur, Freeload)",
    "yilobyte beethoven": "Beethoven",
    "bilobyte birdvox": "Birdvox",
    "trilobyte epidemic (24 bit)": "Epidemic Sound",
    "bilobyte librispeech": "LibriSpeech",
    "bilobyte LJSpeech ": "LJSpeech",
    "yilobyte sc09": "SC09",
    "bilobyte vctk": "VCTK",
    "yilobyte youtube_mix": "YouTube Mix",
}

DATASET_FANCY_NAME_TO_SHORT_NAME: Dict[str, str] = {
    "MusDB18 Mono (All)": "musdb18mono",
    "MusDB18 Stereo (Mixes)": "musdb18stereo_mixes",
    "Torrent 16-bit (Amateur, Freeload)": "torrent16b_amateur",
    "Torrent 24-bit (Amateur, Freeload)": "torrent24b_amateur",
    "Beethoven": "beethoven",
    "Birdvox": "birdvox",
    "Epidemic Sound": "epidemic",
    "LibriSpeech": "librispeech",
    "LJSpeech": "ljspeech",
    "SC09": "sc09",
    "VCTK": "vctk",
    "YouTube Mix": "youtube_mix",
}

# Configuration constants
WANDB_PROJECT = "t5_lnac"
WANDB_ENTITY_DEFAULT = "znovack"
DEFAULT_EMA_TAU = 0.99
FLAC_EVAL_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "flac_eval_results.csv")
FLAC_COMPRESSION_LEVEL = 5  # Default FLAC level; change if needed for 24-bit experiments

# Epidemic Sound was run in 24-bit but dataset is likely 16-bit transcoded; use estimated rate.
# Update this constant when re-running with corrected bit depth.
ESTIMATED_EPIDEMIC_SOUND_16_BIT_COMPRESSION_RATE = 3.4042553191


def _summary_to_dict(summary) -> dict:
    """Safely convert WandB summary to dict. Handles broken _json_dict (string) case."""
    if summary is None:
        return {}
    try:
        return dict(summary) if hasattr(summary, "items") else {}
    except (AttributeError, TypeError):
        pass
    # WandB HTTPSummary sometimes has _json_dict as a JSON string instead of dict
    jd = getattr(summary, "_json_dict", None)
    if isinstance(jd, str):
        try:
            return json.loads(jd)
        except json.JSONDecodeError:
            pass
    return {}


def _config_to_dict(config) -> dict:
    """Safely convert WandB config to dict. Config is often a JSON string."""
    if config is None:
        return {}
    if isinstance(config, str):
        try:
            return json.loads(config)
        except json.JSONDecodeError:
            return {}
    try:
        return dict(config) if hasattr(config, "items") else {}
    except Exception:
        return {}


def _unwrap_value(v):
    """Unwrap WandB config values like {"value": X} -> X."""
    if isinstance(v, dict) and "value" in v and len(v) == 1:
        return v["value"]
    return v


def _debug_config(run, dataset_name: str) -> None:
    """Print config structure to help find max_bit_depth."""
    config_raw = getattr(run, "config", None)
    print(f"\n  [DEBUG] Config for {dataset_name!r} (run {run.name or run.id}):")
    print(f"    Run URL: https://wandb.ai/{run.entity or 'znovack'}/{run.project}/{run.id}")
    print(f"    config type: {type(config_raw).__name__}")
    if config_raw is None:
        print("    config is None")
        return
    flat = _config_to_dict(config_raw)
    if not flat:
        print("    config parsed to empty dict")
        if isinstance(config_raw, str):
            print(f"    config (first 300 chars): {config_raw[:300]!r}...")
        return
    print(f"    Top-level keys ({len(flat)}): {sorted(flat.keys())}")
    # Keys that might contain bit depth
    for k, v in flat.items():
        kstr = str(k).lower()
        if "bit" in kstr or "depth" in kstr or "max" in kstr:
            unwrapped = _unwrap_value(v)
            print(f"    {k!r} = {v!r} -> unwrapped: {unwrapped!r}")
    # Recurse into dict values (e.g. _wandb) for nested bit/depth keys
    for k, v in flat.items():
        if isinstance(v, dict) and k != "_wandb":  # skip huge _wandb blob
            for k2, v2 in v.items():
                k2str = str(k2).lower()
                if "bit" in k2str or "depth" in k2str or "max" in k2str:
                    print(f"    {k!r}.{k2!r} = {v2!r}")
    summary = getattr(run, "summary", None)
    sd = _summary_to_dict(summary)
    if sd:
        for k in sd:
            if "bit" in str(k).lower() or "depth" in str(k).lower() or "max" in str(k).lower():
                print(f"    summary[{k!r}] = {sd[k]!r}")


def _get_flat_config(run) -> dict:
    """Get config as a flat dict; WandB/Lightning may nest or use different key names."""
    config_raw = getattr(run, "config", None)
    flat = _config_to_dict(config_raw)

    def find_key(d: dict, key: str, default=None):
        if key in d and d[key] is not None:
            return d[key]
        for v in d.values():
            if isinstance(v, dict) and "value" not in v:  # skip {"value": X} wrappers
                found = find_key(v, key, None)
                if found is not None:
                    return found
        return default

    def get_val(d: dict, key: str):
        v = d.get(key) or find_key(d, key)
        return _unwrap_value(v) if v is not None else None

    result = {}
    for k, v in flat.items():
        key = k.rstrip("_") if isinstance(k, str) else k
        result[key] = _unwrap_value(v) if isinstance(v, dict) and "value" in v else v
    # Try various key names (underscore, hyphen, etc.)
    max_bit_depth = (
        get_val(flat, "max_bit_depth")
        or get_val(flat, "max-bit-depth")
    )
    if max_bit_depth is None:
        sd = _summary_to_dict(getattr(run, "summary", None))
        if sd:
            max_bit_depth = sd.get("max_bit_depth") or sd.get("max-bit-depth")
    if max_bit_depth is None:
        # Fallback: scan all keys for variations
        def scan_for_max_bit_depth(d, prefix=""):
            for k, v in (d.items() if hasattr(d, "items") else []):
                key = str(k).lower().replace("-", "_").replace(" ", "_")
                if "max" in key and "bit" in key and "depth" in key and v is not None:
                    return _unwrap_value(v)
                if isinstance(v, dict) and "value" not in v:
                    found = scan_for_max_bit_depth(v)
                    if found is not None:
                        return found
            return None
        max_bit_depth = scan_for_max_bit_depth(flat)
    result["max_bit_depth"] = max_bit_depth
    return result


def _parse_bit_depth(max_bit_depth) -> Optional[int]:
    """Parse max_bit_depth to integer bit depth. For '16-8' or '16_8', use only the part before the separator."""
    if max_bit_depth is None:
        return None
    if isinstance(max_bit_depth, int):
        return max_bit_depth
    s = str(max_bit_depth).strip()
    if not s:
        return None
    # For "16-8" or "16_8", take only what comes before the hyphen/underscore
    for sep in ("-", "_"):
        if sep in s:
            try:
                return int(s.split(sep)[0])
            except (ValueError, IndexError):
                return None
    try:
        return int(s)
    except ValueError:
        return None


def _load_flac_compression_rates(
    path: Optional[str] = None,
    flac_level: Optional[int] = None,
) -> Dict[Tuple[str, int], float]:
    """
    Load FLAC compression rates from flac_eval_results.csv.
    Returns dict: (dataset_short_name, bit_depth) -> overall_compression_rate.
    Missing entries (e.g. 24-bit) will show as dash in table.
    Uses FLAC with constant, fixed, and verbatim subframes disabled.
    """
    csv_path = path or FLAC_EVAL_RESULTS_PATH
    level = flac_level if flac_level is not None else FLAC_COMPRESSION_LEVEL
    if not os.path.exists(csv_path):
        print(f"Warning: FLAC results not found at {csv_path}")
        return {}

    flac_df = pd.read_csv(csv_path)
    flac_df = flac_df[flac_df["flac_compression_level"] == level]
    # Normalize boolean columns (CSV may store "True"/"False" as strings)
    for col in ["disable_constant_subframes", "disable_fixed_subframes", "disable_verbatim_subframes"]:
        if col in flac_df.columns:
            flac_df[col] = flac_df[col].astype(str).str.lower().isin(("true", "1"))
    flac_df = flac_df[flac_df["disable_constant_subframes"] == True]
    flac_df = flac_df[flac_df["disable_fixed_subframes"] == True]
    flac_df = flac_df[flac_df["disable_verbatim_subframes"] == True]
    if flac_df.empty:
        return {}

    result = {}
    for (dataset, bit_depth), group in flac_df.groupby(["dataset", "bit_depth"]):
        group = group.sort_values("datetime", ascending=False)
        rate = group.iloc[0]["overall_compression_rate"]
        result[(str(dataset), int(bit_depth))] = float(rate)
    return result


def extract_metric_history(run, metric_name: str) -> Optional[pd.Series]:
    """Extract metric history from a WandB run."""
    try:
        history = run.history(keys=[metric_name])
        if history.empty or metric_name not in history.columns:
            return None
        history = history.dropna(subset=[metric_name])
        if history.empty:
            return None
        if "_step" in history.columns:
            history = history.set_index("_step")
        elif "step" in history.columns:
            history = history.set_index("step")
        else:
            history.index = range(len(history))
        return history[metric_name]
    except Exception as e:
        print(f"Warning: Could not extract {metric_name} from run {run.id}: {e}")
        return None


def time_weighted_ema(
    values: pd.Series, times: Optional[pd.Series] = None, tau: float = DEFAULT_EMA_TAU
) -> float:
    """Time-weighted exponential moving average."""
    if len(values) == 0:
        return np.nan
    if len(values) == 1:
        return float(values.iloc[0])
    if times is None:
        times = values.index
    values_arr = values.values
    times_arr = times.values
    ema = values_arr[0]
    for i in range(1, len(values_arr)):
        dt = times_arr[i] - times_arr[i - 1]
        if dt <= 0:
            dt = 1.0
        alpha = 1 - np.exp(-dt / tau)
        ema = alpha * values_arr[i] + (1 - alpha) * ema
    return float(ema)


def _parse_timestamp(t) -> datetime:
    """Parse run timestamp for sorting (most recent first)."""
    if t is None:
        return datetime.min
    if isinstance(t, str):
        try:
            return datetime.fromisoformat(t.replace("Z", "+00:00"))
        except Exception:
            return datetime.min
    return t


def format_latex_table(df: pd.DataFrame) -> str:
    """Format DataFrame as LaTeX table with booktabs style and multirow for Bit Depth."""
    df_latex = df.copy()

    def escape_latex(text):
        if pd.isna(text):
            return ""
        text = str(text)
        for a, b in [
            ("&", r"\&"),
            ("%", r"\%"),
            ("$", r"\$"),
            ("#", r"\#"),
            ("^", r"\^{}"),
            ("_", r"\_"),
            ("{", r"\{"),
            ("}", r"\}"),
            ("~", r"\textasciitilde{}"),
            ("\\", r"\textbackslash{}"),
        ]:
            text = text.replace(a, b)
        return text

    def fmt_num(x, decimals=2):
        if x is None or (isinstance(x, float) and np.isnan(x)) or (hasattr(pd, "isna") and pd.isna(x)):
            return None
        try:
            return f"{float(x):.{decimals}f}"
        except (TypeError, ValueError):
            return None

    def fmt_cr_cell(cr_val, flac_val):
        """Format compression rate cells, bolding the greater of the two."""
        cr_str = fmt_num(cr_val, 2) if not pd.isna(cr_val) else "—"
        flac_str = fmt_num(flac_val, 2) if not pd.isna(flac_val) else "—"
        cr_num = float(cr_val) if not pd.isna(cr_val) else None
        flac_num = float(flac_val) if not pd.isna(flac_val) else None
        if cr_num is not None and flac_num is not None:
            if cr_num >= flac_num:
                cr_str = f"\\textbf{{{cr_str}}}"
            else:
                flac_str = f"\\textbf{{{flac_str}}}"
        elif cr_num is not None:
            cr_str = f"\\textbf{{{cr_str}}}"
        elif flac_num is not None:
            flac_str = f"\\textbf{{{flac_str}}}"
        return cr_str, flac_str

    latex_lines = [
        "% Requires \\usepackage{booktabs} and \\usepackage{multirow}",
        "",
        "   \\begin{tabular}{cl|cc}",
        "        \\toprule",
        "        \\textbf{Bit Depth} & \\textbf{Dataset} & \\textbf{Trilobyte Compression Rate} (x) & \\textbf{FLAC Compression Rate} (x) \\\\",
        "        \\midrule",
    ]

    # Group by bit depth and emit rows with multirow
    for bit_depth, group in df_latex.groupby("Bit Depth", sort=True):
        n_rows = len(group)
        bit_str = str(int(bit_depth)) if bit_depth is not None and not pd.isna(bit_depth) else "N/A"
        for i, (_, row) in enumerate(group.iterrows()):
            is_estimated = row.get("is_estimated", False)
            if is_estimated:
                latex_lines.append(
                    "        % Epidemic Sound: estimated (dataset likely 16-bit transcoded to 24-bit). "
                    "Update ESTIMATED_EPIDEMIC_SOUND_16_BIT_COMPRESSION_RATE in trilobyte_table.py when re-running."
                )
            dataset = escape_latex(row["Dataset"])
            cr_str, flac_str = fmt_cr_cell(
                row["Compression Rate (x)"],
                row["FLAC Compression Rate (x)"],
            )
            if i == 0:
                bit_cell = f"\\multirow{{{n_rows}}}{{*}}{{{bit_str}}}"
            else:
                bit_cell = ""
            row_str = f"         {bit_cell} & {dataset} & {cr_str} & {flac_str} \\\\"
            latex_lines.append(row_str)
        latex_lines.append("        \\midrule")

    # Remove trailing midrule before bottomrule
    if latex_lines[-1] == "        \\midrule":
        latex_lines.pop()
    latex_lines.append("        \\bottomrule")
    latex_lines.append("    \\end{tabular}")
    return "\n".join(latex_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate LaTeX table from WandB t5_lnac runs specified in RUNS_TO_INCLUDE"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Optional: save table DataFrame to this CSV path",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print config keys and structure when bit depth is not found",
    )
    parser.add_argument(
        "--flac-csv",
        type=str,
        default=None,
        help=f"Path to FLAC eval results CSV (default: {FLAC_EVAL_RESULTS_PATH})",
    )
    parser.add_argument(
        "--flac-level",
        type=int,
        default=FLAC_COMPRESSION_LEVEL,
        help=f"FLAC compression level to use (default: {FLAC_COMPRESSION_LEVEL})",
    )
    args = parser.parse_args()

    if not RUNS_TO_INCLUDE:
        print("RUNS_TO_INCLUDE is empty. Populate it with run_name -> dataset_name pairs.")
        return

    print("Connecting to WandB...")
    api = wandb.Api()
    entity = os.environ.get("WANDB_ENTITY", WANDB_ENTITY_DEFAULT)

    print(f"Fetching runs from project: {WANDB_PROJECT}")
    if entity:
        print(f"Entity: {entity}")

    try:
        runs = list(api.runs(f"{entity}/{WANDB_PROJECT}") if entity else api.runs(WANDB_PROJECT))
    except Exception as e:
        print(f"Error fetching runs: {e}")
        try:
            runs = list(api.runs(WANDB_PROJECT))
        except Exception as e2:
            print(f"Error: Could not fetch runs: {e2}")
            return

    # Index runs by name; if duplicates, keep most recent
    runs_by_name: Dict[str, List] = defaultdict(list)
    for run in runs:
        name = run.name or run.id
        runs_by_name[name].append(run)

    flac_rates = _load_flac_compression_rates(
        path=args.flac_csv,
        flac_level=args.flac_level,
    )
    runs_data = []
    for run_name, dataset_name in RUNS_TO_INCLUDE.items():
        candidates = runs_by_name.get(run_name, [])
        if not candidates:
            print(f"Warning: No run found with name {run_name!r}")
            continue
        # Pick most recent by timestamp
        run = max(candidates, key=lambda r: _parse_timestamp(getattr(r, "created_at", None)))
        print(f"Processing: {run_name} -> {dataset_name}")

        config = _get_flat_config(run)
        max_bit_depth = config.get("max_bit_depth")
        bit_depth = _parse_bit_depth(max_bit_depth)
        if bit_depth is None:
            _debug_config(run, dataset_name)

        val_loss_hist = extract_metric_history(run, "val/loss")
        val_bpb_hist = extract_metric_history(run, "val/bpb")

        if val_loss_hist is None:
            print(f"  Skipping: no val/loss")
            continue
        if val_bpb_hist is None:
            print(f"  Skipping: no val/bpb")
            continue

        val_bpb = time_weighted_ema(val_bpb_hist)
        compression_rate = 8 / val_bpb if val_bpb > 0 else np.nan

        # Epidemic Sound: override to 16-bit with estimated rate (dataset likely transcoded 16→24)
        if dataset_name == "Epidemic Sound":
            bit_depth = 16
            compression_rate = ESTIMATED_EPIDEMIC_SOUND_16_BIT_COMPRESSION_RATE

        short_name = DATASET_FANCY_NAME_TO_SHORT_NAME.get(dataset_name)
        flac_rate = None
        if short_name and bit_depth is not None:
            flac_rate = flac_rates.get((short_name, bit_depth))

        runs_data.append({
            "Bit Depth": bit_depth,
            "Dataset": dataset_name,
            "Compression Rate (x)": compression_rate,
            "FLAC Compression Rate (x)": flac_rate,
            "is_estimated": dataset_name == "Epidemic Sound",
        })
        flac_str = f"{flac_rate:.2f}" if flac_rate is not None else "—"
        est_note = " (estimated)" if dataset_name == "Epidemic Sound" else ""
        print(f"  Bit Depth: {bit_depth}, CR: {compression_rate:.2f}, FLAC: {flac_str}{est_note}")

    if not runs_data:
        print("No runs with valid data found.")
        return

    df = pd.DataFrame(runs_data)
    df = df[["Bit Depth", "Dataset", "Compression Rate (x)", "FLAC Compression Rate (x)", "is_estimated"]]
    df = df.sort_values(["Bit Depth", "Dataset"], na_position="last").reset_index(drop=True)

    print("\n" + "=" * 80)
    print("LaTeX Table:")
    print("=" * 80)
    print(format_latex_table(df))
    print("=" * 80)

    if args.csv:
        df[["Bit Depth", "Dataset", "Compression Rate (x)", "FLAC Compression Rate (x)"]].to_csv(
            args.csv, index=False
        )
        print(f"\nTable saved to {args.csv}")

    print("\nDone!")


if __name__ == "__main__":
    main()
