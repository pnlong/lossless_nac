#!/usr/bin/env python3
"""
Generate LaTeX table from WandB runs (t5_lnac) for specified Trilobyte runs.

Uses RUNS_TO_INCLUDE dict (run_name -> dataset_name). For each run, extracts
val/loss, val/bpb, and max_bit_depth; computes compression rate. Outputs table
with columns: Dataset, Bit Depth, Cross Entropy Loss, BPB, Compression Rate (x).
"""

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

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

# Configuration constants
WANDB_PROJECT = "t5_lnac"
WANDB_ENTITY_DEFAULT = "znovack"
DEFAULT_EMA_TAU = 0.99


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
    """Format DataFrame as LaTeX table with booktabs style."""
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

    def fmt_num(x, decimals=3):
        if x is None or (isinstance(x, float) and np.isnan(x)) or (hasattr(pd, "isna") and pd.isna(x)):
            return "N/A"
        try:
            return f"{float(x):.{decimals}f}"
        except (TypeError, ValueError):
            return "N/A"

    if "Cross Entropy Loss" in df_latex.columns:
        df_latex["Cross Entropy Loss"] = df_latex["Cross Entropy Loss"].apply(lambda x: fmt_num(x, 4))
    if "BPB" in df_latex.columns:
        df_latex["BPB"] = df_latex["BPB"].apply(lambda x: fmt_num(x, 3))
    if "Compression Rate (x)" in df_latex.columns:
        df_latex["Compression Rate (x)"] = df_latex["Compression Rate (x)"].apply(lambda x: fmt_num(x, 2))
    if "Bit Depth" in df_latex.columns:
        df_latex["Bit Depth"] = df_latex["Bit Depth"].apply(
            lambda x: "N/A" if x is None or (isinstance(x, float) and np.isnan(x)) else str(int(x))
        )

    col_align = "l" + "r" * (len(df_latex.columns) - 1)
    latex_lines = [
        "% Requires \\usepackage{booktabs}",
        "",
        "\\begin{table}",
        "\\centering",
        f"\\begin{{tabular}}{{{col_align}}}",
        "\\toprule",
        " & ".join(escape_latex(c) for c in df_latex.columns) + " \\\\",
        "\\midrule",
    ]
    special = {"N/A", ""}
    for _, row in df_latex.iterrows():
        cells = []
        for v in row.values:
            s = str(v)
            cells.append(s if s in special else escape_latex(s))
        latex_lines.append(" & ".join(cells) + " \\\\")
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
    latex_lines.append("\\end{table}")
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

        val_loss = time_weighted_ema(val_loss_hist)
        val_bpb = time_weighted_ema(val_bpb_hist)
        compression_rate = 8 / val_bpb if val_bpb > 0 else np.nan

        runs_data.append({
            "Bit Depth": bit_depth,
            "Dataset": dataset_name,
            "Cross Entropy Loss": val_loss,
            "BPB": val_bpb,
            "Compression Rate (x)": compression_rate,
        })
        print(f"  Bit Depth: {bit_depth}, Loss: {val_loss:.4f}, BPB: {val_bpb:.3f}, CR: {compression_rate:.2f}")

    if not runs_data:
        print("No runs with valid data found.")
        return

    df = pd.DataFrame(runs_data)
    df = df[["Bit Depth", "Dataset", "Cross Entropy Loss", "BPB", "Compression Rate (x)"]]
    df = df.sort_values("Bit Depth", na_position="last").reset_index(drop=True)

    print("\n" + "=" * 80)
    print("LaTeX Table:")
    print("=" * 80)
    print(format_latex_table(df))
    print("=" * 80)

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nTable saved to {args.csv}")

    print("\nDone!")


if __name__ == "__main__":
    main()
