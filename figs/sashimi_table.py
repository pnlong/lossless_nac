#!/usr/bin/env python3
"""
Generate LaTeX table from WandB runs comparing Sashimi model configurations.

This script fetches Sashimi model runs from WandB, extracts BPB metrics,
applies time-weighted EMA smoothing, calculates compression rates, and
generates a LaTeX table comparing different model configurations.
"""

import argparse
import json
import os
import pandas as pd
import numpy as np
import wandb
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime


# Configuration constants
WANDB_PROJECT = "LNAC"
FLAC_EVAL_RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "flac_eval_results.csv")
FLAC_COMPRESSION_LEVEL = 5
DEFAULT_EMA_TAU = 0.99  # Time constant for EMA smoothing (epochs)


def _unwrap_config_value(v):
    """Unwrap WandB config values like {"value": X} -> X."""
    if isinstance(v, dict) and "value" in v and len(v) == 1:
        return v["value"]
    return v


def extract_config_params(run) -> Dict:
    """
    Extract configuration parameters from a WandB run.
    
    Args:
        run: WandB run object
        
    Returns:
        Dictionary with config parameters: stereo, interleaving_strategy, dml, bits, timestamp
    """
    config_raw = run.config
    # WandB can return config as a JSON string; parse to dict
    if isinstance(config_raw, str):
        try:
            config = json.loads(config_raw)
        except json.JSONDecodeError:
            config = {}
    elif config_raw is not None and hasattr(config_raw, "items"):
        try:
            config = dict(config_raw)
        except (TypeError, ValueError):
            config = {}
    else:
        config = {} if config_raw is None else {}
    params = {
        'run_id': run.id,
        'run_name': run.name,
        'timestamp': run.created_at if hasattr(run, 'created_at') else None,
    }
    # Unwrap WandB {"value": X} so we read actual dataset/model config
    dataset_cfg = _unwrap_config_value(config.get('dataset'))
    model_cfg = _unwrap_config_value(config.get('model'))
    if not isinstance(dataset_cfg, dict):
        dataset_cfg = {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    
    # Extract stereo/mono
    is_stereo = False
    if dataset_cfg:
        is_stereo = dataset_cfg.get('is_stereo', False)
        if isinstance(is_stereo, dict):
            is_stereo = _unwrap_config_value(is_stereo)
        if not isinstance(is_stereo, bool) and isinstance(dataset_cfg.get('is_stereo'), str):
            is_stereo = 'stereo' in str(dataset_cfg.get('is_stereo', '')).lower()
    if not is_stereo and run.name:
        is_stereo = 'stereo' in run.name.lower()
    params['stereo'] = bool(is_stereo)
    
    # Extract interleaving strategy (determines Blocking-N: temporal=1, blocking-4=4, etc.)
    interleaving_strategy = ""
    if is_stereo and dataset_cfg:
        interleaving_strategy = dataset_cfg.get('interleaving_strategy', 'temporal') or 'temporal'
        if isinstance(interleaving_strategy, dict):
            interleaving_strategy = _unwrap_config_value(interleaving_strategy) or 'temporal'
        interleaving_strategy = str(interleaving_strategy).strip()
    params['interleaving_strategy'] = interleaving_strategy if is_stereo else ""
    
    # Extract DML usage
    dml = False
    if model_cfg:
        out_head = model_cfg.get('output_head', '')
        if isinstance(out_head, dict):
            out_head = _unwrap_config_value(out_head) or ''
        dml = str(out_head).lower() == 'dml'
    if not dml and run.name:
        dml = 'dml' in run.name.lower()
    params['dml'] = dml
    
    # Extract bit depth
    bits = None
    if dataset_cfg:
        bits = dataset_cfg.get('bits', None)
        if isinstance(bits, dict):
            bits = _unwrap_config_value(bits)
    if bits is None:
        if run.name and '8bit' in run.name.lower():
            bits = 8
        elif run.name and '16bit' in run.name.lower():
            bits = 16
        else:
            bits = 16  # Default assumption
    params['bits'] = bits
    
    # Extract d_model
    d_model = None
    if model_cfg:
        d_model = model_cfg.get('d_model', None)
        if isinstance(d_model, dict):
            d_model = _unwrap_config_value(d_model)
    params['d_model'] = d_model
    
    # Extract sample_len
    sample_len = None
    if dataset_cfg:
        sample_len = dataset_cfg.get('sample_len', None)
        if isinstance(sample_len, dict):
            sample_len = _unwrap_config_value(sample_len)
    params['sample_len'] = sample_len
    
    return params


def extract_bpb_history(run, metric_name: str) -> Optional[pd.Series]:
    """
    Extract BPB history from a WandB run.
    
    Args:
        run: WandB run object
        metric_name: Name of the metric (e.g., 'val/bpb' or 'test/bpb')
        
    Returns:
        pandas Series with step/epoch as index and BPB values, or None if metric doesn't exist
    """
    try:
        history = run.history(keys=[metric_name])
        if history.empty or metric_name not in history.columns:
            return None
        
        # Remove NaN values
        history = history.dropna(subset=[metric_name])
        if history.empty:
            return None
        
        # Use step or _step as index if available, otherwise use row number
        if '_step' in history.columns:
            history = history.set_index('_step')
        elif 'step' in history.columns:
            history = history.set_index('step')
        else:
            history.index = range(len(history))
        
        return history[metric_name]
    except Exception as e:
        print(f"Warning: Could not extract {metric_name} from run {run.id}: {e}")
        return None


def time_weighted_ema(values: pd.Series, times: Optional[pd.Series] = None, tau: float = DEFAULT_EMA_TAU) -> float:
    """
    Apply time-weighted exponential moving average smoothing.
    
    Formula: EMA_t = α * value_t + (1 - α) * EMA_{t-1}
    where α = 1 - exp(-Δt / τ)
    
    Args:
        values: Series of values to smooth
        times: Series of time values (if None, uses index as time)
        tau: Time constant for smoothing
        
    Returns:
        Final smoothed value
    """
    if len(values) == 0:
        return np.nan
    
    if len(values) == 1:
        return float(values.iloc[0])
    
    # Use index as time if times not provided
    if times is None:
        times = values.index
    
    # Convert to numpy arrays for easier computation
    values_arr = values.values
    times_arr = times.values
    
    # Initialize EMA with first value
    ema = values_arr[0]
    
    # Apply EMA with time weighting
    for i in range(1, len(values_arr)):
        dt = times_arr[i] - times_arr[i-1]
        if dt <= 0:
            dt = 1.0  # Handle non-increasing times
        
        # Calculate time-weighted alpha
        alpha = 1 - np.exp(-dt / tau)
        
        # Update EMA
        ema = alpha * values_arr[i] + (1 - alpha) * ema
    
    return float(ema)


def calculate_compression_rate(bpb: float, bit_depth: int) -> float:
    """
    Calculate compression rate from BPB and bit depth.
    
    Args:
        bpb: Bits per byte
        bit_depth: Bit depth (8 or 16)
        
    Returns:
        Compression rate (bit_depth / bpb)
    """
    if np.isnan(bpb) or bpb <= 0:
        return np.nan
    return bit_depth / bpb


def deduplicate_runs(runs_data: List[Dict]) -> List[Dict]:
    """
    Deduplicate runs with equivalent configurations, keeping the most recent.
    Shows debugging info about what differs between duplicate configs.
    
    Args:
        runs_data: List of dictionaries, each containing run data with config params
        
    Returns:
        Deduplicated list of runs (most recent kept for each config)
    """
    # Group runs by configuration signature
    config_groups = defaultdict(list)
    
    for run_data in runs_data:
        # Create configuration signature (stereo, blocking-N, bits, dml)
        # Format blocking-N value for comparison
        blocking_strategy = run_data.get('interleaving_strategy', '')
        blocking_n = None
        if blocking_strategy:
            blocking_str = str(blocking_strategy).lower()
            if blocking_str == 'temporal':
                blocking_n = 1
            elif blocking_str.startswith('blocking'):
                if '-' in blocking_str:
                    parts = blocking_str.split('-')
                    if len(parts) > 1:
                        try:
                            blocking_n = int(parts[-1])
                        except:
                            # If can't parse, infer from sample_len
                            sample_len = run_data.get('sample_len')
                            blocking_n = sample_len if sample_len is not None else 1
                else:
                    # Just "blocking" without number - infer from sample_len
                    sample_len = run_data.get('sample_len')
                    blocking_n = sample_len if sample_len is not None else 1
            else:
                blocking_n = 1
        
        config_sig = (
            run_data.get('stereo', False),
            blocking_n,
            run_data.get('bits', None),
            run_data.get('dml', False)
        )
        config_groups[config_sig].append(run_data)
    
    # For each group, keep only the most recent run
    deduplicated = []
    for config_sig, group in config_groups.items():
        if len(group) == 1:
            deduplicated.append(group[0])
        else:
            # Sort by timestamp (most recent first)
            # Handle None timestamps by putting them last
            def get_timestamp_key(x):
                ts = x.get('timestamp')
                if ts is None:
                    return datetime.min
                # Handle string timestamps
                if isinstance(ts, str):
                    try:
                        return datetime.fromisoformat(ts.replace('Z', '+00:00'))
                    except:
                        try:
                            return datetime.strptime(ts, '%Y-%m-%d %H:%M:%S')
                        except:
                            return datetime.min
                # Handle datetime objects
                if isinstance(ts, datetime):
                    return ts
                return datetime.min
            
            # Sort by timestamp (most recent first), but prefer "temporal" over "blocking"
            def get_sort_key(x):
                timestamp_key = get_timestamp_key(x)
                # Prefer temporal over blocking when timestamps are similar
                strategy = str(x.get('interleaving_strategy', '')).lower()
                # temporal gets priority 0, blocking gets priority 1
                temporal_priority = 0 if strategy == 'temporal' else 1
                # Convert timestamp to comparable value
                if isinstance(timestamp_key, datetime):
                    # Use negative timestamp so more recent (larger) comes first
                    timestamp_val = -timestamp_key.timestamp()
                else:
                    timestamp_val = float('inf')
                # Return tuple: (temporal_priority, timestamp_val) so temporal comes first
                # and more recent comes first within same priority
                return (temporal_priority, timestamp_val)
            
            group_sorted = sorted(
                group,
                key=get_sort_key
            )
            most_recent = group_sorted[0]
            deduplicated.append(most_recent)
            
            # Log deduplication with debugging info
            print(f"\n{'='*80}")
            print(f"Found {len(group)} runs with equivalent config signature: {config_sig}")
            print(f"  (stereo={config_sig[0]}, blocking-N={config_sig[1]}, bits={config_sig[2]}, dml={config_sig[3]})")
            print(f"\n  Keeping: {most_recent.get('run_name', most_recent.get('run_id'))}")
            print(f"    Timestamp: {most_recent.get('timestamp')}")
            val_bpb = most_recent.get('val_bpb')
            test_bpb = most_recent.get('test_bpb')
            val_bpb_str = f"{val_bpb:.3f}" if val_bpb is not None and not pd.isna(val_bpb) else "N/A"
            test_bpb_str = f"{test_bpb:.3f}" if test_bpb is not None and not pd.isna(test_bpb) else "N/A"
            print(f"    Val BPB: {val_bpb_str}")
            print(f"    Test BPB: {test_bpb_str}")
            
            # Show what's different between runs
            all_keys = set()
            for run in group:
                all_keys.update(run.keys())
            
            # Find keys that differ between runs
            differing_keys = []
            for key in all_keys:
                if key in ['run_id', 'run_name', 'timestamp', 'val_bpb', 'test_bpb', 'val_compression', 'test_compression']:
                    continue  # Skip these as they're expected to differ
                values = [run.get(key) for run in group]
                if len(set(str(v) for v in values if v is not None)) > 1:
                    differing_keys.append(key)
            
            if differing_keys:
                print(f"\n  Differences found in these fields:")
                for key in differing_keys:
                    print(f"    {key}:")
                    for run in group:
                        val = run.get(key, 'N/A')
                        print(f"      {run.get('run_name', run.get('run_id'))}: {val}")
            
            print(f"\n  Discarding:")
            for other in group_sorted[1:]:
                print(f"    {other.get('run_name', other.get('run_id'))} "
                      f"(timestamp: {other.get('timestamp')})")
            print(f"{'='*80}\n")
    
    return deduplicated


def _get_max_blocking_n(bits: int, channels: str, dml: bool) -> int:
    """Max blocking-n for (bits, channels, dml). 16-bit stereo non-DML: 2048; else 8192."""
    if bits == 16 and channels == "Stereo" and not dml:
        return 2048
    return 8192


def _filter_blocking_n_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only blocking-n in {1, 512, max_n} per (Bit Depth, Channels, DML).
    max_n = 2048 for 16-bit stereo non-DML, else 8192.
    """
    def _blocking_n_to_int(x):
        if pd.isna(x) or x == '' or x == '--':
            return None
        s = str(x).replace(',', '')
        return int(s) if s.isdigit() else None

    rows_to_keep = []
    for idx, row in df.iterrows():
        bits = row['Bit Depth']
        channels = row['Channels']
        dml = row['DML']
        bn = _blocking_n_to_int(row['Blocking-N'])

        if bn is None:  # Mono: keep (-- is the only value)
            rows_to_keep.append(idx)
            continue

        max_n = _get_max_blocking_n(bits, channels, dml)
        if bn in (1, 512, max_n):
            rows_to_keep.append(idx)

    return df.loc[rows_to_keep].copy()


def _load_flac_compression_rates() -> Dict[Tuple[int, str], float]:
    """
    Load FLAC level 5 compression rates for musdb18mono and musdb18stereo.
    Returns dict: (bit_depth, channels) -> mean_compression_rate
    """
    if not os.path.exists(FLAC_EVAL_RESULTS_PATH):
        print(f"Warning: FLAC results not found at {FLAC_EVAL_RESULTS_PATH}")
        return {}

    flac_df = pd.read_csv(FLAC_EVAL_RESULTS_PATH)
    flac_df = flac_df[
        (flac_df["flac_compression_level"] == FLAC_COMPRESSION_LEVEL)
        & (flac_df["dataset"].isin(["musdb18mono", "musdb18stereo"]))
        & (flac_df["disable_constant_subframes"] == True)
        & (flac_df["disable_fixed_subframes"] == True)
        & (flac_df["disable_verbatim_subframes"] == True)
    ]
    if flac_df.empty:
        return {}

    result = {}
    for (dataset, bit_depth), group in flac_df.groupby(["dataset", "bit_depth"]):
        # Take most recent if multiple rows
        group = group.sort_values("datetime", ascending=False)
        rate = group.iloc[0]["mean_compression_rate"]
        channels = "Stereo" if "stereo" in dataset else "Mono"
        result[(int(bit_depth), channels)] = float(rate)
    return result


def _format_blocking_n(value: str) -> str:
    """Format Blocking-n for display: add thousands separators (e.g., 8192 -> 8,192)."""
    if not value or value == '--':
        return '--'
    try:
        n = int(value)
        return f"{n:,}"
    except (ValueError, TypeError):
        return str(value)


def _identify_best_compression_per_group(df: pd.DataFrame, comp_col: str) -> set:
    """
    Identify row indices with best compression rate in each group.
    Groups: (Bit Depth, Channels) - e.g., (8, Mono), (8, Stereo), (16, Mono), (16, Stereo).
    Best = highest compression rate.
    """
    best_indices = set()
    for (bits, channels), group in df.groupby(['Bit Depth', 'Channels']):
        valid = group[comp_col].dropna()
        if len(valid) == 0:
            continue
        best_val = valid.max()
        best_idx = group[group[comp_col] == best_val].index
        best_indices.update(best_idx)
    return best_indices


def format_latex_table(df: pd.DataFrame, use_val: bool = False, flac_rates: Optional[Dict[Tuple[int, str], float]] = None) -> str:
    """
    Format DataFrame as LaTeX table with booktabs style and multirow.
    
    Columns: Bit Depth, Channels, Blocking-$n$, DML, Compression Rate (x)
    Uses \\multirow for Bit Depth and Channels.
    Adds FLAC level 5 comparison rows when flac_rates is provided.
    
    Args:
        df: DataFrame with columns: Bit Depth, Channels, Blocking-N, DML, Compression Rate (x)
        use_val: If True, use validation metrics; if False, use test metrics
        flac_rates: Optional dict (bit_depth, channels) -> compression rate for FLAC comparison
        
    Returns:
        LaTeX table as string
    """
    flac_rates = flac_rates or {}
    df_latex = df.copy()
    
    # Format DML column with checkmarks
    df_latex['DML'] = df_latex['DML'].apply(lambda x: r'\checkmark' if x else '')
    
    # Format Blocking-n with thousands separators
    df_latex['Blocking-N'] = df_latex['Blocking-N'].apply(_format_blocking_n)
    
    # Format compression rate
    comp_col = 'Val Compression Rate (x)' if use_val else 'Test Compression Rate (x)'
    df_latex['Compression Rate'] = df_latex[comp_col].apply(
        lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A"
    )
    
    # Identify best compression in each group for bold formatting
    best_indices = _identify_best_compression_per_group(df_latex, comp_col)
    
    def escape_latex(text):
        """Escape special LaTeX characters in text."""
        if pd.isna(text):
            return ""
        text = str(text)
        text = text.replace('&', r'\&')
        text = text.replace('%', r'\%')
        text = text.replace('$', r'\$')
        text = text.replace('#', r'\#')
        text = text.replace('^', r'\^{}')
        text = text.replace('_', r'\_')
        text = text.replace('{', r'\{')
        text = text.replace('}', r'\}')
        text = text.replace('~', r'\textasciitilde{}')
        text = text.replace('\\', r'\textbackslash{}')
        return text
    
    # Build LaTeX with multirow
    latex_lines = []
    latex_lines.append("% Requires \\usepackage{amssymb} for \\checkmark")
    latex_lines.append("% Requires \\usepackage{booktabs} for table formatting")
    latex_lines.append("% Requires \\usepackage{multirow} for \\multirow")
    latex_lines.append("")
    latex_lines.append("\\begin{table}")
    latex_lines.append("    \\centering")
    latex_lines.append("    \\begin{tabular}{cc|cc|c}")
    latex_lines.append("        \\toprule")
    latex_lines.append("        \\textbf{Bit Depth} & \\textbf{Channels} & \\textbf{Blocking-$n$} & \\textbf{DML} & \\textbf{Compression Rate} (x) \\\\")
    latex_lines.append("        \\midrule")
    
    # Group by Bit Depth, then Channels (Mono first, Stereo second)
    for bits in sorted(df_latex['Bit Depth'].unique()):
        bit_group = df_latex[df_latex['Bit Depth'] == bits]
        mono_group = bit_group[bit_group['Channels'] == 'Mono']
        stereo_group = bit_group[bit_group['Channels'] == 'Stereo']
        
        # Count rows including FLAC rows for multirow span
        n_flac_mono = 1 if (int(bits), 'Mono') in flac_rates else 0
        n_flac_stereo = 1 if (int(bits), 'Stereo') in flac_rates else 0
        n_bit_rows = len(bit_group) + n_flac_mono + n_flac_stereo
        
        row_in_bit_block = 0
        for channels in ['Mono', 'Stereo']:
            chan_group = mono_group if channels == 'Mono' else stereo_group
            has_flac = (int(bits), channels) in flac_rates
            n_chan_rows = len(chan_group) + (1 if has_flac else 0)
            
            if len(chan_group) == 0 and not has_flac:
                continue
            
            for i, (idx, row) in enumerate(chan_group.iterrows()):
                blocking_n = row['Blocking-N']
                dml = row['DML']
                comp = row['Compression Rate']
                if idx in best_indices and comp != 'N/A':
                    comp = f"\\textbf{{{comp}}}"
                
                # Bit Depth: multirow only on first row of this bit-depth block
                if row_in_bit_block == 0:
                    bit_cell = f"\\multirow{{{n_bit_rows}}}{{*}}{{{int(bits)}}}"
                else:
                    bit_cell = ""
                
                # Channels: multirow only on first row of this channel subsection
                if i == 0:
                    chan_cell = f"\\multirow{{{n_chan_rows}}}{{*}}{{{channels}}}"
                else:
                    chan_cell = ""
                
                parts = [bit_cell, chan_cell, blocking_n, dml, comp]
                row_str = " & ".join(parts)
                latex_lines.append(f"        {row_str} \\\\")
                row_in_bit_block += 1
            
            # Add FLAC row at end of this channel subsection
            if has_flac:
                latex_lines.append("        \\cmidrule(lr){3-5}")
                flac_comp = f"{flac_rates[(int(bits), channels)]:.2f}"
                flac_cell = r"\multicolumn{2}{c|}{FLAC}"
                # First row of subsection (when no Sashimi rows): need bit/chan multirow
                if row_in_bit_block == 0:
                    bit_cell = f"\\multirow{{{n_bit_rows}}}{{*}}{{{int(bits)}}}"
                    chan_cell = f"\\multirow{{{n_chan_rows}}}{{*}}{{{channels}}}"
                else:
                    bit_cell = ""
                    chan_cell = ""
                parts = [bit_cell, chan_cell, flac_cell, flac_comp]
                row_str = " & ".join(parts)
                latex_lines.append(f"        {row_str} \\\\")
                row_in_bit_block += 1
            
            # Add cmidrule between Mono and Stereo
            if channels == 'Mono' and (len(stereo_group) > 0 or n_flac_stereo):
                latex_lines.append("        \\cmidrule(lr){2-5}")
        
        # Add midrule between bit depth sections
        if bits != sorted(df_latex['Bit Depth'].unique())[-1]:
            latex_lines.append("        \\midrule")
    
    latex_lines.append("        \\bottomrule")
    latex_lines.append("    \\end{tabular}")
    latex_lines.append("\\end{table}")
    
    return "\n".join(latex_lines)


def main():
    """Main script to fetch runs, process them, and generate LaTeX table."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Generate LaTeX table from WandB Sashimi runs')
    parser.add_argument('--use_val', action='store_true',
                        help='Use validation metrics instead of test metrics (default: test)')
    args = parser.parse_args()
    
    use_val = args.use_val
    
    # Initialize WandB API
    print("Connecting to WandB...")
    api = wandb.Api()
    
    # Try to get entity from wandb login, or use default
    entity = None
    try:
        # Try to get from environment first
        import os
        entity = os.environ.get('WANDB_ENTITY', None)
        
        # Try to get from wandb settings
        if entity is None:
            try:
                viewer = api.viewer()
                if viewer and hasattr(viewer, 'username'):
                    entity = viewer.username
            except:
                pass
    except Exception as e:
        print(f"Note: Could not auto-detect WandB entity: {e}")
    
    print(f"Fetching runs from project: {WANDB_PROJECT}")
    if entity:
        print(f"Entity: {entity}")
    
    # Fetch runs
    try:
        if entity:
            runs = api.runs(f"{entity}/{WANDB_PROJECT}")
        else:
            # Try with default entity (empty string means use default)
            runs = api.runs(WANDB_PROJECT)
    except Exception as e:
        print(f"Error fetching runs with entity: {e}")
        print("Trying without entity...")
        try:
            runs = api.runs(WANDB_PROJECT)
        except Exception as e2:
            print(f"Error: Could not fetch runs: {e2}")
            return
    
    print(f"Found {len(runs)} runs")
    
    # Process each run
    runs_data = []
    for i, run in enumerate(runs):
        print(f"\nProcessing run {i+1}/{len(runs)}: {run.name or run.id}")
        
        # Extract configuration
        config_params = extract_config_params(run)
        
        # Filter: only keep runs with d_model == 64
        d_model = config_params.get('d_model')
        if d_model is not None and d_model != 64:
            print(f"  Skipping: d_model={d_model} (not 64)")
            continue
        elif d_model is None:
            print(f"  Warning: d_model not found in config, assuming 64")
        
        # Filter: only keep runs with sample_len == 8192
        sample_len = config_params.get('sample_len')
        if sample_len is not None and sample_len != 8192:
            print(f"  Skipping: sample_len={sample_len} (not 8192)")
            continue
        elif sample_len is None:
            print(f"  Warning: sample_len not found in config, assuming 8192")
        
        # Extract BPB and loss metrics
        val_bpb_history = extract_bpb_history(run, 'val/bpb')
        test_bpb_history = extract_bpb_history(run, 'test/bpb')
        val_loss_history = extract_bpb_history(run, 'val/loss')
        test_loss_history = extract_bpb_history(run, 'test/loss')
        
        # Check for required metric based on use_val flag
        if use_val:
            if val_bpb_history is None:
                print(f"  Warning: No val/bpb data for run {run.id}")
                continue
        else:
            if test_bpb_history is None:
                print(f"  Warning: No test/bpb data for run {run.id}")
                continue
        
        # Apply time-weighted EMA smoothing
        val_bpb_smoothed = time_weighted_ema(val_bpb_history) if val_bpb_history is not None else np.nan
        test_bpb_smoothed = time_weighted_ema(test_bpb_history) if test_bpb_history is not None else np.nan
        val_loss_smoothed = time_weighted_ema(val_loss_history) if val_loss_history is not None else np.nan
        test_loss_smoothed = time_weighted_ema(test_loss_history) if test_loss_history is not None else np.nan
        
        # Calculate compression rates
        bit_depth = config_params['bits']
        val_compression = calculate_compression_rate(val_bpb_smoothed, bit_depth)
        test_compression = calculate_compression_rate(test_bpb_smoothed, bit_depth) if not np.isnan(test_bpb_smoothed) else np.nan
        
        # Store run data
        run_data = {
            **config_params,
            'val_loss': val_loss_smoothed,
            'test_loss': test_loss_smoothed,
            'val_bpb': val_bpb_smoothed,
            'test_bpb': test_bpb_smoothed,
            'val_compression': val_compression,
            'test_compression': test_compression,
        }
        runs_data.append(run_data)
        
        print(f"  Config: stereo={config_params['stereo']}, "
              f"interleaving={config_params['interleaving_strategy']}, "
              f"dml={config_params['dml']}, bits={config_params['bits']}")
        print(f"  Val Loss: {val_loss_smoothed:.4f}, Val BPB: {val_bpb_smoothed:.3f}, Test BPB: {test_bpb_smoothed:.3f}")
    
    print(f"\nProcessed {len(runs_data)} runs with valid data")
    
    # Deduplicate runs
    print("\nDeduplicating runs with equivalent configurations...")
    runs_data = deduplicate_runs(runs_data)
    print(f"After deduplication: {len(runs_data)} unique configurations")
    
    # Build DataFrame
    print("\nBuilding DataFrame...")
    if len(runs_data) == 0:
        print("Warning: No runs with valid data found!")
        return
    
    df_data = []
    for run_data in runs_data:
        # Format blocking-N value, inferring from sample_len if needed
        blocking_strategy = run_data.get('interleaving_strategy', '')
        blocking_n_value = '--'  # Default for mono
        if run_data.get('stereo', False) and blocking_strategy:
            strategy_str = str(blocking_strategy).lower()
            if strategy_str == 'temporal':
                blocking_n_value = '1'
            elif strategy_str.startswith('blocking'):
                if '-' in strategy_str:
                    parts = strategy_str.split('-')
                    if len(parts) > 1:
                        blocking_n_value = parts[-1]  # Use the number from blocking-N
                    else:
                        sample_len = run_data.get('sample_len', 8192)
                        blocking_n_value = str(sample_len)
                else:
                    sample_len = run_data.get('sample_len', 8192)
                    blocking_n_value = str(sample_len)
            else:
                blocking_n_value = '1'
        
        row = {
            'Bit Depth': run_data['bits'],
            'Channels': 'Stereo' if run_data['stereo'] else 'Mono',
            'Blocking-N': blocking_n_value,
            'DML': run_data['dml'],
        }
        if use_val:
            row['Val Compression Rate (x)'] = run_data['val_compression']
        else:
            row['Test Compression Rate (x)'] = run_data['test_compression']
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Sort by: bit depth, channels (Mono first), dml (categorical first), blocking-n
    df_sorted = df.copy()
    df_sorted['Channels-sort'] = df_sorted['Channels'].map({'Mono': 0, 'Stereo': 1})
    df_sorted['DML-sort'] = df_sorted['DML'].astype(int)  # False=0, True=1
    def _blocking_n_sort_key(x):
        if pd.isna(x) or x == '' or x == '--':
            return float('inf')
        s = str(x).replace(',', '')
        return float(s) if s.isdigit() else float('inf')
    df_sorted['Blocking-N-sort'] = df_sorted['Blocking-N'].apply(_blocking_n_sort_key)
    df = df_sorted.sort_values(['Bit Depth', 'Channels-sort', 'DML-sort', 'Blocking-N-sort']).drop(
        columns=['Channels-sort', 'DML-sort', 'Blocking-N-sort']
    )

    # Filter blocking-n: keep only 1, 512, and max per (bits, channels, dml)
    df = _filter_blocking_n_values(df)

    # Load FLAC compression rates for comparison
    flac_rates = _load_flac_compression_rates()
    
    # Generate and print LaTeX table
    print("\n" + "="*80)
    metric_type = "Validation" if use_val else "Test"
    print(f"LaTeX Table ({metric_type} metrics):")
    print("="*80)
    latex_table = format_latex_table(df, use_val=use_val, flac_rates=flac_rates)
    print(latex_table)
    print("="*80)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
