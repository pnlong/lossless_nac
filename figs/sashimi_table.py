#!/usr/bin/env python3
"""
Generate LaTeX table from WandB runs comparing Sashimi model configurations.

This script fetches Sashimi model runs from WandB, extracts BPB metrics,
applies time-weighted EMA smoothing, calculates compression rates, and
generates a LaTeX table comparing different model configurations.
"""

import argparse
import json
import pandas as pd
import numpy as np
import wandb
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime


# Configuration constants
WANDB_PROJECT = "LNAC"
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


def format_latex_table(df: pd.DataFrame, use_val: bool = False) -> str:
    """
    Format DataFrame as LaTeX table with booktabs style.
    
    Args:
        df: DataFrame with columns: Stereo, Blocking-N, DML, Bit Depth,
            Loss, BPB, Compression Rate (x) (either val or test based on use_val)
        use_val: If True, use validation metrics; if False, use test metrics
        
    Returns:
        LaTeX table as string
    """
    # Create a copy to avoid modifying original
    df_latex = df.copy()
    
    # Format boolean columns with checkmarks
    if 'Stereo' in df_latex.columns:
        df_latex['Stereo'] = df_latex['Stereo'].apply(lambda x: r'\checkmark' if x else '')
    if 'DML' in df_latex.columns:
        df_latex['DML'] = df_latex['DML'].apply(lambda x: r'\checkmark' if x else '')
    
    # Format bit depth as numeric value (8 or 16)
    if 'Bit Depth' in df_latex.columns:
        df_latex['Bit Depth'] = df_latex['Bit Depth'].apply(lambda x: f"{int(x)}" if not pd.isna(x) else "N/A")
    
    # Blocking-N is already formatted in the DataFrame (handled during DataFrame building)
    # No additional formatting needed here
    
    # Format Loss column (4 decimals) - only one column based on use_val, before BPB
    loss_col = 'Val Loss' if use_val else 'Test Loss'
    if loss_col in df_latex.columns:
        df_latex[loss_col] = df_latex[loss_col].apply(
            lambda x: f"{x:.4f}" if not pd.isna(x) else "N/A"
        )
        df_latex = df_latex.rename(columns={loss_col: 'Loss'})
    
    # Format BPB column (2-3 decimals) - only one column based on use_val
    bpb_col = 'Val BPB' if use_val else 'Test BPB'
    if bpb_col in df_latex.columns:
        df_latex[bpb_col] = df_latex[bpb_col].apply(
            lambda x: f"{x:.3f}" if not pd.isna(x) else "N/A"
        )
        # Rename to generic "BPB"
        df_latex = df_latex.rename(columns={bpb_col: 'BPB'})
    
    # Format compression rate column (2 decimals, numeric only) - only one column based on use_val
    comp_col = 'Val Compression Rate (x)' if use_val else 'Test Compression Rate (x)'
    if comp_col in df_latex.columns:
        df_latex[comp_col] = df_latex[comp_col].apply(
            lambda x: f"{x:.2f}" if not pd.isna(x) else "N/A"
        )
        # Rename to generic "Compression Rate (x)"
        df_latex = df_latex.rename(columns={comp_col: 'Compression Rate (x)'})
    
    # Helper function to escape LaTeX special characters
    def escape_latex(text):
        """Escape special LaTeX characters in text."""
        if pd.isna(text):
            return ""
        text = str(text)
        # Escape special characters
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
    
    # Generate LaTeX table
    latex_lines = []
    latex_lines.append("% Requires \\usepackage{amssymb} for \\checkmark")
    latex_lines.append("% Requires \\usepackage{booktabs} for table formatting")
    latex_lines.append("")
    latex_lines.append("\\begin{table}")
    latex_lines.append("\\centering")
    # Use better column alignment: l for text, c for checkmarks, r for numbers
    col_align = []
    for col in df_latex.columns:
        if col in ['Stereo', 'DML']:
            col_align.append('c')  # Center for checkmarks
        elif col in ['Loss', 'BPB', 'Compression Rate (x)', 'Bit Depth']:
            col_align.append('r')  # Right-align for numbers
        else:
            col_align.append('c')  # Center for Blocking-N (numbers)
    latex_lines.append("\\begin{tabular}{" + "".join(col_align) + "}")
    latex_lines.append("\\toprule")
    
    # Header row - escape special LaTeX characters
    headers = [escape_latex(h) for h in df_latex.columns.tolist()]
    latex_lines.append(" & ".join(headers) + " \\\\")
    latex_lines.append("\\midrule")
    
    # Data rows - escape LaTeX special characters in data
    # Special values that should not be escaped: checkmark, N/A
    special_values = {r'\checkmark', 'N/A'}
    prev_bit_depth = None
    for idx, (_, row) in enumerate(df_latex.iterrows()):
        # Check if bit depth changed (and not the first row)
        current_bit_depth = row.get('Bit Depth', '')
        if prev_bit_depth is not None and current_bit_depth != prev_bit_depth:
            latex_lines.append("\\midrule")
        prev_bit_depth = current_bit_depth
        
        row_values = []
        for val in row.values:
            val_str = str(val)
            if val_str in special_values:
                row_values.append(val_str)
            else:
                row_values.append(escape_latex(val))
        row_str = " & ".join(row_values)
        latex_lines.append(row_str + " \\\\")
    
    latex_lines.append("\\bottomrule")
    latex_lines.append("\\end{tabular}")
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
        blocking_n_value = ''
        if blocking_strategy:
            strategy_str = str(blocking_strategy).lower()
            if strategy_str == 'temporal':
                blocking_n_value = '1'
            elif strategy_str.startswith('blocking'):
                if '-' in strategy_str:
                    parts = strategy_str.split('-')
                    if len(parts) > 1:
                        blocking_n_value = parts[-1]  # Use the number from blocking-N
                    else:
                        # Fallback: infer from sample_len
                        sample_len = run_data.get('sample_len', 8192)
                        blocking_n_value = str(sample_len)
                else:
                    # Just "blocking" - infer from sample_len
                    sample_len = run_data.get('sample_len', 8192)
                    blocking_n_value = str(sample_len)
            else:
                blocking_n_value = '1'
        
        row = {
            'Stereo': run_data['stereo'],
            'Blocking-N': blocking_n_value,
            'DML': run_data['dml'],
            'Bit Depth': run_data['bits'],
        }
        # Add only the selected metric (val or test), with loss before BPB
        if use_val:
            row['Val Loss'] = run_data['val_loss']
            row['Val BPB'] = run_data['val_bpb']
            row['Val Compression Rate (x)'] = run_data['val_compression']
        else:
            row['Test Loss'] = run_data['test_loss']
            row['Test BPB'] = run_data['test_bpb']
            row['Test Compression Rate (x)'] = run_data['test_compression']
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Sort by: bit depth (outermost), then stereo, then dml, then blocking-n
    # Convert Blocking-N to numeric for proper sorting (empty strings will be sorted last)
    df_sorted = df.copy()
    df_sorted['Blocking-N-sort'] = df_sorted['Blocking-N'].apply(
        lambda x: float('inf') if pd.isna(x) or x == '' else float(x) if str(x).isdigit() else float('inf')
    )
    df = df_sorted.sort_values(['Bit Depth', 'Stereo', 'DML', 'Blocking-N-sort']).drop(columns=['Blocking-N-sort'])
    
    # Generate and print LaTeX table
    print("\n" + "="*80)
    metric_type = "Validation" if use_val else "Test"
    print(f"LaTeX Table ({metric_type} metrics):")
    print("="*80)
    latex_table = format_latex_table(df, use_val=use_val)
    print(latex_table)
    print("="*80)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
