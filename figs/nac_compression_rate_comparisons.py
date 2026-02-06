# README
# Phillip Long
# Compare compression rates across different compressors using their best ablation configurations.

# IMPORTS
##################################################

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import exists
from os import makedirs
from typing import Dict, List, Tuple

##################################################


# CONSTANTS
##################################################

BASE_DIR = "/deepfreeze/pnlong/lnac/eval"
COMPRESSORS = [
    "flac",
    # "iflac",
    "ldac",
    "lec",
    # "lnac",
]
OUTPUT_DIR = f"{BASE_DIR}/plots"

# Standard columns that are not ablation columns
STANDARD_COLUMNS = ["path", "size_original", "size_compressed", "compression_rate", 
                    "duration_audio", "duration_encoding", "compression_speed"]

FIGURE_DPI = 200
GRID_ALPHA = 0.3

FANCIER_COMPRESSOR_NAMES = {
    "flac": "FLAC",
    "ldac": "DAC",
    "lec": "EnCodec",
    "lnac": "Custom DAC",
}

##################################################


# HELPER FUNCTIONS
##################################################

def get_ablation_columns(df: pd.DataFrame) -> List[str]:
    """Get columns that come after compression_speed (ablation columns)."""
    if "compression_speed" not in df.columns:
        return []
    
    compression_speed_idx = df.columns.get_loc("compression_speed")
    ablation_cols = df.columns[compression_speed_idx + 1:].tolist()
    return ablation_cols


def find_best_ablation(df: pd.DataFrame, ablation_columns: List[str]) -> Tuple[pd.DataFrame, Dict]:
    """
    Find the best ablation configuration (highest mean compression_rate).
    Returns filtered dataframe and the best ablation configuration dict.
    """
    if len(ablation_columns) == 0:
        # No ablation columns, return all rows
        return df.copy(), {}
    
    # Group by ablation columns and calculate mean compression_rate
    grouped = df.groupby(by=ablation_columns)["compression_rate"].mean()
    
    # Find the best ablation (highest mean compression_rate)
    best_ablation_idx = grouped.idxmax()
    
    # Convert to dict if it's a tuple (multiple ablation columns) or single value
    if isinstance(best_ablation_idx, tuple):
        best_ablation = dict(zip(ablation_columns, best_ablation_idx))
    else:
        best_ablation = {ablation_columns[0]: best_ablation_idx}
    
    # Filter dataframe to only include rows with best ablation
    # Build mask by checking all ablation column values match
    mask = pd.Series([True] * len(df), index=df.index)
    for col, val in best_ablation.items():
        mask = mask & (df[col] == val)
    
    filtered_df = df[mask].copy()
    
    return filtered_df, best_ablation


def print_compression_speed_stats(compressor_stats: Dict[str, float], file_counts: Dict[str, int], title: str = ""):
    """Print compression speed statistics and file counts in a nicely formatted way."""
    print("\n" + "=" * 60)
    if title:
        print(f"Average Compression Speed (x Real-Time) and File Counts - {title}")
    else:
        print("Average Compression Speed (x Real-Time) and File Counts")
    print("=" * 60)
    for compressor in compressor_stats.keys():
        avg_speed = compressor_stats[compressor]
        n_files = file_counts[compressor]
        print(f"{compressor.upper():>10}: {avg_speed:>8.4f}x  (n = {n_files:>5})")
    print("=" * 60 + "\n")


def create_boxplot(df: pd.DataFrame, output_filepath: str):
    """Create a boxplot comparing compression rates across compressors."""
    # Map compressor names to fancier display names
    df_plot = df.copy()
    df_plot["compressor"] = df_plot["compressor"].map(
        lambda x: FANCIER_COMPRESSOR_NAMES.get(x, x.upper())
    )
    
    # Get unique compressors in order
    unique_compressors = sorted(df_plot["compressor"].unique())
    
    # Define color palette for compressors
    # Using a distinct color palette - one color per compressor
    color_palette = sns.color_palette("Set2", n_colors=len(unique_compressors))
    compressor_colors = dict(zip(unique_compressors, color_palette))
    
    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
    
    # Enable seaborn style
    sns.set_theme(style="whitegrid")
    
    # Create boxplot with custom colors using dictionary palette
    sns.boxplot(ax=ax, data=df_plot, x="compressor", y="compression_rate", 
                palette=compressor_colors)
    
    # Styling
    ax.set_xlabel("Compressor")
    ax.set_ylabel("Compression Rate (x)")
    ax.grid(True, alpha=GRID_ALPHA)
    
    # Rotate x-axis labels if needed
    ax.tick_params(axis="x", rotation=45 if len(df_plot["compressor"].unique()) > 5 else 0)
    
    # Save the figure
    fig.savefig(output_filepath, dpi=FIGURE_DPI, format="pdf")
    plt.close(fig)
    
    return

##################################################


# MAIN SCRIPT
##################################################

if __name__ == "__main__":
    
    # Create output directory if it doesn't exist
    if not exists(OUTPUT_DIR):
        makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load data for each compressor and find best ablation
    print("Loading data and finding best ablation configurations...")
    
    filtered_dfs = {}
    file_counts = {}
    
    for compressor in COMPRESSORS:
        csv_path = f"{BASE_DIR}/{compressor}/test.csv"
        
        if not exists(csv_path):
            print(f"Warning: {csv_path} not found, skipping {compressor}")
            continue
        
        # Load CSV
        df = pd.read_csv(csv_path, sep=",", header=0, index_col=False)
        
        # Get ablation columns
        ablation_columns = get_ablation_columns(df)
        
        # Find best ablation
        filtered_df, best_ablation = find_best_ablation(df, ablation_columns)
        
        # Add compressor column
        filtered_df["compressor"] = compressor
        
        # Store filtered dataframe
        filtered_dfs[compressor] = filtered_df
        
        # Count number of unique files
        n_files = filtered_df["path"].nunique()
        file_counts[compressor] = n_files
        
        # Print info about best ablation
        if ablation_columns:
            print(f"{compressor.upper()}: Best ablation = {best_ablation} (n = {n_files} files)")
        else:
            print(f"{compressor.upper()}: No ablation columns (using all data) (n = {n_files} files)")
    
    # Load lnac data and add it to the main dataframes
    lnac_csv_path = f"{BASE_DIR}/lnac/test.csv"
    if exists(lnac_csv_path):
        print("Loading lnac data...")
        lnac_df_raw = pd.read_csv(lnac_csv_path, sep=",", header=0, index_col=False)
        ablation_columns = get_ablation_columns(lnac_df_raw)
        lnac_df, best_ablation = find_best_ablation(lnac_df_raw, ablation_columns)
        lnac_df["compressor"] = "lnac"
        filtered_dfs["lnac"] = lnac_df
        n_files_lnac = lnac_df["path"].nunique()
        file_counts["lnac"] = n_files_lnac
        if ablation_columns:
            print(f"LNAC: Best ablation = {best_ablation} (n = {n_files_lnac} files)")
        else:
            print(f"LNAC: No ablation columns (using all data) (n = {n_files_lnac} files)")
    else:
        print(f"Warning: {lnac_csv_path} not found, lnac will not be included in plots")
    
    # Combine all dataframes (including lnac if available)
    combined_df = pd.concat(filtered_dfs.values(), ignore_index=True)
    
    # Calculate compression speed stats for all files
    compressor_stats_all = {}
    file_counts_all = {}
    for compressor in list(COMPRESSORS) + ["lnac"]:
        if compressor in filtered_dfs:
            compressor_df = combined_df[combined_df["compressor"] == compressor]
            compressor_stats_all[compressor] = compressor_df["compression_speed"].mean()
            file_counts_all[compressor] = compressor_df["path"].nunique()
    
    # Create boxplot for all files
    print("Creating boxplot for all files...")
    # Print file counts for all files plot
    print("File counts for 'All Files' plot:")
    for compressor in list(COMPRESSORS) + ["lnac"]:
        if compressor in filtered_dfs:
            n_files = combined_df[combined_df["compressor"] == compressor]["path"].nunique()
            print(f"  {compressor.upper()}: n = {n_files}")
    all_files_output = f"{OUTPUT_DIR}/compression_rate_comparison_all_files.pdf"
    create_boxplot(combined_df, all_files_output)
    print(f"Saved: {all_files_output}\n")
    
    # Filter for mixes only (path ends with ".0.npy")
    mixes_df = combined_df[combined_df["path"].str.endswith(".0.npy")].copy()
    
    # Calculate compression speed stats for mixes only
    compressor_stats_mixes = {}
    file_counts_mixes = {}
    for compressor in list(COMPRESSORS) + ["lnac"]:
        if compressor in filtered_dfs:
            compressor_df = mixes_df[mixes_df["compressor"] == compressor]
            if len(compressor_df) > 0:
                compressor_stats_mixes[compressor] = compressor_df["compression_speed"].mean()
                file_counts_mixes[compressor] = compressor_df["path"].nunique()
    
    # Create boxplot for mixes only
    print("Creating boxplot for mixes only...")
    # Print file counts for mixes only plot
    print("File counts for 'Mixes Only' plot:")
    for compressor in list(COMPRESSORS) + ["lnac"]:
        if compressor in filtered_dfs:
            n_files = mixes_df[mixes_df["compressor"] == compressor]["path"].nunique()
            print(f"  {compressor.upper()}: n = {n_files}")
    mixes_output = f"{OUTPUT_DIR}/compression_rate_comparison_mixes_only.pdf"
    create_boxplot(mixes_df, mixes_output)
    print(f"Saved: {mixes_output}")
    
    # Print compression speed statistics at the end
    print_compression_speed_stats(compressor_stats_all, file_counts_all, "All Files")
    print_compression_speed_stats(compressor_stats_mixes, file_counts_mixes, "Mixes Only")
    
    print("\nDone!")
