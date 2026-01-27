import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
import argparse
import sys
from os.path import dirname, realpath

# Import dataset name mapping from flac_eval_plot.py
sys.path.insert(0, dirname(dirname(realpath(__file__))))
from flac_eval_plot import DATASET_NAME_TO_FANCIER_NAME

# read in arguments
def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Plot", description = "Plot LMIC Evaluation Results") # create argument parser
    parser.add_argument("--input_filepath", type = str, default = "/home/pnlong/lnac/lmic/lmic_eval_results.csv", help = "Absolute filepath to the input CSV file.")
    parser.add_argument("--output_filepath", type = str, default = "/home/pnlong/lnac/lmic/lmic_eval_plot.pdf", help = "Absolute filepath to the output PDF file.")
    args = parser.parse_args(args = args, namespace = namespace) # parse arguments
    return args # return parsed arguments
args = parse_args()

# Configuration
PLOTS_SHARE_Y_AXIS = True
X_AXIS_LABEL = "Compressor"
Y_AXIS_LABEL = "Compression Rate (x)"

# Load the CSV file
df = pd.read_csv(args.input_filepath)

# filter df
df = df[df["matches_native_quantization"]] # native quantization must match the quantization used for the dataset

# Define the desired order for compressors (llama-2-13b before llama-2-7b, so llama-2-7b is to the right)
compressor_order = sorted(df["compressor"].unique(), key = lambda x: 0 if x == "llama-2-7b" else 1)

# Define dataset groups configuration
dataset_groups = [
    {
        "mask_func": lambda df: df["dataset"].str.startswith("musdb18"),
        "title": "MusDB18",
        "legend_loc": "upper right"
    },
    {
        "mask_func": lambda df: df["dataset"].str.startswith("torrent"),
        "title": "Torrented Data",
        "legend_loc": "upper left"
    },
    {
        "mask_func": lambda df: ~df["dataset"].str.startswith("musdb18") & ~df["dataset"].str.startswith("torrent"),
        "title": "More",
        "legend_loc": "upper left"
    }
]

# Create figure with nine subplots (3 rows, 3 columns) with height ratios 1:2:2
fig = plt.figure(figsize=(14, 7))
gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1, 2, 2], hspace=0.3, wspace=0.3)
axes = [[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(3)]

# Share y-axis within each plot row (rows 1 and 2)
if PLOTS_SHARE_Y_AXIS:
    # Share y-axis across columns in row 1 (8-bit)
    for col_idx in range(1, 3):
        axes[1][col_idx].sharey(axes[1][0])
    # Share y-axis across columns in row 2 (16-bit)
    for col_idx in range(1, 3):
        axes[2][col_idx].sharey(axes[2][0])

# First, create the legend row (row 0) for each column
for col_idx, group_config in enumerate(dataset_groups):
    ax_legend = axes[0][col_idx]
    
    # Get data from one of the bit depths to create the legend
    df_bit = df[df["bit_depth"] == 8].copy()
    df_bit["compressor"] = pd.Categorical(df_bit["compressor"], categories=compressor_order, ordered=True)
    df_bit = df_bit.sort_values(by="compressor").reset_index(drop=True)
    df_group = df_bit[group_config["mask_func"](df_bit)]
    
    # Create a temporary plot just to extract the legend handles and labels
    temp_fig, temp_ax = plt.subplots(figsize=(1, 1))
    sns.lineplot(
        data=df_group,
        x="compressor",
        y="compression_rate",
        hue="dataset",
        marker="o",
        ax=temp_ax
    )
    
    # Extract handles and labels from the temporary plot's legend
    # seaborn stores the legend in temp_ax.legend_
    if hasattr(temp_ax, 'legend_') and temp_ax.legend_ is not None:
        legend = temp_ax.legend_
        handles = legend.legend_handles
        labels = [t.get_text() for t in legend.get_texts()]
    else:
        # Fallback: get handles and labels from the axes
        handles, labels = temp_ax.get_legend_handles_labels()
    plt.close(temp_fig)
    
    # Map dataset names to fancier names
    labels = [DATASET_NAME_TO_FANCIER_NAME.get(label, label) for label in labels]
    
    # Hide the axes and show only the legend, centered
    ax_legend.axis('off')
    ax_legend.set_title(group_config['title'])
    # Calculate number of columns for wrapping (aim for 2-3 items per column)
    n_items = len(handles)
    ncol = 2 # max(2, (n_items + 2) // 3)  # Aim for roughly 3 items per column, minimum 2 columns
    ax_legend.legend(
        handles,
        labels,
        # title="Dataset",
        loc="center",
        ncol=ncol, fontsize=8, title_fontsize=9)

# Loop over bit depths (rows 1 and 2) and dataset groups (columns)
for row_idx, bit_depth in enumerate([8, 16]):
    df_bit = df[df["bit_depth"] == bit_depth].copy()
    df_bit["compressor"] = pd.Categorical(df_bit["compressor"], categories=compressor_order, ordered=True)
    df_bit = df_bit.sort_values(by="compressor").reset_index(drop=True)
    
    for col_idx, group_config in enumerate(dataset_groups):
        ax = axes[row_idx + 1][col_idx]  # +1 because row 0 is for legends
        df_group = df_bit[group_config["mask_func"](df_bit)]
        
        sns.lineplot(
            data=df_group,
            x="compressor",
            y="compression_rate",
            hue="dataset",
            marker="o",
            ax=ax,
            legend=False
        )
        
        # Set x-axis label only on the bottom row (16-bit), explicitly remove for 8-bit
        if row_idx == 0:  # 8-bit row
            ax.set_xlabel("")  # Explicitly remove x-axis label
        elif row_idx == 1:  # 16-bit row
            ax.set_xlabel(X_AXIS_LABEL)
        
        # Set y-axis label only on first column when sharing y-axis
        if PLOTS_SHARE_Y_AXIS:
            if col_idx == 0:
                ax.set_ylabel(Y_AXIS_LABEL)
            else:
                ax.set_ylabel("")  # Hide y-axis label on non-first columns
        else:
            ax.set_ylabel(Y_AXIS_LABEL)
        
        # Remove individual titles - we'll use row titles instead
        ax.set_title("")
        ax.grid(True)

# Overall title
# fig.suptitle("Comparing LMIC Compressors", fontsize=16, y=1.02)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Add vertical row titles on the left side (after tight_layout so positions are final)
# Get the position of the leftmost subplot in each row to place text
for row_idx, bit_depth in enumerate([8, 16]):
    ax_left = axes[row_idx + 1][0]  # Leftmost subplot in the row
    # Get the position in figure coordinates
    pos = ax_left.get_position()
    # Place text to the left of the subplot, vertically centered
    fig.text(pos.x0 - 0.05, pos.y0 + pos.height / 2, f"{bit_depth}-bit", 
             rotation=90, ha='center', va='center', fontsize=12)

# Save the plot
plt.savefig(args.output_filepath, dpi=300, bbox_inches="tight")
print(f"Saved plot to {args.output_filepath}.")
