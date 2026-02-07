import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FixedLocator, FormatStrFormatter
import argparse
import sys
from os.path import dirname, join, realpath

# Import dataset name mapping from flac_eval_plot.py
sys.path.insert(0, dirname(dirname(realpath(__file__))))
from flac_eval_plot import DATASET_NAME_TO_FANCIER_NAME, DATASET_LEGEND_ORDER

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
X_AXIS_LABEL = "Dataset"
Y_AXIS_LABEL = "Compression Rate (x)"
X_TICK_ROTATION = 30

# Compressor display names for legend
COMPRESSOR_DISPLAY_NAMES = {
    "llama-2-7b": "Llama-2-7B",
    "llama-2-13b": "Llama-2-13B",
    "FLAC": "FLAC",
}

# Load LMIC CSV
df = pd.read_csv(args.input_filepath)
df = df[df["matches_native_quantization"]]

# Load FLAC results (compression level 5, disabled subframes)
flac_path = join(dirname(realpath(__file__)), "..", "flac_eval_results.csv")
df_flac = pd.read_csv(flac_path)
df_flac = df_flac[
    (df_flac["matches_native_quantization"])
    & (df_flac["flac_compression_level"] == 5)
    & (df_flac["disable_constant_subframes"] == True)
    & (df_flac["disable_fixed_subframes"] == True)
    & (df_flac["disable_verbatim_subframes"] == True)
][["dataset", "bit_depth", "mean_compression_rate"]].copy()
df_flac = df_flac.rename(columns={"mean_compression_rate": "compression_rate"})
df_flac["compressor"] = "FLAC"

# Combine LMIC and FLAC
df = pd.concat([
    df[["dataset", "bit_depth", "compression_rate", "compressor"]],
    df_flac
], ignore_index=True)

# Deduplicate (e.g. LMIC has duplicate rows for torrent24b_freeload 16-bit)
df = df.drop_duplicates(subset=["dataset", "bit_depth", "compressor"], keep="first")

# Define compressor order (llama-2-7b, llama-2-13b, FLAC)
compressor_order = ["llama-2-7b", "llama-2-13b", "FLAC"]
compressor_order = [c for c in compressor_order if c in df["compressor"].unique()]

# 8-bit-only datasets (exclude from 16-bit row)
DATASETS_8BIT_ONLY = {"sc09", "youtube_mix", "beethoven"}

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

# Create figure with 6 bar chart subplots (2 rows, 3 columns)
fig = plt.figure(figsize=(14, 6))
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.08, wspace=0.12,
                      left=0.12, right=0.96, top=0.92, bottom=0.14)
axes = [[fig.add_subplot(gs[i, j]) for j in range(3)] for i in range(2)]

# Share y-axis across columns within each row (8-bit row shares y, 16-bit row shares y)
if PLOTS_SHARE_Y_AXIS:
    for row_idx in range(2):
        for col_idx in range(1, 3):
            axes[row_idx][col_idx].sharey(axes[row_idx][0])

# Precompute dataset order per column from 8-bit data (for consistent x-axis across rows)
df_8bit = df[df["bit_depth"] == 8]
dataset_orders = [
    [d for d in DATASET_LEGEND_ORDER if d in df_8bit[gc["mask_func"](df_8bit)]["dataset"].unique()]
    for gc in dataset_groups
]

# Loop over bit depths (rows 0 and 1) and dataset groups (columns)
for row_idx, bit_depth in enumerate([8, 16]):
    df_bit = df[df["bit_depth"] == bit_depth].copy()
    df_bit["compressor"] = pd.Categorical(df_bit["compressor"], categories=compressor_order, ordered=True)

    for col_idx, group_config in enumerate(dataset_groups):
        ax = axes[row_idx][col_idx]
        df_group = df_bit[group_config["mask_func"](df_bit)].copy()
        if row_idx == 1:  # 16-bit row: hide bars for 8-bit-only datasets (keep ticks)
            df_group = df_group[~df_group["dataset"].isin(DATASETS_8BIT_ONLY)]
        dataset_order = dataset_orders[col_idx]  # Same order as 8-bit row

        sns.barplot(
            data=df_group,
            x="dataset",
            y="compression_rate",
            hue="compressor",
            hue_order=compressor_order,
            order=dataset_order,
            ax=ax,
            legend=False
        )

        # Set x-tick labels to fancier names with rotation (FixedLocator avoids warning)
        n_datasets = len(dataset_order)
        ax.xaxis.set_major_locator(FixedLocator(range(n_datasets)))
        if row_idx == 0:
            ax.set_xticklabels([])  # No tick labels on 8-bit row
        else:
            ax.set_xticklabels(
                [DATASET_NAME_TO_FANCIER_NAME.get(d, d) for d in dataset_order],
                rotation=X_TICK_ROTATION, ha="right"
            )

        # Y-axis: one decimal place for compression rate
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

        # Set x-axis label only on bottom row (16-bit)
        if row_idx == 0:
            ax.set_xlabel("")
        else:
            ax.set_xlabel(X_AXIS_LABEL)

        # Set y-axis label only on first column
        if col_idx == 0:
            ax.set_ylabel(Y_AXIS_LABEL)
        else:
            ax.set_ylabel("")

        # Column titles on top row only
        if row_idx == 0:
            ax.set_title(group_config["title"])
        else:
            ax.set_title("")

        ax.grid(True, axis="y")

# Create temporary barplot to extract compressor legend handles
df_legend = df[(df["bit_depth"] == 8) & (df["dataset"] == "musdb18mono")]
temp_fig, temp_ax = plt.subplots(figsize=(1, 1))
sns.barplot(
    data=df_legend,
    x="dataset",
    y="compression_rate",
    hue="compressor",
    hue_order=compressor_order,
    ax=temp_ax
)
if hasattr(temp_ax, "legend_") and temp_ax.legend_ is not None:
    legend = temp_ax.legend_
    handles = legend.legend_handles
    labels = [t.get_text() for t in legend.get_texts()]
else:
    handles, labels = temp_ax.get_legend_handles_labels()
plt.close(temp_fig)

labels = [COMPRESSOR_DISPLAY_NAMES.get(label, label) for label in labels]

fig.legend(handles, labels, title="Compressor", loc="upper center",
          bbox_to_anchor=(0.5, 1.05), ncol=3)

# Add vertical row titles on the left side
for row_idx, bit_depth in enumerate([8, 16]):
    ax_left = axes[row_idx][0]
    pos = ax_left.get_position()
    fig.text(pos.x0 - 0.05, pos.y0 + pos.height / 2, f"{bit_depth}-bit",
             rotation=90, ha="center", va="center", fontsize=14)

# Save the plot
plt.savefig(args.output_filepath, dpi=300, bbox_inches="tight")
print(f"Saved plot to {args.output_filepath}.")
