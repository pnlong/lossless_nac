import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

# read in arguments
def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog = "Plot", description = "Plot FLAC Evaluation Results") # create argument parser
    parser.add_argument("--input_filepath", type = str, default = "/home/pnlong/lnac/flac_eval_results.csv", help = "Absolute filepath to the input CSV file.")
    parser.add_argument("--disable_constant_subframes", action = "store_true", help = "Disable constant subframes for FLAC.")
    parser.add_argument("--disable_fixed_subframes", action = "store_true", help = "Disable fixed subframes for FLAC.")
    parser.add_argument("--disable_verbatim_subframes", action = "store_true", help = "Disable verbatim subframes for FLAC.")
    parser.add_argument("--output_filepath", type = str, default = "/home/pnlong/lnac/flac_eval_plot.pdf", help = "Absolute filepath to the output PDF file.")
    args = parser.parse_args(args = args, namespace = namespace) # parse arguments
    return args # return parsed arguments
args = parse_args()

# Configuration
PLOTS_SHARE_Y_AXIS = True
X_AXIS_LABEL = "FLAC Compression Level"
Y_AXIS_LABEL = "Compression Rate (x)"

# Load the CSV file
df = pd.read_csv(args.input_filepath)

# Filter
df_filtered = df[(df["is_native_bit_depth"] == True) & (df["matches_native_quantization"] == True)]
df_filtered = df_filtered[df_filtered["disable_constant_subframes"] == args.disable_constant_subframes]
df_filtered = df_filtered[df_filtered["disable_fixed_subframes"] == args.disable_fixed_subframes]
df_filtered = df_filtered[df_filtered["disable_verbatim_subframes"] == args.disable_verbatim_subframes]

# Split data into three groups
musdb18_mask = df_filtered["dataset"].str.startswith("musdb18")
torrent_mask = df_filtered["dataset"].str.startswith("torrent")
df_musdb18 = df_filtered[musdb18_mask]
df_torrent = df_filtered[torrent_mask]
df_other = df_filtered[~musdb18_mask & ~torrent_mask]

# Create figure with three subplots (horizontal)
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=PLOTS_SHARE_Y_AXIS)

# First subplot: MUSDB18
ax1 = axes[0]
sns.lineplot(
    data=df_musdb18,
    x="flac_compression_level",
    y="overall_compression_rate",
    hue="dataset",
    marker="o",
    ax=ax1
)
ax1.set_xlabel(X_AXIS_LABEL)
ax1.set_ylabel(Y_AXIS_LABEL)
ax1.set_title("FLAC on MUSDB18")
ax1.grid(True)
ax1.legend(title="Dataset", loc="upper right")

# Second subplot: Torrent
ax2 = axes[1]
sns.lineplot(
    data=df_torrent,
    x="flac_compression_level",
    y="overall_compression_rate",
    hue="dataset",
    marker="o",
    ax=ax2
)
ax2.set_xlabel(X_AXIS_LABEL)
if not PLOTS_SHARE_Y_AXIS:
    ax2.set_ylabel(Y_AXIS_LABEL)
ax2.set_title("FLAC on Torrented Data")
ax2.grid(True)
ax2.legend(title="Dataset", loc="upper left")

# Third subplot: Everything else
ax3 = axes[2]
sns.lineplot(
    data=df_other,
    x="flac_compression_level",
    y="overall_compression_rate",
    hue="dataset",
    marker="o",
    ax=ax3
)
ax3.set_xlabel(X_AXIS_LABEL)
if not PLOTS_SHARE_Y_AXIS:
    ax3.set_ylabel(Y_AXIS_LABEL)
ax3.set_title("FLAC on More")
ax3.grid(True)
ax3.legend(title="Dataset", loc="upper left")

# Overall title
fig.suptitle("Comparing FLAC Compression Levels", fontsize=16, y=1.02)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig(args.output_filepath, dpi=300, bbox_inches="tight")
print(f"Saved plot to {args.output_filepath}.")
