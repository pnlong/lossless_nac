import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

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
df = df[df["is_native_bit_depth"]] # native bit depth must match the bit depth used for the dataset
df = df[df["matches_native_quantization"]] # native quantization must match the quantization used for the dataset

# Define the desired order for compressors (llama-2-13b before llama-2-7b, so llama-2-7b is to the right)
compressor_order = sorted(df["compressor"].unique(), key = lambda x: 0 if x == "llama-2-7b" else 1)  # Alphabetically: llama-2-13b, llama-2-7b
df["compressor"] = pd.Categorical(df["compressor"], categories=compressor_order, ordered=True)
df = df.sort_values(by = "compressor") # sort by compressor so Llama-models appear in order
df = df.reset_index(drop = True)

# Split data into three groups
musdb18_mask = df["dataset"].str.startswith("musdb18")
torrent_mask = df["dataset"].str.startswith("torrent")
df_musdb18 = df[musdb18_mask]
df_torrent = df[torrent_mask]
df_other = df[~musdb18_mask & ~torrent_mask]

# Create figure with three subplots (horizontal)
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=PLOTS_SHARE_Y_AXIS)

# First subplot: MUSDB18
ax1 = axes[0]
sns.lineplot(
    data=df_musdb18,
    x="compressor",
    y="compression_rate",
    hue="dataset",
    marker="o",
    ax=ax1
)
ax1.set_xlabel(X_AXIS_LABEL)
ax1.set_ylabel(Y_AXIS_LABEL)
ax1.set_title("LMIC on MUSDB18")
ax1.grid(True)
ax1.legend(title="Dataset", loc="upper right")

# Second subplot: Torrent
ax2 = axes[1]
sns.lineplot(
    data=df_torrent,
    x="compressor",
    y="compression_rate",
    hue="dataset",
    marker="o",
    ax=ax2
)
ax2.set_xlabel(X_AXIS_LABEL)
if not PLOTS_SHARE_Y_AXIS:
    ax2.set_ylabel(Y_AXIS_LABEL)
ax2.set_title("LMIC on Torrented Data")
ax2.grid(True)
ax2.legend(title="Dataset", loc="upper left")

# Third subplot: Everything else
ax3 = axes[2]
sns.lineplot(
    data=df_other,
    x="compressor",
    y="compression_rate",
    hue="dataset",
    marker="o",
    ax=ax3
)
ax3.set_xlabel(X_AXIS_LABEL)
if not PLOTS_SHARE_Y_AXIS:
    ax3.set_ylabel(Y_AXIS_LABEL)
ax3.set_title("LMIC on More")
ax3.grid(True)
ax3.legend(title="Dataset", loc="upper left")

# Overall title
fig.suptitle("Comparing LMIC Compressors", fontsize=16, y=1.02)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig(args.output_filepath, dpi=300, bbox_inches="tight")
print(f"Saved plot to {args.output_filepath}.")
