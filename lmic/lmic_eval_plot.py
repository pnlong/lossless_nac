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
df = df[df["matches_native_quantization"]] # native quantization must match the quantization used for the dataset

# Define the desired order for compressors (llama-2-13b before llama-2-7b, so llama-2-7b is to the right)
compressor_order = sorted(df["compressor"].unique(), key = lambda x: 0 if x == "llama-2-7b" else 1)

# Define dataset groups configuration
dataset_groups = [
    {
        "mask_func": lambda df: df["dataset"].str.startswith("musdb18"),
        "title": "LMIC on MUSDB18",
        "legend_loc": "upper right"
    },
    {
        "mask_func": lambda df: df["dataset"].str.startswith("torrent"),
        "title": "LMIC on Torrented Data",
        "legend_loc": "upper left"
    },
    {
        "mask_func": lambda df: ~df["dataset"].str.startswith("musdb18") & ~df["dataset"].str.startswith("torrent"),
        "title": "LMIC on More",
        "legend_loc": "upper left"
    }
]

# Create figure with six subplots (2 rows, 3 columns)
fig, axes = plt.subplots(2, 3, figsize=(18, 12), sharey=PLOTS_SHARE_Y_AXIS)

# Loop over bit depths (rows) and dataset groups (columns)
for row_idx, bit_depth in enumerate([8, 16]):
    df_bit = df[df["bit_depth"] == bit_depth].copy()
    df_bit["compressor"] = pd.Categorical(df_bit["compressor"], categories=compressor_order, ordered=True)
    df_bit = df_bit.sort_values(by="compressor").reset_index(drop=True)
    
    for col_idx, group_config in enumerate(dataset_groups):
        ax = axes[row_idx, col_idx]
        df_group = df_bit[group_config["mask_func"](df_bit)]
        
        sns.lineplot(
            data=df_group,
            x="compressor",
            y="compression_rate",
            hue="dataset",
            marker="o",
            ax=ax
        )
        ax.set_xlabel(X_AXIS_LABEL)
        if col_idx == 0 or not PLOTS_SHARE_Y_AXIS:
            ax.set_ylabel(Y_AXIS_LABEL)
        ax.set_title(f"{group_config['title']} ({bit_depth}-bit)")
        ax.grid(True)
        ax.legend(title="Dataset", loc=group_config["legend_loc"])

# Overall title
fig.suptitle("Comparing LMIC Compressors", fontsize=16, y=1.02)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig(args.output_filepath, dpi=300, bbox_inches="tight")
print(f"Saved plot to {args.output_filepath}.")
