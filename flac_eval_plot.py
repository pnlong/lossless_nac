import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
import argparse

# Configuration
PLOTS_SHARE_Y_AXIS = True
X_AXIS_LABEL = "FLAC Compression Level"
Y_AXIS_LABEL = "Compression Rate (x)"

# map dataset name to fancier names
DATASET_NAME_TO_FANCIER_NAME = {
    "musdb18mono": "MusDB18 Mono (All)",
    "musdb18mono_mixes": "MusDB18 Mono (Mixes)",
    "musdb18mono_stems": "MusDB18 Mono (Stems)",
    "musdb18stereo": "MusDB18 Stereo (All)",
    "musdb18stereo_mixes": "MusDB18 Stereo (Mixes)",
    "musdb18stereo_stems": "MusDB18 Stereo (Stems)",
    "librispeech": "LibriSpeech",
    "ljspeech": "LJSpeech",
    "birdvox": "Birdvox",
    "beethoven": "Beethoven",
    "youtube_mix": "YouTube Mix",
    "sc09": "SC09",
    "vctk": "VCTK",
    "epidemic": "Epidemic Sound",
    "torrent16b_pro": "Commercial 16-bit (Pro)",
    "torrent16b_amateur": "Commercial 16-bit (Amateur)",
    "torrent16b_freeload": "Commercial 16-bit (Freeload)",
    "torrent16b_amateur_freeload": "Commercial 16-bit",
    "torrent24b_pro": "Commercial 24-bit (Pro)",
    "torrent24b_amateur": "Commercial 24-bit (Amateur)",
    "torrent24b_freeload": "Commercial 24-bit (Freeload)",
    "torrent24b_amateur_freeload": "Commercial 24-bit",
    "torrent16b": "Commercial 16-bit (All)",
    "torrent24b": "Commercial 24-bit (All)",
}

# Canonical order for dataset legend (shared with lmic_eval_plot for consistent line colors)
DATASET_LEGEND_ORDER = sorted(DATASET_NAME_TO_FANCIER_NAME.keys())

# Domain -> list of dataset names (column count = len(DOMAINS), column width ∝ len(datasets))
DOMAINS = ["Music", "Speech", "Bioacoustics", "Sound Effects"]
domain_to_datasets = {
    "Music": [
        "beethoven", 
        "youtube_mix",
        "musdb18mono",
        # "musdb18mono_mixes", 
        # "musdb18mono_stems",
        # "musdb18stereo", 
        "musdb18stereo_mixes", 
        # "musdb18stereo_stems",
        # "torrent16b_pro", 
        # "torrent16b_amateur", 
        # "torrent16b_freeload", 
        "torrent16b_amateur_freeload",
        # "torrent24b_pro", 
        # "torrent24b_amateur", 
        # "torrent24b_freeload", 
        "torrent24b_amateur_freeload",
        # "torrent16b", 
        # "torrent24b",
    ],
    "Speech": [
        "sc09",
        "vctk",
        "librispeech",
        "ljspeech",
    ],
    "Bioacoustics": [
        "birdvox",
    ],
    "Sound Effects": [
        "epidemic",
    ],
}

# Omit these datasets at 16-bit only (they appear at 8-bit)
DATASETS_OMIT_AT_16BIT = {"youtube_mix", "sc09", "beethoven"}

# FLAC plot: column (title, list of domain keys). Bioacoustics + Sound Effects combined.
FLAC_PLOT_COLUMNS = [
    ("Music", ["Music"]),
    ("Speech", ["Speech"]),
    ("Bioacoustics / Sound Effects", ["Bioacoustics", "Sound Effects"]),
]

if __name__ == "__main__":
    # read in arguments
    def parse_args(args = None, namespace = None):
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(prog = "Plot", description = "Plot FLAC Evaluation Results") # create argument parser
        parser.add_argument("--input_filepath", type = str, default = "/home/pnlong/lnac/flac_eval_results.csv", help = "Absolute filepath to the input CSV file.")
        parser.add_argument("--disable_constant_subframes", action = "store_true", help = "Disable constant subframes for FLAC.")
        parser.add_argument("--disable_fixed_subframes", action = "store_true", help = "Disable fixed subframes for FLAC.")
        parser.add_argument("--disable_verbatim_subframes", action = "store_true", help = "Disable verbatim subframes for FLAC.")
        parser.add_argument("--output_filepath", type = str, default = "/home/pnlong/lnac/flac_eval_plot.pdf", help = "Absolute filepath to the output PDF file.")
        parser.add_argument("--skinny", action = "store_true", help = "Square figure, single-column legends, equal row heights, x-axis labels only at 0 and 8.")
        args = parser.parse_args(args = args, namespace = namespace) # parse arguments
        return args # return parsed arguments
    args = parse_args()

    # Load the CSV file
    df = pd.read_csv(args.input_filepath)

    # Filter
    df_filtered = df[df["matches_native_quantization"] == True]
    df_filtered = df_filtered[df_filtered["disable_constant_subframes"] == args.disable_constant_subframes]
    df_filtered = df_filtered[df_filtered["disable_fixed_subframes"] == args.disable_fixed_subframes]
    df_filtered = df_filtered[df_filtered["disable_verbatim_subframes"] == args.disable_verbatim_subframes]

    # One column per FLAC_PLOT_COLUMNS entry, equal widths
    n_cols = len(FLAC_PLOT_COLUMNS)
    skinny = getattr(args, "skinny", False)

    # Create figure: 3 rows (legend + 8-bit + 16-bit), n_cols columns (equal width)
    figsize = (9, 6) if skinny else (14, 6)
    height_ratios = [0.5, 1, 1] if skinny else [0.5, 2, 2]
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, n_cols, figure=fig, height_ratios=height_ratios, hspace=0.1, wspace=0.05)
    axes = [[fig.add_subplot(gs[i, j]) for j in range(n_cols)] for i in range(3)]

    # Share y-axis within each plot row (rows 1 and 2)
    if PLOTS_SHARE_Y_AXIS:
        for col_idx in range(1, n_cols):
            axes[1][col_idx].sharey(axes[1][0])
            axes[2][col_idx].sharey(axes[2][0])

    # First, create the legend row (row 0) for each column
    df_8bit = df_filtered[df_filtered["bit_depth"] == 8]
    hue_orders_by_column = []
    for col_idx, (col_title, domain_keys) in enumerate(FLAC_PLOT_COLUMNS):
        datasets = [d for key in domain_keys for d in domain_to_datasets[key]]
        ax_legend = axes[0][col_idx]
        df_group = df_8bit[df_8bit["dataset"].isin(datasets)]
        hue_order = [d for d in datasets if d in df_group["dataset"].unique()]
        hue_orders_by_column.append(hue_order)

        # Create a temporary plot just to extract the legend handles and labels
        temp_fig, temp_ax = plt.subplots(figsize=(1, 1))
        sns.lineplot(
            data=df_group,
            x="flac_compression_level",
            y="overall_compression_rate",
            hue="dataset",
            hue_order=hue_order,
            marker="o",
            ax=temp_ax
        )
        if hasattr(temp_ax, 'legend_') and temp_ax.legend_ is not None:
            legend = temp_ax.legend_
            handles = legend.legend_handles
            labels = [t.get_text() for t in legend.get_texts()]
        else:
            handles, labels = temp_ax.get_legend_handles_labels()
        plt.close(temp_fig)
        labels = [DATASET_NAME_TO_FANCIER_NAME.get(label, label) for label in labels]

        ax_legend.axis('off')
        ax_legend.set_title(col_title)
        ncol = 1 if skinny else max(2, (len(handles) + 2) // 3)
        ax_legend.legend(handles, labels, loc="center", ncol=ncol, fontsize=8)

    # Loop over bit depths (rows 1 and 2) and columns
    for row_idx, bit_depth in enumerate([8, 16]):
        df_bit = df_filtered[df_filtered["bit_depth"] == bit_depth]
        for col_idx, (col_title, domain_keys) in enumerate(FLAC_PLOT_COLUMNS):
            datasets = [d for key in domain_keys for d in domain_to_datasets[key]]
            ax = axes[row_idx + 1][col_idx]
            df_group = df_bit[df_bit["dataset"].isin(datasets)]
            if bit_depth == 16:
                df_group = df_group[~df_group["dataset"].isin(DATASETS_OMIT_AT_16BIT)]
            hue_order = hue_orders_by_column[col_idx]
            sns.lineplot(
                data=df_group,
                x="flac_compression_level",
                y="overall_compression_rate",
                hue="dataset",
                hue_order=hue_order,
                marker="o",
                ax=ax,
                legend=False
            )
            if row_idx == 0:
                ax.set_xlabel("")
                ax.set_xticklabels([])
            else:
                x_label = "Compression Level" if skinny else X_AXIS_LABEL
                ax.set_xlabel(x_label)
                if skinny:
                    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: str(int(x)) if x in (0, 2, 4, 6, 8) else ""))
            if PLOTS_SHARE_Y_AXIS:
                ax.set_ylabel(Y_AXIS_LABEL if col_idx == 0 else "")
            else:
                ax.set_ylabel(Y_AXIS_LABEL)
            if col_idx != 0:
                ax.tick_params(axis="y", labelleft=False)
            ax.set_title("")
            ax.grid(True)
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.08, top=0.94, hspace=0.1, wspace=0.05)

    row_title_x_offset = 0.09 if skinny else 0.055
    for row_idx, bit_depth in enumerate([8, 16]):
        ax_left = axes[row_idx + 1][0]
        pos = ax_left.get_position()
        fig.text(pos.x0 - row_title_x_offset, pos.y0 + pos.height / 2, f"{bit_depth}-bit",
                rotation=90, ha='center', va='center', fontsize=14)

    # Save the plot
    plt.savefig(args.output_filepath, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {args.output_filepath}.")
