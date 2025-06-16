# README
# Phillip Long
# June 10, 2025

# Generate plots to compare naive-FLAC and naive-LDAC encoding methods.

# IMPORTS
##################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from typing import Any, Dict
from os import mkdir
from os.path import exists
import warnings

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))

import utils

##################################################


# HELPER FUNCTIONS
##################################################

# format a value for use in a legend
def formatter(val: Any) -> str:
    """Format a value for use in a legend."""
    if isinstance(val, int):
        return f"{val:,}"
    elif isinstance(val, float):
        return f"{val:.4f}"
    elif isinstance(val, bool):
        return "True" if val else "False"
    elif isinstance(val, str):
        return val
    else:
        return str(val)

##################################################


# FUNCTION TO PLOT STATISTICS
##################################################

def plot_compression_statistics(df: pd.DataFrame, output_dir: str):
    """
    Plot compression statistics.
    """

    # determine facet columns
    facet_columns = list(filter(lambda column: column not in utils.TEST_COMPRESSION_COLUMN_NAMES, df.columns))

    # prepare the output filepath
    output_filepath = f"{output_dir}/percentiles.pdf"

    # compute percentiles (0 to 100)
    percentiles = np.linspace(start = 0, stop = 100, num = 101)

    # set up the matplotlib figure
    fig, (ax_rate, ax_speed) = plt.subplots(nrows = 2, ncols = 1, figsize = (8, 10), sharex = True, constrained_layout = True)

    # enable seaborn style
    sns.set_theme(style = "whitegrid")

    # # plot for each facet group
    # grouped = df.groupby(by = facet_columns)
    # for facet_values, group in grouped:

    #     # compute percentile values for compression_rate and compression_speed
    #     compression_rate_percentiles = np.percentile(a = group["compression_rate"], q = percentiles)
    #     compression_speed_percentiles = np.percentile(a = group["compression_speed"], q = percentiles)

    #     # plot percentiles
    #     label = ", ".join(map(formatter, facet_values)) if isinstance(facet_values, tuple) else formatter(facet_values)
    #     ax_rate.plot(percentiles, compression_rate_percentiles, label = label)
    #     ax_speed.plot(percentiles, compression_speed_percentiles, label = label)

    # construct data frame with percentiles data
    data = pd.DataFrame(columns = ["percentile", "compression_rate", "compression_speed"] + facet_columns)
    get_facet_column = lambda facet_value: utils.rep(x = formatter(facet_value), times = len(percentiles))
    grouped = df.groupby(by = facet_columns)
    for facet_values, group in grouped:
        facet_data = {
            "percentile": percentiles,
            "compression_rate": np.percentile(a = group["compression_rate"], q = percentiles), # percentile values for compression rate
            "compression_speed": np.percentile(a = group["compression_speed"], q = percentiles), # percentile values for compression rate
        }
        facet_data.update(dict(zip(facet_columns, map(get_facet_column, facet_values))) if isinstance(facet_values, tuple) else {facet_columns[0]: get_facet_column(facet_value = facet_values)}) # repeat facet values in facet columns
        data = pd.concat((data, pd.DataFrame(data = facet_data)), axis = 0, ignore_index = True)
        del facet_data
    del get_facet_column, grouped

    # plot data
    facetting_attributes = ["hue"] if len(facet_columns) == 1 else (["hue", "style"] if len(facet_columns) == 2 else ["hue", "style", "size"])
    if len(facet_columns) > 3:
        warnings.warn(message = "Only facetting on the first 3 facet columns.", category = RuntimeWarning)
    legend_facets = dict(zip(facetting_attributes, facet_columns))
    sns.lineplot(ax = ax_rate, data = data, x = "percentile", y = "compression_rate", legend = "auto", **legend_facets)
    sns.lineplot(ax = ax_speed, data = data, x = "percentile", y = "compression_speed", legend = False, **legend_facets)

    # styling the plots
    ax_rate.set_title("Compression Rate")
    ax_rate.set_ylabel("Compression Rate")
    ax_speed.set_title("Encoding Speed")
    ax_speed.set_ylabel("Encoding Speed (seconds of computation per second of audio)")
    ax_speed.set_xlabel("Percentile")
    ax_rate.legend(fontsize = "small", title_fontsize = "small")
    # ax_speed.legend().remove() # remove legend from bottom panel
    ax_rate.grid(True)
    ax_speed.grid(True)

    # remove x-axis tick labels for top plot
    ax_rate.tick_params(labelbottom = False)

    # save the figure
    fig.savefig(output_filepath, dpi = utils.FIGURE_DPI)
    plt.close(fig)

    # return nothing
    return

##################################################


# COMPARISON PLOT
##################################################

def plot_comparison(dfs: Dict[str, pd.DataFrame], output_dir: str):
    """
    Plot comparison of naive-FLAC and naive-LDAC.
    """

    # methods
    methods = list(dfs.keys())

    # determine facet columns
    facet_columns = {method: list(filter(lambda column: column not in utils.TEST_COMPRESSION_COLUMN_NAMES, dfs[method].columns)) for method in methods}

    # prepare the output filepath
    output_filepath = f"{output_dir}/percentiles_comparison.pdf"

    # compute percentiles (0 to 100)
    percentiles = np.linspace(start = 0, stop = 100, num = 101)

    # set up the matplotlib figure
    fig, (ax_rate, ax_speed) = plt.subplots(nrows = 2, ncols = 1, figsize = (8, 10), sharex = True, constrained_layout = True)
    fig.suptitle("Comparing Lossy Estimators")

    # enable seaborn style
    sns.set_theme(style = "whitegrid")

    # construct data frame with percentiles data
    data = pd.DataFrame(columns = ["percentile", "compression_rate", "compression_speed", "lossy_estimator"])
    for lossy_estimator, df in dfs.items():
        grouped = df.groupby(by = facet_columns[lossy_estimator])
        for _, group in grouped:
            data = pd.concat((data, pd.DataFrame(data = {
                "percentile": percentiles,
                "compression_rate": np.percentile(a = group["compression_rate"], q = percentiles), # percentile values for compression rate
                "compression_speed": np.percentile(a = group["compression_speed"], q = percentiles), # percentile values for compression rate
                "lossy_estimator": utils.rep(x = lossy_estimator.upper(), times = len(percentiles))
            })), axis = 0, ignore_index = True)
        del grouped

    # get average across different facets for each lossy estimator
    averaged_data = data.groupby(by = ["percentile", "lossy_estimator"]).mean().reset_index(drop = False)

    # plot data
    dots_alpha = 0.4
    sns.scatterplot(ax = ax_rate, data = data, x = "percentile", y = "compression_rate", hue = "lossy_estimator", legend = False, alpha = dots_alpha)
    sns.lineplot(ax = ax_rate, data = averaged_data, x = "percentile", y = "compression_rate", hue = "lossy_estimator", legend = "auto")
    sns.scatterplot(ax = ax_speed, data = data, x = "percentile", y = "compression_speed", hue = "lossy_estimator", legend = False, alpha = dots_alpha)
    sns.lineplot(ax = ax_speed, data = averaged_data, x = "percentile", y = "compression_speed", hue = "lossy_estimator", legend = False)

    # styling the plots
    ax_rate.set_title("Compression Rate")
    ax_rate.set_ylabel("Compression Rate")
    ax_speed.set_title("Encoding Speed")
    ax_speed.set_ylabel("Encoding Speed (seconds of computation per second of audio)")
    ax_speed.set_xlabel("Percentile")
    ax_rate.legend(title = "Lossy Estimator", fontsize = "small", title_fontsize = "small")
    # ax_speed.legend().remove() # remove legend from bottom panel
    ax_rate.grid(True)
    ax_speed.grid(True)

    # remove x-axis tick labels for top plot
    ax_rate.tick_params(labelbottom = False)

    # save the figure
    fig.savefig(output_filepath, dpi = utils.FIGURE_DPI)
    plt.close(fig)

    # return nothing
    return

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # SETUP
    ##################################################

    # read in arguments
    def parse_args(args = None, namespace = None):
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(prog = "Plots", description = "Create Plots to Compare Naive-FLAC and Naive-LDAC Implementations") # create argument parser
        parser.add_argument("--input_dir", type = str, default = utils.EVAL_DIR, help = "Absolute filepath to the input evaluation directory.")
        args = parser.parse_args(args = args, namespace = namespace) # parse arguments
        return args # return parsed arguments
    args = parse_args()

    # infer other directories
    flac_dir = f"{args.input_dir}/flac"
    ldac_dir = f"{args.input_dir}/ldac"

    ##################################################


    # PLOT STATISTICS FOR FLAC AND LDAC
    ##################################################

    # flac
    flac_results = pd.read_csv(filepath_or_buffer = f"{flac_dir}/test.csv", sep = ",", header = 0, index_col = False)
    plot_compression_statistics(df = flac_results, output_dir = flac_dir)

    # ldac
    ldac_results = pd.read_csv(filepath_or_buffer = f"{ldac_dir}/test.csv", sep = ",", header = 0, index_col = False)
    plot_compression_statistics(df = ldac_results, output_dir = ldac_dir)

    ##################################################


    # PLOT COMPARISON PLOT
    ##################################################

    plots_dir = f"{args.input_dir}/plots"
    if not exists(plots_dir):
        mkdir(plots_dir)
    plot_comparison(dfs = {"flac": flac_results, "ldac": ldac_results}, output_dir = plots_dir)

    ##################################################

##################################################