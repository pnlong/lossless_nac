# README
# Phillip Long
# June 10, 2025

# Generate plots to compare different lossless compressors.

# IMPORTS
##################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from typing import Any, Dict, List
from os import mkdir, listdir
from os.path import exists, dirname, realpath
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

def plot_compression_statistics_percentiles(df: pd.DataFrame, facet_columns: List[str], output_dir: str):
    """
    Plot compression statistics with a percentile plot.
    """

    # prepare the output filepath
    output_filepath = f"{output_dir}/percentiles.pdf"

    # compute percentiles (0 to 100)
    percentiles = np.linspace(start = 0, stop = 100, num = 101)

    # set up the matplotlib figure
    fig, (ax_rate, ax_speed) = plt.subplots(nrows = 2, ncols = 1, figsize = (8, 10), sharex = True, constrained_layout = True)

    # enable seaborn style
    sns.set_theme(style = "whitegrid")

    # plot for each facet group
    grouped = df.groupby(by = facet_columns)
    for facet_values, group in grouped:

        # compute percentile values for compression_rate and compression_speed
        compression_rate_percentiles = np.percentile(a = group["compression_rate"], q = percentiles)
        compression_speed_percentiles = np.percentile(a = group["compression_speed"], q = percentiles)

        # plot percentiles
        label = ", ".join(map(formatter, facet_values)) if isinstance(facet_values, tuple) else formatter(facet_values)
        ax_rate.plot(percentiles, compression_rate_percentiles, label = label)
        ax_speed.plot(percentiles, compression_speed_percentiles, label = label)

    # # construct data frame with percentiles data
    # data = pd.DataFrame(columns = ["percentile", "compression_rate", "compression_speed"] + facet_columns)
    # get_facet_column = lambda facet_value: utils.rep(x = formatter(facet_value), times = len(percentiles))
    # grouped = df.groupby(by = facet_columns)
    # for facet_values, group in grouped:
    #     facet_data = {
    #         "percentile": percentiles,
    #         "compression_rate": np.percentile(a = group["compression_rate"], q = percentiles), # percentile values for compression rate
    #         "compression_speed": np.percentile(a = group["compression_speed"], q = percentiles), # percentile values for compression rate
    #     }
    #     facet_data.update(dict(zip(facet_columns, map(get_facet_column, facet_values))) if isinstance(facet_values, tuple) else {facet_columns[0]: get_facet_column(facet_value = facet_values)}) # repeat facet values in facet columns
    #     data = pd.concat((data, pd.DataFrame(data = facet_data)), axis = 0, ignore_index = True)
    #     del facet_data
    # del get_facet_column, grouped

    # # plot data
    # facetting_attributes = ["hue"] if len(facet_columns) == 1 else (["hue", "style"] if len(facet_columns) == 2 else ["hue", "style", "size"])
    # if len(facet_columns) > 3:
    #     warnings.warn(message = "Only facetting on the first 3 facet columns.", category = RuntimeWarning)
    # legend_facets = dict(zip(facetting_attributes, facet_columns))
    # sns.lineplot(ax = ax_rate, data = data, x = "percentile", y = "compression_rate", legend = "auto", **legend_facets)
    # sns.lineplot(ax = ax_speed, data = data, x = "percentile", y = "compression_speed", legend = False, **legend_facets)

    # styling the plots
    ax_rate.set_title("Compression Rate")
    ax_rate.set_ylabel("Compression Rate (%)")
    ax_speed.set_title("Encoding Speed")
    ax_speed.set_ylabel("Encoding Speed (seconds of audio encoded per second)")
    ax_speed.set_xlabel("Percentile")
    ax_rate.legend(title = ", ".join(map(lambda column: column.replace("_", " ").title(), facet_columns)), fontsize = "x-small", title_fontsize = "x-small")
    # ax_speed.legend().remove() # remove legend from bottom panel

    # add grid
    ax_rate.grid(True, alpha = 0.3)
    ax_speed.grid(True, alpha = 0.3)

    # remove x-axis tick labels for top plot
    ax_rate.tick_params(labelbottom = False)

    # save the figure
    fig.savefig(output_filepath, dpi = utils.FIGURE_DPI)
    plt.close(fig)

    # return nothing
    return


def plot_compression_statistics_boxplots(df: pd.DataFrame, facet_columns: List[str], output_dir: str):
    """
    Plot compression statistics with box plots.
    """

    # prepare the output filepath
    output_filepath = f"{output_dir}/boxplots.pdf"

    # set up the matplotlib figure
    fig, (ax_rate, ax_speed) = plt.subplots(nrows = 2, ncols = 1, figsize = (8, 10), sharex = True, constrained_layout = True)

    # enable seaborn style
    sns.set_theme(style = "whitegrid")

    # prepare data for plotting, melting the dataframe to long format
    facet_column_name = ", ".join(map(lambda column: column.replace("_", " ").title(), facet_columns))
    df_plot = df.copy()
    if len(facet_columns) > 1: # create a combined facet label if multiple facet columns exist
        df_plot[facet_column_name] = df_plot[facet_columns].apply(lambda row: ", ".join(map(formatter, row.values)), axis = 1)
    else:
        df_plot[facet_column_name] = df_plot[facet_columns[0]].apply(formatter)

    # create box plots
    sns.boxplot(ax = ax_rate, data = df_plot, x = facet_column_name, y = "compression_rate")
    sns.boxplot(ax = ax_speed, data = df_plot, x = facet_column_name, y = "compression_speed")

    # styling the plots
    ax_rate.set_title("Compression Rate")
    ax_rate.set_ylabel("Compression Rate (%)")
    ax_rate.set_xlabel("") # hide x label
    ax_speed.set_title("Encoding Speed")
    ax_speed.set_ylabel("Encoding Speed (seconds of audio encoded per second)")
    ax_speed.set_xlabel(f"Configuration: {facet_column_name}")

    # rotate x-axis labels if there are many categories or long labels
    if len(df_plot[facet_column_name].unique()) > 3 or any(len(str(label)) > 10 for label in df_plot[facet_column_name].unique()):
        ax_rate.tick_params(axis = "x", rotation = 45)
        ax_speed.tick_params(axis = "x", rotation = 45)

    # remove x-axis tick labels for top plot
    ax_rate.tick_params(labelbottom = False)

    # add grid
    ax_rate.grid(True, alpha = utils.GRID_ALPHA)
    ax_speed.grid(True, alpha = utils.GRID_ALPHA)

    # save the figure
    fig.savefig(output_filepath, dpi = utils.FIGURE_DPI)
    plt.close(fig)

    # return nothing
    return

##################################################


# COMPARISON PLOT
##################################################

def plot_comparison_percentiles(dfs: Dict[str, pd.DataFrame], facet_columns: Dict[str, List[str]], output_dir: str):
    """
    Plot comparison of different lossless compressors with a percentile plot.
    """

    # prepare the output filepath
    output_filepath = f"{output_dir}/percentiles_comparison.pdf"

    # compute percentiles (0 to 100)
    percentiles = np.linspace(start = 0, stop = 100, num = 101)

    # set up the matplotlib figure
    fig, (ax_rate, ax_speed) = plt.subplots(nrows = 2, ncols = 1, figsize = (8, 10), sharex = True, constrained_layout = True)
    fig.suptitle("Comparing Lossless Compressors")

    # enable seaborn style
    sns.set_theme(style = "whitegrid")

    # construct data frame with percentiles data
    data = pd.DataFrame(columns = ["percentile", "compression_rate", "compression_speed", "lossless_compressor"])
    for lossless_compressor, df in dfs.items():
        grouped = df.groupby(by = facet_columns[lossless_compressor])
        for _, group in grouped:
            data = pd.concat((data, pd.DataFrame(data = {
                "percentile": percentiles,
                "compression_rate": np.percentile(a = group["compression_rate"], q = percentiles), # percentile values for compression rate
                "compression_speed": np.percentile(a = group["compression_speed"], q = percentiles), # percentile values for compression rate
                "lossless_compressor": utils.rep(x = lossless_compressor.upper(), times = len(percentiles)),
            })), axis = 0, ignore_index = True)
        del grouped

    # get average across different facets for each lossless compressor
    averaged_data = data.groupby(by = ["percentile", "lossless_compressor"]).mean().reset_index(drop = False)

    # plot data
    dots_alpha = 0.4
    sns.scatterplot(ax = ax_rate, data = data, x = "percentile", y = "compression_rate", hue = "lossless_compressor", legend = False, alpha = dots_alpha)
    sns.lineplot(ax = ax_rate, data = averaged_data, x = "percentile", y = "compression_rate", hue = "lossless_compressor", legend = "auto")
    sns.scatterplot(ax = ax_speed, data = data, x = "percentile", y = "compression_speed", hue = "lossless_compressor", legend = False, alpha = dots_alpha)
    sns.lineplot(ax = ax_speed, data = averaged_data, x = "percentile", y = "compression_speed", hue = "lossless_compressor", legend = False)

    # styling the plots
    ax_rate.set_title("Compression Rate")
    ax_rate.set_ylabel("Compression Rate (%)")
    ax_speed.set_title("Encoding Speed")
    ax_speed.set_ylabel("Encoding Speed (seconds of audio encoded per second)")
    ax_speed.set_xlabel("Percentile")
    ax_rate.legend(title = "Lossless Compressor", fontsize = "small", title_fontsize = "small")
    # ax_speed.legend().remove() # remove legend from bottom panel

    # add grid
    ax_rate.grid(True, alpha = utils.GRID_ALPHA)
    ax_speed.grid(True, alpha = utils.GRID_ALPHA)

    # remove x-axis tick labels for top plot
    ax_rate.tick_params(labelbottom = False)

    # save the figure
    fig.savefig(output_filepath, dpi = utils.FIGURE_DPI)
    plt.close(fig)

    # return nothing
    return


def plot_comparison_boxplots(dfs: Dict[str, pd.DataFrame], facet_columns: Dict[str, List[str]], output_dir: str):
    """
    Plot comparison of different lossless compressors with a box plot, where the box plot is for the best configuration according to best compression rate.
    """

    # get lossless compressor
    lossless_compressors = list(dfs.keys())

    # prepare the output filepath
    output_filepath = f"{output_dir}/boxplots_comparison.pdf"

    # set up the matplotlib figure
    fig, (ax_rate, ax_speed) = plt.subplots(nrows = 2, ncols = 1, figsize = (8, 10), sharex = True, constrained_layout = True)
    fig.suptitle("Comparing Lossless Compressors")

    # enable seaborn style
    sns.set_theme(style = "whitegrid")

    # construct data frame
    data_columns = ["lossless_compressor", "compression_rate", "compression_speed", "parameters"]
    data = pd.DataFrame(columns = data_columns)
    for lossless_compressor, df in dfs.items():
        current_facet_columns = facet_columns[lossless_compressor]
        optimal_configuration = df[current_facet_columns + ["compression_rate"]].groupby(by = current_facet_columns).mean().reset_index(drop = False) # get mean compression rate for each configuration
        optimal_configuration = optimal_configuration.loc[optimal_configuration["compression_rate"].argmax(axis = 0)] # select configuration with the greatest compression rate
        optimal_configuration = {facet_column: optimal_configuration[facet_column] for facet_column in current_facet_columns}
        df_optimal_configuration = df[np.all(a = np.array([df[facet_column] == optimal_configuration[facet_column] for facet_column in current_facet_columns]), axis = 0)].copy()
        df_optimal_configuration["parameters"] = utils.rep(x = str(optimal_configuration), times = len(df_optimal_configuration))
        df_optimal_configuration["lossless_compressor"] = utils.rep(x = lossless_compressor.upper(), times = len(df_optimal_configuration))
        df_optimal_configuration = df_optimal_configuration[data_columns]
        data = pd.concat((data, df_optimal_configuration), axis = 0, ignore_index = True)
        del optimal_configuration, df_optimal_configuration # free up memory

    # logging
    print("BEST CONFIGURATION PER LOSSLESS COMPRESSOR:")
    summary = data.copy()
    longest_lossless_compressor_string_length = max(map(len, lossless_compressors))
    summary["lossless_compressor"] = list(map(lambda i: data.at[i, "lossless_compressor"] + ":" + (" " * (1 + (longest_lossless_compressor_string_length - len(data.at[i, "lossless_compressor"])))) + data.at[i, "parameters"], data.index))
    max_descriptor_length = max(map(len, summary["lossless_compressor"]))
    summary["lossless_compressor"] = list(map(lambda descriptor: descriptor + (" " * (max_descriptor_length - len(descriptor))), summary["lossless_compressor"])) # end pad so that they are left aligned
    summary = summary.drop(columns = "parameters") # because parameters is baked into the lossless_compressor column
    summary = summary.groupby(by = "lossless_compressor").mean().reset_index(drop = False)
    print(utils.pretty_dataframe_string(df = summary, max_colwidth = 150, border_style = "simple"))
    del summary, longest_lossless_compressor_string_length, max_descriptor_length # free up memory

    # plot data
    sns.boxplot(ax = ax_rate, data = data, x = "lossless_compressor", y = "compression_rate")
    sns.boxplot(ax = ax_speed, data = data, x = "lossless_compressor", y = "compression_speed")

    # styling the plots
    ax_rate.set_title("Compression Rate")
    ax_rate.set_ylabel("Compression Rate (%)")
    ax_rate.set_xlabel("") # hide x label
    ax_speed.set_title("Encoding Speed")
    ax_speed.set_ylabel("Encoding Speed (seconds of audio encoded per second)")
    ax_speed.set_xlabel("Lossless Compressor")

    # add grid
    ax_rate.grid(True, alpha = utils.GRID_ALPHA)
    ax_speed.grid(True, alpha = utils.GRID_ALPHA)

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
        parser = argparse.ArgumentParser(prog = "Plots", description = "Create Plots to Compare Lossless Compressors") # create argument parser
        parser.add_argument("--input_dir", type = str, default = utils.EVAL_DIR, help = "Absolute filepath to the input evaluation directory.")
        args = parser.parse_args(args = args, namespace = namespace) # parse arguments
        return args # return parsed arguments
    args = parse_args()

    # get lossless compressors
    lossless_compressors = [test_path.split(".")[0] for test_path in listdir(f"{dirname(realpath(__file__))}/lossless_compressors")]

    # infer other directories
    lossless_compressor_dirs = [f"{args.input_dir}/{lossless_compressor}" for lossless_compressor in lossless_compressors]

    ##################################################


    # PLOT STATISTICS
    ##################################################

    # plot statistics for different lossless compressors
    print(utils.MAJOR_SEPARATOR_LINE)
    dfs = dict()
    facet_columns = dict()
    for i, lossless_compressor in enumerate(lossless_compressors):

        # read in data frame
        directory = f"{args.input_dir}/{lossless_compressor}"
        df_filepath = f"{directory}/test.csv"
        if not exists(df_filepath): # skip if doesn't exist yet
            continue
        dfs[lossless_compressor] = pd.read_csv(filepath_or_buffer = df_filepath, sep = ",", header = 0, index_col = False)
        facet_columns[lossless_compressor] = list(filter(lambda column: column not in utils.TEST_COMPRESSION_COLUMN_NAMES, dfs[lossless_compressor].columns))

        # pretty print the data frame
        summary = dfs[lossless_compressor].groupby(by = facet_columns[lossless_compressor])[["compression_rate", "compression_speed"]].mean()
        summary = summary.reset_index(drop = False) # make it so that group by columns are columns themselves
        print(f"{lossless_compressor.upper()}:")
        print(utils.pretty_df(df = summary))
        print(utils.MINOR_SEPARATOR_LINE if i < len(lossless_compressors) - 1 else utils.MAJOR_SEPARATOR_LINE)
        del summary # free up memory

        # plot
        plot_compression_statistics_percentiles(df = dfs[lossless_compressor], facet_columns = facet_columns[lossless_compressor], output_dir = directory)
        plot_compression_statistics_boxplots(df = dfs[lossless_compressor], facet_columns = facet_columns[lossless_compressor], output_dir = directory)

    ##################################################


    # PLOT COMPARISON PLOT
    ##################################################

    # create plots directory
    plots_dir = f"{args.input_dir}/plots"
    if not exists(plots_dir):
        mkdir(plots_dir)

    # plot comparison plots
    plot_comparison_percentiles(dfs = dfs, facet_columns = facet_columns, output_dir = plots_dir)
    plot_comparison_boxplots(dfs = dfs, facet_columns = facet_columns, output_dir = plots_dir)
    print(utils.MAJOR_SEPARATOR_LINE)

    ##################################################

##################################################