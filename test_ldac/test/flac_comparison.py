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
from os import mkdir, get_terminal_size
from os.path import exists
import warnings

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(dirname(dirname(realpath(__file__)))))

from preprocess_musdb18 import get_mixes_only_mask, get_test_only_mask

##################################################


# CONSTANTS
##################################################

FLAC_CSV = "/deepfreeze/pnlong/lnac/eval/flac/test.csv"
LDAC_CSV = "/deepfreeze/pnlong/lnac/eval/ldac_new/test.csv"
OUTPUT_DIR = "/deepfreeze/pnlong/lnac/eval/ldac_new/flac_comparison"

MAJOR_SEPARATOR_LINE = "=" * get_terminal_size().columns
MINOR_SEPARATOR_LINE = "-" * get_terminal_size().columns
FIGURE_DPI = 200
GRID_ALPHA = 0.3

##################################################


# HELPER FUNCTIONS
##################################################

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

def rep(x: Any, times: int) -> List[Any]:
    """Repeat a value a given number of times."""
    return [x] * times

def pretty_dataframe_string(
        df: pd.DataFrame,
        max_rows: int = None,
        max_cols: int = None, 
        max_colwidth: int = 50,
        float_format: str = "{:.3f}",
        border_style: str = "grid",
    ) -> str:
    """
    Return a pretty string representation of a pandas DataFrame for command line output.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to format
    max_rows : int, optional
        Maximum number of rows to display. If None, uses pandas default.
    max_cols : int, optional
        Maximum number of columns to display. If None, uses pandas default.
    max_colwidth : int, default 50
        Maximum width of each column in characters
    float_format : str, default "{:.3f}"
        Format string for floating point numbers
    border_style : str, default "grid"
        Style of borders. Options: "grid", "simple", "plain", "minimal"
        
    Returns:
    --------
    str
        Formatted string representation of the DataFrame
    """
    
    # store original pandas options
    original_options = {
        "display.max_rows": pd.get_option("display.max_rows"),
        "display.max_columns": pd.get_option("display.max_columns"),
        "display.max_colwidth": pd.get_option("display.max_colwidth"),
        "display.width": pd.get_option("display.width"),
        "display.float_format": pd.get_option("display.float_format"),
    }
    
    try:
        # set temporary display options
        if max_rows is not None:
            pd.set_option("display.max_rows", max_rows)
        if max_cols is not None:
            pd.set_option("display.max_columns", max_cols)
        
        pd.set_option("display.max_colwidth", max_colwidth)
        pd.set_option("display.width", None) # auto-detect terminal width
        
        # set float format if DataFrame contains numeric data
        if df.select_dtypes(include = [np.number]).shape[1] > 0:
            pd.set_option("display.float_format", lambda x: float_format.format(x) if pd.notnull(x) else str(x))
        
        # get the basic string representation
        df_string = str(df)
        
        # apply border styling
        if border_style == "grid": # default pandas style with grid lines
            pass # df_string is already in grid format
            
        # simple borders with just horizontal lines
        elif border_style == "simple":
            lines = df_string.split("\n")
            if len(lines) > 2: # add horizontal line after header
                header_line = lines[0] # usually the column names line
                separator = "-" * len(header_line)
                lines.insert(1, separator)
                df_string = "\n".join(lines)
        
        # plain style removes all border characters
        elif border_style == "plain":
            lines = df_string.split("\n")
            cleaned_lines = []
            for line in lines:
                cleaned_line = line.strip() # remove leading/trailing whitespace and border characters
                if cleaned_line and not all(c in " |-+" for c in cleaned_line):
                    cleaned_lines.append(cleaned_line)
            df_string = "\n".join(cleaned_lines)
            
        # just column headers and data, no borders
        elif border_style == "minimal":
            lines = df_string.split("\n")
            data_lines = [line for line in lines if line.strip() and not all(c in " |-+" for c in line.strip())] # keep only lines that contain actual data (not just separators)
            df_string = "\n".join(data_lines)
        
        # return string
        return df_string
        
    finally: # restore original pandas options
        for option, value in original_options.items():
            pd.set_option(option, value)

def pretty_df(df: pd.DataFrame) -> str:
    """
    Simple version of pretty data frame printing that returns a nicely formatted DataFrame string.
    """
    return pretty_dataframe_string(df = df, max_rows = 20, max_cols = 10, max_colwidth = 30, border_style = "simple")

##################################################


# FUNCTION TO PLOT STATISTICS
##################################################

def plot_compression_statistics_percentiles(df: pd.DataFrame, facet_columns: List[str], output_filepath: str):
    """
    Plot compression statistics with a percentile plot.
    """

    # compute percentiles (0 to 100)
    percentiles = np.linspace(start = 0, stop = 100, num = 101)

    # set up the matplotlib figure
    fig, (ax_rate, ax_speed) = plt.subplots(nrows = 2, ncols = 1, figsize = (8, 10), sharex = True, constrained_layout = True)

    # enable seaborn style
    sns.set_theme(style = "whitegrid")

    # handle case with no facet columns
    if len(facet_columns) == 0:

        # simple case: no grouping, just compute percentiles for entire dataset
        compression_rate_percentiles = np.percentile(a = df["compression_rate"], q = percentiles)
        compression_speed_percentiles = np.percentile(a = df["compression_speed"], q = percentiles)
        
        # plot percentiles
        ax_rate.plot(percentiles, compression_rate_percentiles, label = "All Data")
        ax_speed.plot(percentiles, compression_speed_percentiles, label = "All Data")
        
        # styling
        ax_rate.legend(fontsize = "small", title_fontsize = "small")

    else:

        # construct data frame with percentiles data
        data = pd.DataFrame(columns = ["percentile", "compression_rate", "compression_speed"] + facet_columns)
        get_facet_column = lambda facet_value: rep(x = formatter(facet_value), times = len(percentiles))
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
    ax_rate.set_ylabel("Compression Rate (%)")
    ax_speed.set_title("Encoding Speed")
    ax_speed.set_ylabel("Encoding Speed (audio seconds compressed per second)")
    ax_speed.set_xlabel("Percentile")
    ax_rate.legend(fontsize = "small", title_fontsize = "small")
    # ax_speed.legend().remove() # remove legend from bottom panel

    # add grid
    ax_rate.grid(True, alpha = 0.3)
    ax_speed.grid(True, alpha = 0.3)

    # remove x-axis tick labels for top plot
    ax_rate.tick_params(labelbottom = False)

    # save the figure
    fig.savefig(output_filepath, dpi = FIGURE_DPI)
    plt.close(fig)

    return


def plot_compression_statistics_boxplots(df: pd.DataFrame, facet_columns: List[str], output_filepath: str):
    """
    Plot compression statistics with box plots.
    """

    # set up the matplotlib figure
    fig, (ax_rate, ax_speed) = plt.subplots(nrows = 2, ncols = 1, figsize = (8, 10), sharex = True, constrained_layout = True)

    # enable seaborn style
    sns.set_theme(style = "whitegrid")

    # handle case with no facet columns
    if len(facet_columns) == 0:

        # simple case: no grouping, just create simple box plots
        df_plot = df.copy()
        df_plot["category"] = "All Data"  # create a single category for plotting
        
        # create box plots
        sns.boxplot(ax = ax_rate, data = df_plot, x = "category", y = "compression_rate")
        sns.boxplot(ax = ax_speed, data = df_plot, x = "category", y = "compression_speed")
        
        # styling
        ax_rate.set_xlabel("")
        ax_speed.set_xlabel("Data")

    else:

        # prepare data for plotting, melting the dataframe to long format
        df_plot = df.copy()
        if len(facet_columns) > 1: # create a combined facet label if multiple facet columns exist
            import json
            facet_column_name = "Parameters"
            df_plot[facet_column_name] = list(map(lambda i: ",".join([formatter(df_plot.at[i, facet_column]) for facet_column in facet_columns]), df_plot.index))
            rotation_angle = 45
        else:
            facet_column_name = " ".join(facet_columns[0].split("_")).title()
            df_plot[facet_column_name] = df_plot[facet_columns[0]].apply(formatter)
            rotation_angle = 45 if (len(df_plot[facet_column_name].unique()) > 3 or any(len(str(label)) > 10 for label in df_plot[facet_column_name].unique())) else 0 # rotate x-axis labels if there are many categories or long labels

        # create box plots
        sns.boxplot(ax = ax_rate, data = df_plot, x = facet_column_name, y = "compression_rate")
        sns.boxplot(ax = ax_speed, data = df_plot, x = facet_column_name, y = "compression_speed")
        
        # styling the plots
        ax_rate.set_xlabel("") # hide x label
        ax_speed.set_xlabel(f"Configuration: {facet_column_name}")

        # apply rotation
        if rotation_angle > 0:
            ax_rate.tick_params(axis = "x", rotation = rotation_angle, labelsize = 8)
            ax_speed.tick_params(axis = "x", rotation = rotation_angle, labelsize = 8)

    # common styling
    ax_rate.set_title("Compression Rate")
    ax_rate.set_ylabel("Compression Rate (%)")
    ax_speed.set_title("Encoding Speed")
    ax_speed.set_ylabel("Encoding Speed (audio seconds compressed per second)")

    # remove x-axis tick labels for top plot
    ax_rate.tick_params(labelbottom = False)

    # add grid
    ax_rate.grid(True, alpha = GRID_ALPHA)
    ax_speed.grid(True, alpha = GRID_ALPHA)

    # save the figure
    fig.savefig(output_filepath, dpi = FIGURE_DPI)
    plt.close(fig)

    return

##################################################


# COMPARISON PLOT
##################################################

def plot_comparison_percentiles(dfs: Dict[str, pd.DataFrame], facet_columns: Dict[str, List[str]], output_filepath: str):
    """
    Plot comparison of different lossless compressors with a percentile plot.
    """

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
        current_facet_columns = facet_columns[lossless_compressor]
        if len(current_facet_columns) == 0:
            data = pd.concat((data, pd.DataFrame(data = {
                "percentile": percentiles,
                "compression_rate": np.percentile(a = df["compression_rate"], q = percentiles), # percentile values for compression rate
                "compression_speed": np.percentile(a = df["compression_speed"], q = percentiles), # percentile values for compression rate
                "lossless_compressor": rep(x = lossless_compressor.upper(), times = len(percentiles)),
            })), axis = 0, ignore_index = True)
        else:
            grouped = df.groupby(by = current_facet_columns)
            for _, group in grouped:
                data = pd.concat((data, pd.DataFrame(data = {
                    "percentile": percentiles,
                    "compression_rate": np.percentile(a = group["compression_rate"], q = percentiles), # percentile values for compression rate
                    "compression_speed": np.percentile(a = group["compression_speed"], q = percentiles), # percentile values for compression rate
                    "lossless_compressor": rep(x = lossless_compressor.upper(), times = len(percentiles)),
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
    ax_speed.set_ylabel("Encoding Speed (audio seconds compressed per second)")
    ax_speed.set_xlabel("Percentile")
    ax_rate.legend(title = "Lossless Compressor", fontsize = "small", title_fontsize = "small")
    # ax_speed.legend().remove() # remove legend from bottom panel

    # add grid
    ax_rate.grid(True, alpha = GRID_ALPHA)
    ax_speed.grid(True, alpha = GRID_ALPHA)

    # remove x-axis tick labels for top plot
    ax_rate.tick_params(labelbottom = False)

    # save the figure
    fig.savefig(output_filepath, dpi = FIGURE_DPI)
    plt.close(fig)

    return


def plot_comparison_boxplots(dfs: Dict[str, pd.DataFrame], facet_columns: Dict[str, List[str]], output_filepath: str):
    """
    Plot comparison of different lossless compressors with a box plot, where the box plot is for the best configuration according to best compression rate.
    """

    # get lossless compressor
    lossless_compressors = list(dfs.keys())

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
        if len(current_facet_columns) == 0: # no facet columns, use entire dataframe
            df_optimal_configuration = df.copy()
            df_optimal_configuration["parameters"] = rep(x = "No Parameters", times = len(df_optimal_configuration))
            df_optimal_configuration["lossless_compressor"] = rep(x = lossless_compressor.upper(), times = len(df_optimal_configuration))
            df_optimal_configuration = df_optimal_configuration[data_columns]
            data = pd.concat((data, df_optimal_configuration), axis = 0, ignore_index = True)
            del df_optimal_configuration # free up memory
        else: # facet columns, get optimal configuration
            optimal_configuration = df[current_facet_columns + ["compression_rate"]].groupby(by = current_facet_columns).mean().reset_index(drop = False) # get mean compression rate for each configuration
            optimal_configuration = optimal_configuration.loc[optimal_configuration["compression_rate"].argmax(axis = 0)] # select configuration with the greatest compression rate
            optimal_configuration = {facet_column: optimal_configuration[facet_column] for facet_column in current_facet_columns}
            df_optimal_configuration = df[np.all(a = np.array([df[facet_column] == optimal_configuration[facet_column] for facet_column in current_facet_columns]), axis = 0)].copy()
            if len(current_facet_columns) > 1: # format parameters as JSON for multiple columns
                import json
                json_safe_config = {k: int(v) if hasattr(v, 'dtype') else v for k, v in optimal_configuration.items()}
                df_optimal_configuration["parameters"] = rep(x = json.dumps(json_safe_config, separators = (",", ":")), times = len(df_optimal_configuration))
            else:
                df_optimal_configuration["parameters"] = rep(x = str(optimal_configuration), times = len(df_optimal_configuration))
            df_optimal_configuration["lossless_compressor"] = rep(x = lossless_compressor.upper(), times = len(df_optimal_configuration))
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
    print(pretty_dataframe_string(df = summary, max_colwidth = 100, border_style = "simple"))
    del summary, longest_lossless_compressor_string_length, max_descriptor_length # free up memory

    # plot data
    sns.boxplot(ax = ax_rate, data = data, x = "lossless_compressor", y = "compression_rate")
    sns.boxplot(ax = ax_speed, data = data, x = "lossless_compressor", y = "compression_speed")

    # styling the plots
    ax_rate.set_title("Compression Rate")
    ax_rate.set_ylabel("Compression Rate (%)")
    ax_rate.set_xlabel("") # hide x label
    ax_speed.set_title("Encoding Speed")
    ax_speed.set_ylabel("Encoding Speed (audio seconds compressed per second)")
    ax_speed.set_xlabel("Lossless Compressor")

    # rotate x-axis labels if any contain JSON (indicating multiple parameters)
    has_json_labels = any('{' in str(label) for label in data["lossless_compressor"].unique())
    if has_json_labels:
        ax_rate.tick_params(axis = "x", rotation = 45, labelsize = 8)
        ax_speed.tick_params(axis = "x", rotation = 45, labelsize = 8)

    # add grid
    ax_rate.grid(True, alpha = GRID_ALPHA)
    ax_speed.grid(True, alpha = GRID_ALPHA)

    # remove x-axis tick labels for top plot
    ax_rate.tick_params(labelbottom = False)

    # save the figure
    fig.savefig(output_filepath, dpi = FIGURE_DPI)
    plt.close(fig)

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
        parser = argparse.ArgumentParser(prog = "FLAC vs LDAC Comparison", description = "Compare FLAC and LDAC Lossless Compressors") # create argument parser
        parser.add_argument("--flac_csv", type = str, default = FLAC_CSV, help = "Path to FLAC test results CSV file.")
        parser.add_argument("--ldac_csv", type = str, default = LDAC_CSV, help = "Path to LDAC test results CSV file.")
        parser.add_argument("--output_dir", type = str, default = OUTPUT_DIR, help = "Output directory for comparison plots.")
        parser.add_argument("--mixes_only", action = "store_true", help = "Only use mixes in MUSDB18, not all stems.")
        args = parser.parse_args(args = args, namespace = namespace) # parse arguments
        if not exists(args.flac_csv):
            raise FileNotFoundError(f"FLAC CSV file not found: {args.flac_csv}")
        elif not exists(args.ldac_csv):
            raise FileNotFoundError(f"LDAC CSV file not found: {args.ldac_csv}")
        return args # return parsed arguments
    args = parse_args()

    # create output directory if it doesn't exist
    if not exists(args.output_dir):
        mkdir(args.output_dir)

    ##################################################


    # LOAD DATA
    ##################################################
    
    # load the CSV files
    dfs = {}
    facet_columns = {}
    
    # standard columns that are not facet columns
    standard_columns = ["path", "size_original", "size_compressed", "compression_rate", "duration_audio", "duration_encoding", "compression_speed"]

    # filter data frame according to arguments
    def filter_df(df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter the data frame according to the arguments.
        """
        df = df[get_test_only_mask(paths = df["path"])]
        df = df[get_mixes_only_mask(paths = df["path"])] if args.mixes_only else df
        df = df.reset_index(drop = True)
        return df
    
    # load FLAC data
    dfs["flac"] = pd.read_csv(filepath_or_buffer = args.flac_csv, sep = ",", header = 0, index_col = False)
    dfs["flac"] = filter_df(df = dfs["flac"])
    facet_columns["flac"] = list(filter(lambda column: column not in standard_columns, dfs["flac"].columns))
        
    # load LDAC data
    ldac = pd.read_csv(filepath_or_buffer = args.ldac_csv, sep = ",", header = 0, index_col = False)
    dfs["ldac"] = filter_df(df = ldac[ldac["model_path"] == "/home/pnlong/.cache/descript/dac/weights_44khz_8kbps_0.0.1.pth"]).drop(columns = ["model_path"])
    dfs["lzac"] = filter_df(df = ldac[ldac["model_path"] == "/data3/pnlong/zachdac/latest/dac/weights.pth"]).drop(columns = ["model_path"])
    del ldac # free up memory
    facet_columns["ldac"] = list(filter(lambda column: column not in standard_columns, dfs["ldac"].columns))
    facet_columns["lzac"] = list(filter(lambda column: column not in standard_columns, dfs["lzac"].columns))

    ##################################################


    # PLOT STATISTICS FOR EACH COMPRESSOR
    ##################################################

    # determine suffix based on mixes_only
    suffix = "_mixes_only" if args.mixes_only else ""

    print(MAJOR_SEPARATOR_LINE)
    for i, (lossless_compressor, df) in enumerate(dfs.items()):
        
        # pretty print the data frame summary
        current_facet_columns = facet_columns[lossless_compressor]
        if len(current_facet_columns) == 0:
            summary = df[["compression_rate", "compression_speed"]].mean().to_frame().T
            summary.index = ["Overall"]
        else:
            summary = df.groupby(by = current_facet_columns)[["compression_rate", "compression_speed"]].mean()
            summary = summary.reset_index(drop = False) # make it so that group by columns are columns themselves
        print(f"{lossless_compressor.upper()} SUMMARY:")
        print(pretty_df(df = summary))
        print(MINOR_SEPARATOR_LINE if i < len(dfs) - 1 else MAJOR_SEPARATOR_LINE)
        del summary # free up memory

        # create individual plots for this compressor
        compressor_output_dir = f"{args.output_dir}/{lossless_compressor}"
        if not exists(compressor_output_dir):
            mkdir(compressor_output_dir)
            
        # construct filepaths with suffix
        percentiles_filepath = f"{compressor_output_dir}/percentiles{suffix}.pdf"
        boxplots_filepath = f"{compressor_output_dir}/boxplots{suffix}.pdf"
        
        plot_compression_statistics_percentiles(df = df, facet_columns = facet_columns[lossless_compressor], output_filepath = percentiles_filepath)
        plot_compression_statistics_boxplots(df = df, facet_columns = facet_columns[lossless_compressor], output_filepath = boxplots_filepath)

    ##################################################


    # PLOT COMPARISON PLOTS
    ##################################################

    print("Creating comparison plots...")

    # construct comparison filepaths with suffix
    percentiles_comparison_filepath = f"{args.output_dir}/percentiles_comparison{suffix}.pdf"
    boxplots_comparison_filepath = f"{args.output_dir}/boxplots_comparison{suffix}.pdf"

    plot_comparison_percentiles(dfs = dfs, facet_columns = facet_columns, output_filepath = percentiles_comparison_filepath)
    plot_comparison_boxplots(dfs = dfs, facet_columns = facet_columns, output_filepath = boxplots_comparison_filepath)

    print(f"Comparison plots saved to: {args.output_dir}")
    print(MAJOR_SEPARATOR_LINE)

    ##################################################
