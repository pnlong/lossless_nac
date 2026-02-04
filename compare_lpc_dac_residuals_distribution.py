# README
# Phillip Long
# July 8, 2025

# Compares the distribution of residuals from LPC and DAC.

# IMPORTS
##################################################

import numpy as np
from collections import Counter
from typing import Dict
from os import mkdir, listdir
from os.path import exists
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
import pickle

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))

import utils
from preprocess_musdb18 import get_mixes_only_mask

##################################################


# CONSTANTS
##################################################

CONVERT_TO_ABSOLUTE_MAGNITUDES_DEFAULT = True
USE_LOG_SCALE_DEFAULT = True
MIXES_ONLY_DEFAULT = False
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan"]

CODEBOOK_LEVELS_BY_COMPRESSOR = {
    "flac": None,
    "ldac": 3,
    "lec": 4,
    "lnac": 3,
}

##################################################


# HELPER FUNCTIONS
##################################################

def get_compressor_name_from_estimator(estimator: str) -> str:
    """Map estimator key to compressor name for display."""
    if estimator == "lpc":
        return "flac"
    elif estimator.startswith("dac"):
        # Extract codebook level from estimator (e.g., "dac3" -> 3)
        n_codebooks = int(estimator[3:])
        if n_codebooks == CODEBOOK_LEVELS_BY_COMPRESSOR["ldac"]:
            return "ldac"
    elif estimator.startswith("cdac"):
        # Extract codebook level from estimator (e.g., "cdac3" -> 3)
        n_codebooks = int(estimator[4:])
        if n_codebooks == CODEBOOK_LEVELS_BY_COMPRESSOR["lnac"]:
            return "lnac"
    elif estimator == "ec4":
        return "lec"
    # If no match, return estimator as-is (shouldn't happen with proper filtering)
    return estimator

##################################################


# CONVERT COUNTER TO ABSOLUTE MAGNITUDES
##################################################

def convert_counter_to_absolute_magnitudes(counter: dict, use_log_scale: bool = USE_LOG_SCALE_DEFAULT) -> dict:
    """Convert a counter to absolute magnitudes."""
    absolute_magnitudes = dict()
    for key, value in counter.items():
        absolute_magnitudes[abs(key)] = absolute_magnitudes.get(abs(key), 0) + value
    if use_log_scale:
        absolute_magnitudes = {(np.log2(key) if key > 0 else -1): value for key, value in absolute_magnitudes.items()} # replace negative values with -1 to avoid log(0)
    return absolute_magnitudes

def convert_probabilities_to_absolute_magnitudes(probabilities: list, residual_values: list, x_values: list, use_log_scale: bool = USE_LOG_SCALE_DEFAULT) -> list:
    """Convert a list of probabilities to absolute magnitudes."""
    counter = dict(zip(residual_values, probabilities))
    counter = convert_counter_to_absolute_magnitudes(counter = counter, use_log_scale = use_log_scale)
    probabilities = [counter.get(x_value, 0) for x_value in x_values]
    return probabilities

##################################################


# PLOT THE OVERALL RESIDUALS DISTRIBUTION
##################################################

def plot_overall_residuals_distribution(residuals_dir_by_estimator: Dict[str, str], output_filepath: str, data_dir: str, mixes_only: bool = MIXES_ONLY_DEFAULT, reset: bool = False, convert_to_absolute_magnitudes: bool = CONVERT_TO_ABSOLUTE_MAGNITUDES_DEFAULT, use_log_scale: bool = USE_LOG_SCALE_DEFAULT):
    """Plot the overall residuals distribution."""

    # check for invalid combinations
    if not convert_to_absolute_magnitudes and use_log_scale:
        raise ValueError("Cannot use log scale without converting to absolute magnitudes.")
    
    # setup plot
    plt.figure(figsize = (7, 3))
    sns.set_style(style = "whitegrid")

    # for each estimator
    mean_absolute_magnitudes = dict()
    for i, (estimator, residuals_dir) in enumerate(residuals_dir_by_estimator.items()):
        
        # get compressor name for display
        compressor_name = get_compressor_name_from_estimator(estimator)
        
        # get all residual files
        try:
            residual_filepaths = [f"{residuals_dir}/{filename}" for filename in listdir(residuals_dir) if filename.endswith(".npy")]
            if mixes_only:
                residual_filepaths = [residual_filepath for residual_filepath, is_mix in zip(residual_filepaths, get_mixes_only_mask(paths = pd.Series(residual_filepaths))) if is_mix]
        except FileNotFoundError:
            print(f"Warning: {residuals_dir} not found, using existing pickle files if possible.")
            residual_filepaths = []
        
        # count frequencies across all files
        data_filepath = f"{data_dir}/{estimator}{'.mixes_only' if mixes_only else ''}.pkl"
        if not exists(data_filepath) or reset:
            counter = Counter()
            for residual_file in tqdm(iterable = residual_filepaths, desc = f"Processing {compressor_name.upper()} Residuals", total = len(residual_filepaths), leave = False):
                try:
                    residuals = np.load(residual_file)
                    counter.update(residuals.flatten())
                except Exception as e:
                    print(f"Error loading residual file {residual_file}: {e}")
                    continue
            with open(data_filepath, "wb") as f: # save counter to file
                pickle.dump(obj = counter, file = f)
            print(f"Saved {compressor_name.upper()} residuals counter to {data_filepath}.")
        else:
            with open(data_filepath, "rb") as f: # load counter from file
                counter = pickle.load(file = f)
            print(f"Loaded {compressor_name.upper()} residuals counter from {data_filepath}.")
            
        # convert to absolute magnitudes
        absolute_magnitudes = convert_counter_to_absolute_magnitudes(counter = counter, use_log_scale = False)
        if convert_to_absolute_magnitudes:
            counter = convert_counter_to_absolute_magnitudes(counter = counter, use_log_scale = use_log_scale)
        else:
            counter = dict(counter)
        
        # convert to probability distribution
        total_count = sum(counter.values())
        residual_values = sorted(counter.keys())
        probabilities = [counter.get(residual_value, 0) / total_count for residual_value in residual_values]
        print(f"Sum of Probabilities (Sanity Check): {sum(probabilities)}")
        mean_absolute_magnitudes[compressor_name] = sum(map(lambda x: x * absolute_magnitudes[x], absolute_magnitudes.keys())) / sum(absolute_magnitudes.values())
        del counter # free up memory
        
        # plot
        sns.lineplot(x = residual_values, y = probabilities, label = compressor_name.upper(), color = COLORS[i])

    # customize plot
    x_label = "|Residual|" if convert_to_absolute_magnitudes else "Residual"
    if use_log_scale:
        x_label = f"log2({x_label})"
    plt.xlabel(x_label)
    plt.ylabel("Density")
    # plt.title("Distribution of Residuals by Estimator")
    plt.legend(title = "Compressor")

    # limit x-axis range: max 16 in log2-space, max 2^16 otherwise
    xmin, xmax = plt.xlim()
    potential_xmax = 16 if use_log_scale else 2 ** 16
    plt.xlim(xmin, min(xmax, potential_xmax))
    
    # save plot
    plt.tight_layout()
    plt.savefig(output_filepath, dpi = 300, bbox_inches = "tight")
    plt.close()
    print(f"Saved overall residuals distribution plot to {output_filepath}.")

    # print mean absolute magnitudes
    print(utils.MINOR_SEPARATOR_LINE)
    for compressor_name, mean_absolute_magnitude in mean_absolute_magnitudes.items():
        print(f"{compressor_name.upper()} Mean Absolute Magnitude: {mean_absolute_magnitude:.2f}")
    print(f"Therefore, if we want to find the best compressor, we should choose the one with the lowest mean absolute magnitude: {min(mean_absolute_magnitudes, key = mean_absolute_magnitudes.get).upper()}.")

    # return nothing
    return

##################################################


# PLOT THE MEAN RESIDUALS DISTRIBUTION
##################################################

def plot_mean_residuals_distribution(residuals_dir_by_estimator: Dict[str, str], output_filepath: str, data_dir: str, mixes_only: bool = MIXES_ONLY_DEFAULT, reset: bool = False, convert_to_absolute_magnitudes: bool = CONVERT_TO_ABSOLUTE_MAGNITUDES_DEFAULT, use_log_scale: bool = USE_LOG_SCALE_DEFAULT):
    """Plot the mean residuals distribution."""

    # check for invalid combinations
    if not convert_to_absolute_magnitudes and use_log_scale:
        raise ValueError("Cannot use log scale without converting to absolute magnitudes.")

    # setup plot
    plt.figure(figsize = (7, 3))
    sns.set_style(style = "whitegrid")

    # for each estimator
    for i, (estimator, residuals_dir) in enumerate(residuals_dir_by_estimator.items()):
        
        # get compressor name for display
        compressor_name = get_compressor_name_from_estimator(estimator)
        
        # get all residual files
        try:
            residual_filepaths = [f"{residuals_dir}/{filename}" for filename in listdir(residuals_dir) if filename.endswith(".npy")]
            if mixes_only:
                residual_filepaths = [residual_filepath for residual_filepath, is_mix in zip(residual_filepaths, get_mixes_only_mask(paths = pd.Series(residual_filepaths))) if is_mix]      
        except FileNotFoundError:
            print(f"Warning: {residuals_dir} not found, using existing pickle files if possible.")
            residual_filepaths = []

        # get range of residual values efficiently
        range_filepath = f"{data_dir}/{estimator}_range{'.mixes_only' if mixes_only else ''}.pkl"
        if not exists(range_filepath) or reset:
            min_val = float('inf')
            max_val = float('-inf')
            for residual_file in tqdm(iterable = residual_filepaths, desc = f"Finding range for {compressor_name.upper()}", total = len(residual_filepaths), leave = False):
                try:
                    residuals = np.load(residual_file)
                    min_val = min(min_val, residuals.min())
                    max_val = max(max_val, residuals.max())
                    del residuals
                except Exception as e:
                    print(f"Error loading residual file {residual_file}: {e}")
                    continue
            with open(range_filepath, "wb") as f:
                pickle.dump(obj = (min_val, max_val), file = f)
            print(f"Saved {compressor_name.upper()} range to {range_filepath}.")
        else:
            with open(range_filepath, "rb") as f:
                min_val, max_val = pickle.load(file = f)
            print(f"Loaded {compressor_name.upper()} range from {range_filepath}.")
        
        # create residual values list - limit range to prevent memory issues
        min_val, max_val = int(min_val), int(max_val)
        all_residual_values = list(range(min_val, max_val + 1, 1))

        # process each file to get matrix of probability distributions with shape (len(residual_filepaths), len(all_residual_values))
        all_distributions_filepath = f"{data_dir}/{estimator}_distributions{'.mixes_only' if mixes_only else ''}.npy"
        if not exists(all_distributions_filepath) or reset:
            all_distributions = np.zeros(shape = (len(residual_filepaths), len(all_residual_values)), dtype = np.float32) # store probability distributions for each file
            for j, residual_file in tqdm(iterable = enumerate(residual_filepaths), desc = f"Processing {compressor_name.upper()} Residuals", total = len(residual_filepaths), leave = False):
                residuals = np.load(residual_file)
                counter = Counter(residuals.flatten()) # get residuals and count frequencies
                total_count = sum(counter.values())
                probabilities = [counter[residual_value] / total_count for residual_value in all_residual_values] # convert to probability distribution
                all_distributions[j, :] = probabilities
            np.save(file = all_distributions_filepath, arr = all_distributions)
            print(f"Saved {compressor_name.upper()} distributions to {all_distributions_filepath}.")
        else:
            all_distributions = np.load(all_distributions_filepath)
            print(f"Loaded {compressor_name.upper()} distributions from {all_distributions_filepath}.")

        # set x values to absolute magnitudes if converting to absolute magnitudes
        if convert_to_absolute_magnitudes:
            x_values = sorted(list(set(map(abs, all_residual_values))))
            if use_log_scale:
                x_values = [np.log2(x_value) if x_value > 0 else -1 for x_value in x_values]
        else:
            x_values = all_residual_values

        # plot individual distribution with low alpha (no legend to avoid clutter)
        for probabilities in tqdm(iterable = all_distributions, desc = f"Plotting {compressor_name.upper()} distributions", total = len(all_distributions), leave = False):
            if convert_to_absolute_magnitudes:
                probabilities = convert_probabilities_to_absolute_magnitudes(probabilities = probabilities, residual_values = all_residual_values, x_values = x_values, use_log_scale = use_log_scale)
            sns.lineplot(x = x_values, y = probabilities, alpha = 0.01, color = COLORS[i])
                    
        # calculate and plot mean distribution
        mean_distribution = np.mean(a = all_distributions, axis = 0)
        if convert_to_absolute_magnitudes:
            mean_distribution = convert_probabilities_to_absolute_magnitudes(probabilities = mean_distribution, residual_values = all_residual_values, x_values = x_values, use_log_scale = use_log_scale)
        sns.lineplot(x = x_values, y = mean_distribution, label = compressor_name.upper(), alpha = 1.0, color = COLORS[i])

    # customize plot
    x_label = "|Residual|" if convert_to_absolute_magnitudes else "Residual"
    if use_log_scale:
        x_label = f"log2({x_label})"
    plt.xlabel(x_label)
    plt.ylabel("Density")
    # plt.title("Distribution of Residuals by Estimator")
    
    # create legend with unique labels only
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), title = "Compressor")

    # limit x-axis range: max 16 in log2-space, max 2^16 otherwise
    xmin, xmax = plt.xlim()
    potential_xmax = 16 if use_log_scale else 2 ** 16
    plt.xlim(xmin, min(xmax, potential_xmax))
    
    # save plot
    plt.tight_layout()
    plt.savefig(output_filepath, dpi = 300, bbox_inches = "tight")
    plt.close()
    print(f"Saved mean residuals distribution plot to {output_filepath}.")

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
        parser = argparse.ArgumentParser(prog = "Plots", description = "Create Plots to Compare LPC and DAC Residuals") # create argument parser
        parser.add_argument("--output_dir", type = str, default = utils.EVAL_DIR, help = "Absolute filepath to the output directory where the plots directory will be created if it does not yet exist.")
        parser.add_argument("--mixes_only", action = "store_true", help = "Compute statistics for only mixes in MUSDB18, not all stems.")
        parser.add_argument("--use_old_ldac", action = "store_true", help = "Use the old LDAC directory.")
        parser.add_argument("--reset", action = "store_true", help = "Reset the data directory.")
        args = parser.parse_args(args = args, namespace = namespace) # parse arguments
        return args # return parsed arguments
    args = parse_args()

    # create files directory if not yet exists
    base_dir = f"{args.output_dir}/lpc_dac_residuals_distribution"
    if not exists(base_dir):
        mkdir(base_dir)

    # create plots directory if not yet exists
    plots_dir = f"{base_dir}/plots"
    if not exists(plots_dir):
        mkdir(plots_dir)

    # create data directory if not yet exists
    data_dir = f"{base_dir}/data"
    if not exists(data_dir):
        mkdir(data_dir)

    ##################################################


    # GET PATHS TO RESIDUAL NUMPY FILES
    ##################################################

    # initialize dictionary of residual paths, with the lossy estimator as the key
    residuals_dir_by_estimator = dict()

    # get paths to LPC residuals from FLAC itself because this directory is organized differently
    lpc_residuals_dir = f"{utils.LOGGING_FOR_ZACH_DIR}/flac/data"
    if not exists(lpc_residuals_dir):
        print(f"Warning: LPC residuals directory not found: {lpc_residuals_dir}")
    residuals_dir_by_estimator["lpc"] = lpc_residuals_dir
    
    # get paths to DAC residuals
    if not exists(utils.LOGGING_FOR_ZACH_DIR):
        print(f"Warning: Logging directory not found: {utils.LOGGING_FOR_ZACH_DIR}")
    elif not exists(utils.LOGGING_FOR_ZACH_FILEPATH):
        print(f"Warning: Logging CSV file not found: {utils.LOGGING_FOR_ZACH_FILEPATH}")
    else:
        ldac_parent_dir = utils.LOGGING_FOR_ZACH_DIR
        ldac_dirs = list(filter(lambda x: x.startswith("ldac"), listdir(ldac_parent_dir)))
        ldac_dirs_parameters_hashes = list(map(lambda x: x.split("_")[-1], ldac_dirs))
        residuals_log_table = pd.read_csv(filepath_or_buffer = utils.LOGGING_FOR_ZACH_FILEPATH, sep = ",", header = 0, index_col = False)
        for ldac_dir, ldac_dir_parameters_hash in zip(ldac_dirs, ldac_dirs_parameters_hashes):
            parameters = residuals_log_table[residuals_log_table["parameters_hash"] == ldac_dir_parameters_hash].reset_index(drop = True).at[0, "parameters"]
            parameters = dict([parameter.split(":") for parameter in parameters.split("-")])
            n_codebooks_for_ldac_dir = int(parameters["n_codebooks"])
            # Only include if it matches the specified codebook level for LDAC
            if n_codebooks_for_ldac_dir == CODEBOOK_LEVELS_BY_COMPRESSOR["ldac"]:
                residuals_dir_by_estimator[f"dac{n_codebooks_for_ldac_dir}"] = f"{ldac_parent_dir}{'/old_ldac' if args.use_old_ldac else ''}/{ldac_dir}" # add to dictionary
        # residuals_dir_by_estimator = {"dac3": "/deepfreeze/user_shares/pnlong/lnac/logging_for_zach/old_ldac/ldac_596c2b31112908f4fba2c31153fc15421c58738badb4a0350833d497e7abaa4e", "dac6": "/deepfreeze/user_shares/pnlong/lnac/logging_for_zach/old_ldac/ldac_c897ac32fbb1926fe922bc427e8e35f70b4fd705325d8201944780e2f943f843", "dac9": "/deepfreeze/user_shares/pnlong/lnac/logging_for_zach/old_ldac/ldac_fe275ec542f94cf437b01be9e4a801303b9f663864a2f855453c2f7a775aa339"}

        # get paths to EC4 residuals (LEC with n_codebooks 4)
        lec_dirs = list(filter(lambda x: x.startswith("lec"), listdir(ldac_parent_dir)))
        for lec_dir in sorted(lec_dirs):
            lec_dir_parameters_hash = lec_dir.split("_")[-1]
            matching = residuals_log_table[residuals_log_table["parameters_hash"] == lec_dir_parameters_hash]
            if len(matching) == 0:
                continue
            parameters = matching.reset_index(drop = True).at[0, "parameters"]
            parameters = dict([parameter.split(":") for parameter in parameters.split("-")])
            n_codebooks = int(parameters["n_codebooks"])
            if n_codebooks == 4:
                residuals_dir_by_estimator["ec4"] = f"{ldac_parent_dir}/{lec_dir}"
                break
        
        # get paths to CDAC residuals (LNAC with n_codebooks matching specified level)
        lnac_dirs = list(filter(lambda x: x.startswith("lnac"), listdir(ldac_parent_dir)))
        for lnac_dir in sorted(lnac_dirs):
            lnac_dir_parameters_hash = lnac_dir.split("_")[-1]
            matching = residuals_log_table[residuals_log_table["parameters_hash"] == lnac_dir_parameters_hash]
            if len(matching) == 0:
                continue
            parameters = matching.reset_index(drop = True).at[0, "parameters"]
            parameters = dict([parameter.split(":") for parameter in parameters.split("-")])
            n_codebooks = int(parameters["n_codebooks"])
            # Only include if it matches the specified codebook level for LNAC
            if n_codebooks == CODEBOOK_LEVELS_BY_COMPRESSOR["lnac"]:
                residuals_dir_by_estimator[f"cdac{n_codebooks}"] = f"{ldac_parent_dir}/{lnac_dir}"


    ##################################################


    # CALL PLOTTING FUNCTIONS
    ##################################################

    # plot overall residuals distribution
    print(utils.MAJOR_SEPARATOR_LINE)  
    overall_residuals_distribution_dir = f"{data_dir}/overall_residuals_distribution"
    if not exists(overall_residuals_distribution_dir):
        mkdir(overall_residuals_distribution_dir)
    print("Plotting Overall Residuals Distribution:")
    plot_overall_residuals_distribution(
        residuals_dir_by_estimator = residuals_dir_by_estimator,
        output_filepath = f"{plots_dir}/overall_residuals_distribution.pdf" if (not args.mixes_only) else f"{plots_dir}/overall_residuals_distribution_mixes_only.pdf",
        mixes_only = args.mixes_only,
        data_dir = overall_residuals_distribution_dir,
        reset = args.reset,
        convert_to_absolute_magnitudes = CONVERT_TO_ABSOLUTE_MAGNITUDES_DEFAULT,
        use_log_scale = USE_LOG_SCALE_DEFAULT,
    )

    # plot mean residuals distribution
    print(utils.MINOR_SEPARATOR_LINE)
    print("Plotting Mean Residuals Distribution:")
    mean_residuals_distribution_dir = f"{data_dir}/mean_residuals_distribution"
    if not exists(mean_residuals_distribution_dir):
        mkdir(mean_residuals_distribution_dir)
    plot_mean_residuals_distribution(
        residuals_dir_by_estimator = residuals_dir_by_estimator,
        output_filepath = f"{plots_dir}/mean_residuals_distribution.pdf" if (not args.mixes_only) else f"{plots_dir}/mean_residuals_distribution_mixes_only.pdf",
        mixes_only = args.mixes_only,
        data_dir = mean_residuals_distribution_dir,
        reset = args.reset,
        convert_to_absolute_magnitudes = CONVERT_TO_ABSOLUTE_MAGNITUDES_DEFAULT,
        use_log_scale = USE_LOG_SCALE_DEFAULT,
    )
    print(utils.MAJOR_SEPARATOR_LINE)

    ##################################################

##################################################