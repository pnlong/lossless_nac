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

##################################################


# PLOT THE OVERALL RESIDUALS DISTRIBUTION
##################################################


def plot_overall_residuals_distribution(residuals_dir_by_estimator: Dict[str, str], output_filepath: str, data_dir: str, reset: bool = False):
    """Plot the overall residuals distribution."""
    
    # setup plot
    plt.figure(figsize = (12, 8))
    sns.set_style(style = "whitegrid")

    # for each estimator
    for estimator, residuals_dir in residuals_dir_by_estimator.items():
        
        # get all residual files
        residual_filepaths = [f"{residuals_dir}/{filename}" for filename in listdir(residuals_dir) if filename.endswith(".npy")]
        
        # count frequencies across all files
        data_filepath = f"{data_dir}/{estimator}.pkl"
        if not exists(data_filepath) or reset:
            counter = Counter()
            for residual_file in tqdm(iterable = residual_filepaths, desc = f"Processing {estimator.upper()} Residuals", total = len(residual_filepaths), leave = False):
                residuals = np.load(residual_file)
                counter.update(residuals.flatten())
            with open(data_filepath, "wb") as f: # save counter to file
                pickle.dump(obj = counter, file = f)
        else:
            with open(data_filepath, "rb") as f: # load counter from file
                counter = pickle.load(file = f)
            
        # convert to probability distribution
        total_count = sum(counter.values())
        residual_values = sorted(counter.keys())
        probabilities = [counter[residual_value] / total_count for residual_value in residual_values]
        del counter # free up memory
        
        # plot
        sns.lineplot(x = residual_values, y = probabilities, label = estimator.upper())

    # customize plot
    plt.xlabel("Residual Value")
    plt.ylabel("Probability")
    plt.title("Distribution of Residuals by Estimator")
    plt.legend(title = "Estimator")
    
    # save plot
    plt.tight_layout()
    plt.savefig(output_filepath, dpi = 300, bbox_inches = "tight")
    plt.close()

    # return nothing
    return

##################################################


# PLOT THE MEAN RESIDUALS DISTRIBUTION
##################################################

def plot_mean_residuals_distribution(residuals_dir_by_estimator: Dict[str, str], output_filepath: str, data_dir: str, reset: bool = False):
    """Plot the mean residuals distribution."""

    # setup plot
    plt.figure(figsize = (12, 8))
    sns.set_style(style = "whitegrid")

    # for each estimator
    for estimator, residuals_dir in residuals_dir_by_estimator.items():
        
        # get all residual files
        residual_filepaths = [f"{residuals_dir}/{filename}" for filename in listdir(residuals_dir) if filename.endswith(".npy")]        

        # get range of residual values efficiently
        range_filepath = f"{data_dir}/{estimator}_range.pkl"
        if not exists(range_filepath) or reset:
            min_val = float('inf')
            max_val = float('-inf')
            for residual_file in tqdm(iterable = residual_filepaths, desc = f"Finding range for {estimator.upper()}", total = len(residual_filepaths), leave = False):
                residuals = np.load(residual_file)
                min_val = min(min_val, residuals.min())
                max_val = max(max_val, residuals.max())
                del residuals
            with open(range_filepath, "wb") as f:
                pickle.dump(obj = (min_val, max_val), file = f)
        else:
            with open(range_filepath, "rb") as f:
                min_val, max_val = pickle.load(file = f)
        
        # create residual values list - limit range to prevent memory issues
        min_val, max_val = int(min_val), int(max_val)
        all_residual_values = list(range(min_val, max_val + 1, 1))

        # process each file to get matrix of probability distributions with shape (len(residual_filepaths), len(all_residual_values))
        all_distributions_filepath = f"{data_dir}/{estimator}_distributions.npy"
        if not exists(all_distributions_filepath) or reset:
            all_distributions = np.zeros(shape = (len(residual_filepaths), len(all_residual_values)), dtype = np.float32) # store probability distributions for each file
            for i, residual_file in tqdm(iterable = enumerate(residual_filepaths), desc = f"Processing {estimator.upper()} Residuals", total = len(residual_filepaths), leave = False):
                residuals = np.load(residual_file)
                counter = Counter(residuals.flatten()) # get residuals and count frequencies
                total_count = sum(counter.values())
                probabilities = [counter[residual_value] / total_count for residual_value in all_residual_values] # convert to probability distribution
                all_distributions[i, :] = probabilities
            np.save(file = all_distributions_filepath, arr = all_distributions)
        else:
            all_distributions = np.load(all_distributions_filepath)

        # plot individual distribution with low alpha (no legend to avoid clutter)
        for probabilities in all_distributions:
            plt.plot(all_residual_values, probabilities, alpha = 0.1, label = estimator.upper(), legend = False)
                    
        # calculate and plot mean distribution
        mean_distribution = np.mean(a = all_distributions, axis = 0)
        sns.lineplot(x = all_residual_values, y = mean_distribution, label = estimator.upper(), alpha = 1.0)

    # customize plot
    plt.xlabel("Residual Value")
    plt.ylabel("Probability")
    plt.title("Distribution of Residuals by Estimator")
    plt.legend(title = "Estimator")
    
    # save plot
    plt.tight_layout()
    plt.savefig(output_filepath, dpi = 300, bbox_inches = "tight")
    plt.close()

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
    else:
        residuals_dir_by_estimator["lpc"] = lpc_residuals_dir
    
    # get paths to DAC residuals
    if not exists(utils.LOGGING_FOR_ZACH_DIR):
        print(f"Warning: Logging directory not found: {utils.LOGGING_FOR_ZACH_DIR}")
    elif not exists(utils.LOGGING_FOR_ZACH_FILEPATH):
        print(f"Warning: Logging CSV file not found: {utils.LOGGING_FOR_ZACH_FILEPATH}")
    else:
        ldac_dirs = list(filter(lambda x: x.startswith("ldac"), listdir(utils.LOGGING_FOR_ZACH_DIR)))
        ldac_dirs_parameters_hashes = list(map(lambda x: x.split("_")[-1], ldac_dirs))
        residuals_log_table = pd.read_csv(filepath_or_buffer = utils.LOGGING_FOR_ZACH_FILEPATH, sep = ",", header = 0, index_col = False)
        for ldac_dir, ldac_dir_parameters_hash in zip(ldac_dirs, ldac_dirs_parameters_hashes):
            parameters = residuals_log_table[residuals_log_table["parameters_hash"] == ldac_dir_parameters_hash].reset_index(drop = True).at[0, "parameters"]
            parameters = dict([parameter.split(":") for parameter in parameters.split("-")])
            n_codebooks_for_ldac_dir = parameters["n_codebooks"]
            residuals_dir_by_estimator[f"dac{n_codebooks_for_ldac_dir}"] = f"{utils.LOGGING_FOR_ZACH_DIR}/{ldac_dir}" # add to dictionary

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
        data_dir = overall_residuals_distribution_dir,
        reset = args.reset,
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
        data_dir = mean_residuals_distribution_dir,
        reset = args.reset,
    )
    print(utils.MAJOR_SEPARATOR_LINE)

    ##################################################

##################################################