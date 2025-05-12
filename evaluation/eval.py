# README
# Phillip Long
# May 11, 2025

# Evaluation pipeline for different compression methods to ensure losslessness.

# IMPORTS
##################################################

import argparse

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

import utils

##################################################


# CONSTANTS
##################################################



##################################################


# PARSE ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""

    # create argument parser
    parser = argparse.ArgumentParser(prog = "Evaluate", description = "Evaluation Pipeline")
    parser.add_argument("-r", "--reset", action = "store_true", help = "Rerun evaluations?")
    
    # parse arguments
    args = parser.parse_args(args = args, namespace = namespace)
    
    # return parsed arguments
    return args

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":
    
    pass


##################################################
