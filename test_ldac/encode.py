# README
# Phillip Long
# July 19, 2025

# Encoder script for Lossless Descript Audio Codec (LDAC).

# IMPORTS
##################################################

import numpy as np
from typing import List, Tuple, Dict, Any
import warnings
import multiprocessing
import torch
from audiotools import AudioSignal
import logging

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))
sys.path.insert(0, dirname((dirname(realpath(__file__)))))
sys.path.insert(0, f"{dirname(dirname(realpath(__file__)))}/dac") # for dac import

import utils
import dac

# ignore deprecation warning from pytorch
warnings.filterwarnings(action = "ignore", message = "torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm")

##################################################


# VERBATIM ENTROPY CODER
##################################################

# verbatim
def entropy_encode_verbatim(data):
    return data

##################################################


# NAIVE RICE ENTROPY CODER
##################################################

# naive rice
def entropy_encode_naive_rice(data):
    return data

##################################################


# ADAPTIVE RICE ENTROPY CODER
##################################################

# adaptive rice
def entropy_encode_adaptive_rice(data):
    return data

##################################################


# NAIVE DAC LOSSLESS COMPRESSOR ENCODE
##################################################


##################################################


# ADAPTIVE DAC LOSSLESS COMPRESSOR ENCODE
##################################################


##################################################


# ENCODE FUNCTION
##################################################


##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # get command line arguments
    
    # load DAC model

    # encode
    pass

##################################################