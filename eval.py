# README
# Phillip Long
# May 11, 2025

# Evaluation pipeline for different compression methods to ensure losslessness.

# IMPORTS
##################################################

import argparse
import numpy as np
import scipy

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

import utils
import flac

##################################################


# CONSTANTS
##################################################

SAMPLE_RATE = 44100

##################################################


# PARSE ARGUMENTS
##################################################

def parse_args(args = None, namespace = None):
    """Parse command-line arguments."""

    # create argument parser
    parser = argparse.ArgumentParser(prog = "Evaluate", description = "Evaluation Pipeline")
    parser.add_argument("-w", "--wav", required = True, type = str, help = "Absolute filepath to the waveform.")
    
    # parse arguments
    args = parser.parse_args(args = args, namespace = namespace)
    
    # return parsed arguments
    return args

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":

    # read in arguments
    args = parse_args()

    # load in wav file
    sample_rate, waveform = scipy.io.wavfile.read(filename = args.wav)
    print(f"Waveform Shape: {tuple(waveform.shape)}")
    print(f"Waveform Data Type: {waveform.dtype}")
    
    # test to ensure losslessness
    bottleneck = flac.encode(waveform)
    np.save(file = "myformat.npy", arr = bottleneck)
    round_trip = flac.decode(np.load(file = "myformat.npy"))
    assert np.array_equal(waveform, round_trip)


##################################################
