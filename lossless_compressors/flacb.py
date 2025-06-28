# README
# Phillip Long
# June 26, 2025

# Implementation of Free Lossless Audio Codec Bindings (FLACB) for use as a baseline.
# See https://xiph.org/flac/documentation_format_overview.html for more.

# IMPORTS
##################################################

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

import utils

##################################################


# CONSTANTS
##################################################

LIBFLAC_PATH = f"{dirname(dirname(realpath(__file__)))}/libflac/src/libflac"

##################################################


# ENCODE
##################################################

def encode(waveform: np.array, log_for_zach_kwargs: dict = None):


##################################################


# DECODE
##################################################


##################################################


# HELPER FUNCTION TO GET THE SIZE IN BYTES OF THE BOTTLENECK
##################################################


##################################################


# MAIN METHODS
##################################################

if __name__ == "__main__":

    # read in arguments
    def parse_args(args = None, namespace = None):
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser(prog = "Evaluate", description = "Evaluate FLACB Implementation on a Test File") # create argument parser
        parser.add_argument("-p", "--path", type = str, default = f"{dirname(dirname(realpath(__file__)))}/test.wav", help = "Absolute filepath to the WAV file.")
        parser.add_argument("--mono", action = "store_true", help = "Ensure that the WAV file is mono (single-channeled).")
        args = parser.parse_args(args = args, namespace = namespace) # parse arguments
        return args # return parsed arguments
    args = parse_args()

    # load in wav file
    sample_rate, waveform = scipy.io.wavfile.read(filename = args.path)

    # force to mono if necessary
    if args.mono and waveform.ndim == 2:
        print("Forcing waveform to mono!")
        waveform = np.round(np.mean(a = waveform, axis = -1)).astype(waveform.dtype)

    # print statistics about waveform
    print(f"Waveform Shape: {tuple(waveform.shape)}")
    print(f"Waveform Sample Rate: {sample_rate:,} Hz")
    print(f"Waveform Data Type: {waveform.dtype}")
    waveform_size = utils.get_waveform_size(waveform = waveform)
    print(f"Waveform Size: {waveform_size:,} bytes")
    
    # encode
    print("Encoding...")
    start_time = time.perf_counter()
    bottleneck = encode(waveform = waveform)
    compression_speed = utils.get_compression_speed(duration_audio = len(waveform) / sample_rate, duration_encoding = time.perf_counter() - start_time)
    del start_time # free up memory
    bottleneck_size = get_bottleneck_size(bottleneck = bottleneck) # compute size of bottleneck in bytes
    print(f"Bottleneck Size: {bottleneck_size:,} bytes")
    print(f"Compression Rate: {100 * utils.get_compression_rate(size_original = waveform_size, size_compressed = bottleneck_size):.4f}%")
    print(f"Compression Speed: {compression_speed:.4f}")

    # decode
    print("Decoding...")
    round_trip = decode(bottleneck = bottleneck, interchannel_decorrelate = args.interchannel_decorrelate, k = args.rice_parameter)

    # verify losslessness
    assert np.array_equal(waveform, round_trip), "Original and reconstructed waveforms do not match!"
    print("Encoding is lossless!")

##################################################