# README
# Phillip Long
# May 11, 2025

# Utility variables and functions.

# IMPORTS
##################################################

from typing import Union, List, Tuple, Iterable, Any
from os.path import exists
from os import makedirs, get_terminal_size
from shutil import rmtree
import json
import pickle
import numpy as np

##################################################


# CONSTANTS
##################################################




##################################################


# MISCELLANEOUS HELPER FUNCTIONS
##################################################

def inverse_dict(d: dict):
    """Return the inverse dictionary."""
    return {v: k for k, v in d.items()}

def rep(x: object, times: int, flatten: bool = False):
    """
    An implementation of R's rep() function.
    This cannot be used to create a list of empty lists 
    (see https://stackoverflow.com/questions/240178/list-of-lists-changes-reflected-across-sublists-unexpectedly)
    ."""
    l = [x] * times
    if flatten:
        l = sum(l, [])
    return l

def unique(l: Iterable) -> list:
    """Returns the unique values from a list while retaining order."""
    return list(dict.fromkeys(list(l)))

def transpose(l: Union[List, Tuple]) -> list:
    """Tranpose a 2-dimension list."""
    return list(map(list, zip(*l)))

##################################################


# FILE HELPER FUNCTIONS
##################################################

def save_json(filepath: str, data: dict):
    """Save data as a JSON file."""
    with open(filepath, "w", encoding = "utf8") as f:
        json.dump(obj = data, fp = f)

def save_args(filepath: str, args):
    """Save the command-line arguments."""
    args_dict = {}
    for key, value in vars(args).items():
        args_dict[key] = value
    save_json(filepath = filepath, data = args_dict)

def load_json(filepath: str):
    """Load data from a JSON file."""
    with open(filepath, encoding = "utf8") as f:
        return json.load(fp = f)
    
def save_csv(filepath: str, data, header: str = ""):
    """Save data as a CSV file."""
    np.savetxt(fname = filepath, X = data, fmt = "%d", delimiter = ",", header = header, comments = "")

def load_csv(filepath: str, skiprows: int = 1):
    """Load data from a CSV file."""
    return np.loadtxt(fname = filepath, dtype = int, delimiter = ",", skiprows = skiprows)

def save_txt(filepath: str, data: list):
    """Save a list to a TXT file."""
    with open(filepath, "w", encoding = "utf8") as f:
        for item in data:
            f.write(f"{item}\n")

def load_txt(filepath: str):
    """Load a TXT file as a list."""
    with open(filepath, encoding = "utf8") as f:
        return [line.strip() for line in f]

def save_pickle(filepath: str, data: Any):
    """Save an object to a pickle file."""
    with open(filepath, "wb") as f:
        pickle.dump(obj = data, file = f)

def load_pickle(filepath: str):
    """Load a pickle file."""
    with open(filepath, "rb") as f:
        return pickle.load(file = f)
    
def count_lines(filepath: str):
    """Count the number of lines in the given file."""
    n = 0
    with open(filepath, "r", encoding = "utf8") as f:
        for _ in f:
            n += 1
    return n

def directory_creator(directory: str, reset: bool = False):
    """Helper function for creating directories."""
    if not exists(directory) or reset:
        if exists(directory):
            rmtree(directory, ignore_errors = True)
        makedirs(directory, exist_ok = True)

##################################################


# MISCELLANEOUS CONSTANTS
##################################################

# wandb constants
WANDB_PROJECT_NAME = "lossless-nac"
WANDB_RUN_NAME_FORMAT_STRING = "%m%d%y%H%M%S" # time format string for determining wandb run names

# file writing
NA_STRING = "NA"

# for multiprocessing
CHUNK_SIZE = 1

# separator line
SEPARATOR_LINE_WIDTH = get_terminal_size().columns
MAJOR_SEPARATOR_LINE = "".join(("=" for _ in range(SEPARATOR_LINE_WIDTH)))
MINOR_SEPARATOR_LINE = "".join(("-" for _ in range(SEPARATOR_LINE_WIDTH)))
DOTTED_SEPARATOR_LINE = "".join(("- " for _ in range(SEPARATOR_LINE_WIDTH // 2)))

##################################################


# WORKING WITH BITS
##################################################

class BitInputStream:
    """
    Stream object for reading in bits and bytes from a bytes stream.
    """
    
    def __init__(self, stream: bytes):
        self.stream = stream
        self.stream_iter = iter(stream)
        self.bit_buffer = None # the contents of the current byte
        self.bit_buffer_position = 0 # takes on values 0-7 (the place in the current byte)
        self.is_byte_aligned = True # no bytes have been read yet

    def read_bit(self) -> int:
        """Read a single bit."""
        if self.bit_buffer_position == 0: # read the next byte if necessary
            _ = self.read_byte() # sets bit buffer to the current byte
        current_bit = self.bit_buffer >> (7 - self.bit_buffer_position) # get the correct bit from the bit buffer
        current_bit &= 1 # mask out all but the rightmost bit
        current_bit = bool(current_bit) # convert current bit to boolean
        self.bit_buffer_position = (self.bit_buffer_position + 1) % 8
        self.is_byte_aligned = (self.bit_buffer_position == 0)
        return current_bit
    
    def read_bits(self, n: int) -> int:
        """Read `n` bits."""
        value = 0
        for _ in range(n):
            value <<= 1
            value |= self.read_bit()
        return value

    def read_byte(self) -> int:
        """Read a single byte."""
        assert self.is_byte_aligned, "Please ensure that the cursor is aligned to a byte (call `align_to_byte`)!" # ensure byte alignment
        try:
            self.bit_buffer = next(self.stream_iter) # read in current byte
        except StopIteration:
            raise RuntimeError("End of stream reached.")
        return self.bit_buffer # return the current byte

    def read_uint(self) -> int:
        """Read an unsigned integer (4 bytes)."""
        assert self.is_byte_aligned, "Please ensure that the cursor is aligned to a byte (call `align_to_byte`)!" # ensure byte alignment
        current_uint = 0
        for _ in range(4):
            current_uint <<= 8
            current_uint |= self.read_byte()
        return current_uint

    def read_int(self) -> int:
        """Read a signed integer (4 bytes)."""
        assert self.is_byte_aligned, "Please ensure that the cursor is aligned to a byte (call `align_to_byte`)!" # ensure byte alignment
        current_int = self.read_uint()
        most_significant_bit = bool(current_int >> 31) # get the most significant bit
        if most_significant_bit == True: # nothing changes if the most significant bit is 0
            current_int ^= (1 << 31) # mask out most significant bit
            current_int -= 2 ** 32 # convert to negative
        return current_int

    def align_to_byte(self):
        """Align to the closest byte boundary."""
        if not self.is_byte_aligned: # no need to align anything if already at a byte boundary
            self.bit_buffer_position = 0
            self.is_byte_aligned = True

    def reset(self):
        """Reset the cursor to the start of the stream."""
        self.stream_iter = iter(self.stream)


class BitOutputStream:
    """
    Stream object for writing bits and bytes to a bytes stream.
    """

    def __init__(self):
        self.stream = []
        self.bit_buffer = 0 # buffer to accumulate bits
        self.bit_buffer_position = 0 # how many bits are currently in the buffer
        self.is_byte_aligned = True

    def write_bit(self, bit: bool):
        """Write a single bit."""
        if bit == True: # if the bit is 1
            self.bit_buffer |= (1 << (7 - self.bit_buffer_position)) # set the bit
        self.bit_buffer_position += 1
        self.is_byte_aligned = (self.bit_buffer_position % 8 == 0)
        if self.bit_buffer_position == 8:
            self.flush_byte()

    def write_bits(self, bits: int, n: int):
        """Write `n` bits."""
        for i in range(n): # iterate over n bits
            bit = bits >> (n - i - 1) # shift relevant bit all the way to right
            bit &= 1 # mask out all but rightmost bit
            bit = bool(bit) # convert bit to boolean
            self.write_bit(bit = bit) # write the bit

    def write_byte(self, byte: int):
        """Write a single byte."""
        assert self.is_byte_aligned, "Please ensure that the cursor is aligned to a byte (call `align_to_byte`)!" # ensure byte alignment
        self.stream.append(byte)

    def write_uint(self, value: int):
        """Write an unsigned integer (4 bytes)."""
        assert self.is_byte_aligned, "Please ensure that the cursor is aligned to a byte (call `align_to_byte`)!" # ensure byte alignment
        for shift in (3, 2, 1, 0):
            self.write_byte((value >> (shift * 8)) & 0xFF)

    def write_int(self, value: int):
        """Write a signed integer (4 bytes)."""
        assert self.is_byte_aligned, "Please ensure that the cursor is aligned to a byte (call `align_to_byte`)!" # ensure byte alignment
        if value < 0:
            value += (1 << 32)
        self.write_uint(value)

    def align_to_byte(self):
        """Flush bits and align to the next byte boundary."""
        if self.bit_buffer_position > 0:
            self.flush_byte()

    def flush_byte(self):
        """Flush current buffer."""
        assert self.bit_buffer_position <= 8, "Bit buffer too large (must be <= 8 bits)!"
        self.stream.append(self.bit_buffer)
        self.bit_buffer = 0
        self.bit_buffer_position = 0
        self.is_byte_aligned = True

    def flush(self) -> bytes:
        """Flush stream contents, returning a bytes object."""
        self.align_to_byte()
        return bytes(self.stream)

##################################################
