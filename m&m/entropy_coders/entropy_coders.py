# README
# Phillip Long
# July 6, 2025

# Entropy Coder interface.

# IMPORTS
##################################################

from abc import ABC, abstractmethod
import numpy as np

##################################################


# CONSTANTS
##################################################



##################################################


# ENTROPY CODER INTERFACE
##################################################

class EntropyCoder(ABC):
    """
    Abstract base class for entropy coders.
    """
    
    @property
    def type_(self) -> str:
        """
        The type of entropy coder. Defaults to the class name.

        Returns
        -------
        str
            The type of entropy coder.
        """
        return self.__class__.__name__

    @abstractmethod
    def __init__(self):
        """
        Initialize the entropy coder.
        """
        pass
    
    @abstractmethod
    def encode(self, nums: np.array) -> bytes:
        """
        Encode the data.

        Parameters
        ----------
        nums : np.array
            The data to encode.

        Returns
        -------
        bytes
            The encoded data.
        """
        pass

    @abstractmethod
    def decode(self, stream: bytes, num_samples: int) -> np.array:
        """
        Decode the data.

        Parameters
        ----------
        stream : bytes
            The encoded data to decode.
        num_samples : int
            The number of samples to decode.

        Returns
        -------
        np.array
            The decoded data.
        """
        pass

    def get_compressed_size(self, stream: bytes) -> int:
        """
        Get the compressed size of the data in bytes.

        Parameters
        ----------
        stream : bytes
            The encoded data.
        
        Returns
        -------
        int
            The compressed size of the data in bytes.
        """
        return len(stream)
        
##################################################


# HELPER FUNCTIONS
##################################################

def int_to_pos(self, x: int) -> int:
    """Maps any integer onto a non-negative integer."""
    if x >= 0: # if x is non-negative
        return 2 * x # map positive values onto even numbers
    else: # if x < 0 (x is negative)
        return (-2 * x) - 1 # map negative values onto odd numbers

def inverse_int_to_pos(self, x: int) -> int:
    """Inverse to the previous function."""
    if x % 2 == 0: # if x is an even number
        return x // 2 # then x must be non-negative
    else: # if x is an odd number
        return (x + 1) // -2 # then x must be negative

def get_dtype_from_bytes_per_element(bytes_per_element: int) -> np.dtype:
    """Get the numpy dtype from the bytes per element."""
    return np.dtype(f"int{bytes_per_element * 8}")

##################################################