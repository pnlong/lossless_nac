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
    @abstractmethod
    def type_(self) -> str:
        """
        Get the type of the entropy coder.

        Returns
        -------
        str
            The type of the entropy coder.
        """
        return self.__class__.__name__

    @abstractmethod
    def __init__(
        self,
    ):
        """
        Initialize the entropy coder.
        """
        pass
    
    @abstractmethod
    def encode(
        self,
        nums: np.ndarray,
    ) -> bytes:
        """
        Encode the data.

        Parameters
        ----------
        nums : np.ndarray
            The data to encode.

        Returns
        -------
        bytes
            The encoded data.
        """
        pass

    @abstractmethod
    def decode(
        self,
        stream: bytes,
        num_samples: int,
    ) -> np.ndarray:
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
        np.ndarray
            The decoded data.
        """
        pass

    def get_compressed_size(
        self,
        stream: bytes,
    ) -> int:
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