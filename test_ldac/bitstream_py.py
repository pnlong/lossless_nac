# README
# Phillip Long
# July 19, 2025

# Helper objects for reading and writing bit streams.

# IMPORTS
##################################################

from bitarray import bitarray
from bitarray.util import ba2int, int2ba

##################################################


# CONSTANTS
##################################################

ENDIANESS = "little"
DEFAULT_BUFFER_SIZE = (1024 ** 2) # initialize bytes buffer to 1MB by default

##################################################


# BIT INPUT STREAM
##################################################

class BitInputStream:
    """
    Stream object for reading in bits and bytes from a bytes stream using bitarray.
    """
    
    def __init__(self, path: str = None, stream: bytes = None):
        """
        Initialize the bit input stream.

        Parameters
        ----------
        path : str, optional
            The path to the file to read from.
        stream : bytes, optional
            The bytes stream to read from.
        """
        self.path = path
        self.original_stream = stream
        self.reset()

    def get_position(self) -> int:
        """
        Get the current position in the stream.

        Returns
        -------
        int
            The current position in the stream.
        """
        return self.position

    def read_bit(self) -> bool:
        """
        Read a single bit.

        Returns
        -------
        bool
            The bit.
        """
        position_after = self.position + 1
        if position_after > len(self.stream):
            raise RuntimeError("End of stream reached.")
        bit = bool(self.stream[self.position])
        self.position = position_after
        self.is_byte_aligned = (self.position % 8 == 0)
        return bit
    
    def read_bits(self, n: int) -> int:
        """
        Read `n` bits.

        Parameters
        ----------
        n : int
            The number of bits to read.

        Returns
        -------
        int
            The bits.
        """
        position_after = self.position + n
        if position_after > len(self.stream):
            raise RuntimeError("End of stream reached.")
        value = ba2int(self.stream[self.position:position_after])
        self.position = position_after
        self.is_byte_aligned = (self.position % 8 == 0)
        return value

    def read_byte(self) -> int:
        """
        Read a single byte.

        Returns
        -------
        int
            The byte.
        """
        return self.read_bits(n = 8)

    def read_bytes(self, n: int) -> bytes:
        """
        Read `n` bytes.

        Parameters
        ----------
        n : int
            The number of bytes to read.

        Returns
        -------
        bytes
            The bytes.
        """
        position_after = self.position + (n * 8)
        if position_after > len(self.stream):
            raise RuntimeError("End of stream reached.")
        bytes_ = self.stream[self.position:position_after]
        bytes_ = bytes_.tobytes()
        self.position = position_after
        self.is_byte_aligned = (self.position % 8 == 0)
        return bytes_

    def read_uint(self) -> int:
        """
        Read an unsigned integer (4 bytes).

        Returns
        -------
        int
            The unsigned integer.
        """
        return self.read_bits(n = 32)

    def align_to_byte(self):
        """Align to the closest byte boundary."""
        if not self.is_byte_aligned:
            self.position += 8 - (self.position % 8)
            self.is_byte_aligned = True

    def reset(self):
        """Reset the cursor to the start of the stream."""
        if self.path is not None:
            with open(self.path, mode = "rb") as f:
                self.stream = bitarray(endian = ENDIANESS)
                self.stream.frombytes(f.read())
        elif self.original_stream is not None:
            self.stream = bitarray(endian = ENDIANESS)
            self.stream.frombytes(self.original_stream)
        else:
            raise ValueError("Either a path or a stream must be provided.")
        self.position = 0 # current bit position in stream
        self.is_byte_aligned = True

##################################################


# BIT OUTPUT STREAM
##################################################

class BitOutputStream:
    """
    Stream object for writing bits and bytes to a bytes stream using bitarray.
    """

    def __init__(self, path: str = None, buffer_size: int = DEFAULT_BUFFER_SIZE):
        """
        Initialize the bit output stream.

        Parameters
        ----------
        path : str, optional
            The path to the file to write to. If None, calling the `write` method will do nothing.
        buffer_size : int, default = DEFAULT_BUFFER_SIZE
            Initial size of the buffer in bytes.
        """
        self.path = path
        assert buffer_size > 0, "Buffer size must be greater than 0!"
        self.buffer_size = buffer_size
        self.buffer_size_bits = self.buffer_size * 8
        self.stream = bitarray(self.buffer_size_bits, endian = ENDIANESS)
        self.position = 0 # current bit position in stream
        self.is_byte_aligned = True

    def get_position(self) -> int:
        """
        Get the current position in the stream.

        Returns
        -------
        int
            The current position in the stream.
        """
        return self.position

    def _extend_buffer(self, n: int = None):
        """
        Extend the buffer if necessary.
        
        Parameters
        ----------
        n : int, optional
            The number of bits to extend the buffer by. If None, extend the buffer by the buffer size.
        """
        if n is None:
            n = self.buffer_size_bits
        self.stream.extend(bitarray(n, endian = ENDIANESS))

    def write_bit(self, bit: bool):
        """
        Write a single bit.

        Parameters
        ----------
        bit : bool
            The bit to write.
        """
        position_after = self.position + 1
        if position_after > len(self.stream):
            self._extend_buffer()
        self.stream[self.position] = bit
        self.position = position_after
        self.is_byte_aligned = (self.position % 8 == 0)

    def write_bits(self, bits: int, n: int):
        """
        Write `n` bits.

        Parameters
        ----------
        bits : int
            The bits to write.
        n : int
            The number of bits to write.
        """
        assert n < self.buffer_size_bits, "Cannot write more than the buffer size!"
        position_after = self.position + n
        if position_after > len(self.stream):
            self._extend_buffer()
        self.stream[self.position:position_after] = int2ba(int(bits), length = n, endian = ENDIANESS)
        self.position = position_after
        self.is_byte_aligned = (self.position % 8 == 0)

    def write_byte(self, byte: int):
        """
        Write a single byte.

        Parameters
        ----------
        byte : int
            The byte to write.
        """
        self.write_bits(bits = byte, n = 8)

    def write_bytes(self, bytes_: bytes):
        """
        Special case for writing a sequence of bytes.

        Parameters
        ----------
        bytes_ : bytes
            The bytes to write.
        """
        bytes_ = bitarray(bytes_, endian = ENDIANESS)
        n = len(bytes_)
        position_after = self.position + n
        if position_after > len(self.stream):
            self._extend_buffer(n = n)
        self.stream[self.position:position_after] = bytes_
        self.position = position_after
        self.is_byte_aligned = (self.position % 8 == 0)

    def write_uint(self, value: int):
        """
        Write an unsigned integer (4 bytes).

        Parameters
        ----------
        value : int
            The unsigned integer to write.
        """
        self.write_bits(bits = value, n = 32)

    def align_to_byte(self):
        """Align to the next byte boundary by padding with zeros."""
        if not self.is_byte_aligned:
            padding = 8 - (self.position % 8)
            self.write_bits(bits = 0, n = padding) # automatically aligns to byte boundary and updates position

    def flush(self) -> bytes:
        """
        Flush stream contents, returning a bytes object.

        Returns
        -------
        bytes
            The stream contents.
        """
        self.align_to_byte()
        return self.stream[:self.position].tobytes()

    def close(self):
        """Write the stream to a file."""
        if self.path is not None:
            with open(self.path, mode = "wb") as f:
                f.write(self.flush())
        else:
            raise ValueError("No path provided.")

##################################################