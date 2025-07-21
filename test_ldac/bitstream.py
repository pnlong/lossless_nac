# README
# Phillip Long
# July 19, 2025

# Helper objects for reading and writing bit streams.

# IMPORTS
##################################################



##################################################


# BIT INPUT STREAM
##################################################

class BitInputStream:
    """
    Stream object for reading in bits and bytes from a bytes stream.
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
        if path is not None:
            with open(path, mode = "rb") as f:
                self.stream = f.read()
        elif stream is not None:
            self.stream = stream
        else:
            raise ValueError("Either a path or a stream must be provided.")
        self.stream_iter = iter(self.stream)
        self.bit_buffer = None # the contents of the current byte
        self.bit_buffer_position = 0 # takes on values 0-7 (the place in the current byte)
        self.is_byte_aligned = True # no bytes have been read yet

    def read_bit(self) -> bool:
        """
        Read a single bit.

        Returns
        -------
        bool
            The bit.
        """
        if self.bit_buffer_position == 0: # read the next byte if necessary
            _ = self.read_byte() # sets bit buffer to the current byte
        current_bit = self.bit_buffer >> (7 - self.bit_buffer_position) # get the correct bit from the bit buffer
        current_bit &= 1 # mask out all but the rightmost bit
        current_bit = bool(current_bit) # convert current bit to boolean
        self.bit_buffer_position = (self.bit_buffer_position + 1) % 8
        self.is_byte_aligned = (self.bit_buffer_position == 0)
        return current_bit
    
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
        value = 0
        for _ in range(n):
            value <<= 1
            value |= self.read_bit()
        return value

    def read_byte(self) -> int:
        """
        Read a single byte.

        Returns
        -------
        int
            The byte.
        """
        assert self.is_byte_aligned, "Please ensure that the cursor is aligned to a byte (call `align_to_byte`)!" # ensure byte alignment
        try:
            self.bit_buffer = next(self.stream_iter) # read in current byte
        except StopIteration:
            raise RuntimeError("End of stream reached.")
        return self.bit_buffer # return the current byte

    def read_uint(self) -> int:
        """
        Read an unsigned integer (4 bytes).

        Returns
        -------
        int
            The unsigned integer.
        """
        assert self.is_byte_aligned, "Please ensure that the cursor is aligned to a byte (call `align_to_byte`)!" # ensure byte alignment
        current_uint = 0
        for _ in range(4):
            current_uint <<= 8
            current_uint |= self.read_byte()
        return current_uint

    def read_int(self) -> int:
        """
        Read a signed integer (4 bytes).

        Returns
        -------
        int
            The signed integer.
        """
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

##################################################


# BIT OUTPUT STREAM
##################################################

class BitOutputStream:
    """
    Stream object for writing bits and bytes to a bytes stream.
    """

    def __init__(self, path: str = None):
        """
        Initialize the bit output stream.

        Parameters
        ----------
        path : str, optional
            The path to the file to write to. If None, calling the `write` method will do nothing.
        """
        self.path = path
        self.stream = []
        self.bit_buffer = 0 # buffer to accumulate bits
        self.bit_buffer_position = 0 # how many bits are currently in the buffer
        self.is_byte_aligned = True

    def write_bit(self, bit: bool):
        """
        Write a single bit.

        Parameters
        ----------
        bit : bool
            The bit to write.
        """
        if bit == True: # if the bit is 1
            self.bit_buffer |= (1 << (7 - self.bit_buffer_position)) # set the bit
        self.bit_buffer_position += 1
        self.is_byte_aligned = (self.bit_buffer_position % 8 == 0)
        if self.bit_buffer_position == 8:
            self.flush_byte()

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
        for i in range(n): # iterate over n bits
            bit = bits >> (n - i - 1) # shift relevant bit all the way to right
            bit &= 1 # mask out all but rightmost bit
            bit = bool(bit) # convert bit to boolean
            self.write_bit(bit = bit) # write the bit

    def write_byte(self, byte: int):
        """
        Write a single byte.

        Parameters
        ----------
        byte : int
            The byte to write.
        """
        assert self.is_byte_aligned, "Please ensure that the cursor is aligned to a byte (call `align_to_byte`)!" # ensure byte alignment
        self.stream.append(byte)

    def write_uint(self, value: int):
        """
        Write an unsigned integer (4 bytes).

        Parameters
        ----------
        value : int
            The unsigned integer to write.
        """
        assert self.is_byte_aligned, "Please ensure that the cursor is aligned to a byte (call `align_to_byte`)!" # ensure byte alignment
        for shift in (3, 2, 1, 0):
            self.write_byte((value >> (shift * 8)) & 0xFF)

    def write_int(self, value: int):
        """
        Write a signed integer (4 bytes).

        Parameters
        ----------
        value : int
            The signed integer to write.
        """
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
        """
        Flush stream contents, returning a bytes object.

        Returns
        -------
        bytes
            The stream contents.
        """
        self.align_to_byte()
        return bytes(self.stream)

    def close(self):
        """Write the stream to a file."""
        if self.path is not None:
            with open(self.path, mode = "wb") as f:
                f.write(self.flush())
        else:
            raise ValueError("No path provided.")

##################################################