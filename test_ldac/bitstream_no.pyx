# cython: language_level=3
# distutils: language = c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
# cython: nonecheck=False

# README
# Phillip Long
# July 19, 2025

# Helper objects for reading and writing bit streams.

# IMPORTS
##################################################

from libc.stdint cimport uint32_t, uint8_t, uint64_t
from libc.string cimport memcpy, memset
from libc.stdlib cimport malloc, free
from cpython.bytes cimport PyBytes_FromStringAndSize
from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cython.parallel cimport parallel, prange

##################################################


# CONSTANTS
##################################################

DEF BITS_PER_BYTE = 8
DEF DEFAULT_BUFFER_SIZE = (1024 ** 2)  # initialize bytes buffer to 1MB by default
DEF PARALLEL_THRESHOLD = 1024  # Minimum size for parallel operations

# Precompute bit masks for faster operations
cdef uint8_t[8] BIT_MASKS = [
    0b00000001,  # Least significant bit
    0b00000010,
    0b00000100,
    0b00001000,
    0b00010000,
    0b00100000,
    0b01000000,
    0b10000000   # Most significant bit
]

# Precompute lookup table for bit counting
cdef uint8_t[256] BIT_COUNT_TABLE
cdef uint8_t[256] REVERSE_BITS_TABLE

# Initialize lookup tables
cdef void init_lookup_tables() nogil:
    cdef int i, j, count
    for i in range(256):
        # Count bits
        count = 0
        for j in range(8):
            if i & (1 << j):
                count += 1
        BIT_COUNT_TABLE[i] = count
        
        # Reverse bits
        REVERSE_BITS_TABLE[i] = (
            ((i * 0x0802LU & 0x22110LU) | (i * 0x8020LU & 0x88440LU)) * 0x10101LU >> 16
        ) & 0xff

init_lookup_tables()

# SIMD operations for 64-bit parallel processing
cdef uint64_t reverse_bits_64(uint64_t x) nogil:
    cdef uint64_t r = x
    cdef int s = 32
    while s > 0:
        r = ((r >> s) & ((1 << s) - 1)) | ((r & ((1 << s) - 1)) << s)
        s >>= 1
    return r

##################################################


# BIT INPUT STREAM
##################################################

cdef class BitInputStream:
    """
    Stream object for reading in bits and bytes from a bytes stream.
    """
    cdef:
        public str path
        public bytes original_stream
        uint8_t* buffer  # Main buffer holding the data
        unsigned long buffer_size  # Size of buffer in bytes
        public unsigned long position  # Current bit position
        public bint is_byte_aligned
        uint64_t* buffer64  # 64-bit aligned buffer for SIMD operations

    def __init__(self, str path = None, bytes stream = None):
        """Initialize the bit input stream."""
        self.path = path
        self.original_stream = stream
        self.buffer = NULL
        self.buffer64 = NULL
        self.buffer_size = 0
        self.reset()

    def __dealloc__(self):
        """Clean up allocated memory."""
        if self.buffer is not NULL:
            free(self.buffer)

    cdef void _ensure_buffer_size(self, unsigned long size) nogil:
        """Ensure the buffer is large enough."""
        cdef:
            unsigned long new_size
            uint8_t* new_buffer
            
        if size > self.buffer_size:
            new_size = size
            new_buffer = <uint8_t*>malloc(new_size * sizeof(uint8_t))
            if new_buffer is NULL:
                with gil:
                    raise MemoryError("Failed to allocate buffer")
            if self.buffer is not NULL:
                memcpy(new_buffer, self.buffer, self.buffer_size)
                free(self.buffer)
            self.buffer = new_buffer
            self.buffer64 = <uint64_t*>self.buffer
            self.buffer_size = new_size

    cpdef unsigned long get_position(self):
        """Get the current bit position in the stream."""
        return self.position

    cdef uint8_t _get_byte_at(self, unsigned long byte_pos) nogil:
        """Get byte at given position."""
        if byte_pos >= self.buffer_size:
            return 0
        return self.buffer[byte_pos]

    def read_bit(self, *args, **kwargs):
        """Read a single bit."""
        if self.position >= self.buffer_size * BITS_PER_BYTE:
            raise RuntimeError("End of stream reached.")
        
        cdef unsigned long byte_pos = self.position // BITS_PER_BYTE
        cdef uint8_t bit_pos = self.position % BITS_PER_BYTE
        cdef bint result = (self._get_byte_at(byte_pos) & BIT_MASKS[bit_pos]) != 0
        
        self.position += 1
        self.is_byte_aligned = (self.position % BITS_PER_BYTE == 0)
        return result

    def read_bits(self, *args, **kwargs):
        """Read n bits as an unsigned integer."""
        cdef:
            unsigned long byte_pos
            uint32_t result = 0
            unsigned long i
            unsigned long n
            
        # Get n from either positional or keyword args
        if args:
            n = args[0]
        else:
            n = kwargs.get('n')
            
        if n > 32:
            raise ValueError("Cannot read more than 32 bits at once")
        if self.position + n > self.buffer_size * BITS_PER_BYTE:
            raise RuntimeError("End of stream reached.")

        # Fast path for byte-aligned reads
        if self.is_byte_aligned and n == 32:
            byte_pos = self.position // BITS_PER_BYTE
            memcpy(&result, &self.buffer[byte_pos], 4)
            self.position += 32
            return result

        # Fast path for byte-aligned reads of 8, 16, or 24 bits
        if self.is_byte_aligned and n % 8 == 0:
            byte_pos = self.position // BITS_PER_BYTE
            memcpy(&result, &self.buffer[byte_pos], n // 8)
            self.position += n
            return result

        # Slow path for unaligned reads
        for i in range(n):
            result = (result << 1) | self.read_bit()
        return result

    cpdef uint8_t read_byte(self) except *:
        """Read a single byte."""
        cdef unsigned long byte_pos
        if self.is_byte_aligned:
            byte_pos = self.position // BITS_PER_BYTE
            self.position += 8
            return self.buffer[byte_pos]
        return <uint8_t>self.read_bits(BITS_PER_BYTE)

    cpdef bytes read_bytes(self, unsigned long n):
        """Read n bytes."""
        if self.position % BITS_PER_BYTE != 0:
            raise ValueError("Must be byte-aligned to read bytes")
        if self.position + n * BITS_PER_BYTE > self.buffer_size * BITS_PER_BYTE:
            raise RuntimeError("End of stream reached.")

        cdef unsigned long start_byte = self.position // BITS_PER_BYTE
        cdef bytes result = PyBytes_FromStringAndSize(<char*>&self.buffer[start_byte], n)
        self.position += n * BITS_PER_BYTE
        return result

    cpdef uint32_t read_uint(self) except *:
        """Read an unsigned 32-bit integer."""
        return self.read_bits(32)

    cpdef void align_to_byte(self):
        """Align to the next byte boundary."""
        cdef unsigned long padding
        if not self.is_byte_aligned:
            padding = BITS_PER_BYTE - (self.position % BITS_PER_BYTE)
            self.position += padding
            self.is_byte_aligned = True

    cpdef void reset(self):
        """Reset the stream to its initial state."""
        if self.path is not None:
            with open(self.path, "rb") as f:
                content = f.read()
                self._ensure_buffer_size(len(content))
                memcpy(self.buffer, <char*>content, len(content))
                self.buffer_size = len(content)
        elif self.original_stream is not None:
            self._ensure_buffer_size(len(self.original_stream))
            memcpy(self.buffer, <char*>self.original_stream, len(self.original_stream))
            self.buffer_size = len(self.original_stream)
        else:
            raise ValueError("Either a path or a stream must be provided.")
        self.position = 0
        self.is_byte_aligned = True


# BIT OUTPUT STREAM
##################################################

cdef class BitOutputStream:
    """
    Stream object for writing bits and bytes to a bytes stream.
    """
    cdef:
        public str path
        uint8_t* buffer
        unsigned long buffer_size
        public unsigned long position
        public bint is_byte_aligned
        uint8_t current_byte
        uint8_t bits_in_current_byte
        uint64_t* buffer64  # 64-bit aligned buffer for SIMD operations

    def __init__(self, str path = None, unsigned long buffer_size = DEFAULT_BUFFER_SIZE):
        """Initialize the bit output stream."""
        self.path = path
        
        # Allocate buffer
        self.buffer = <uint8_t*>malloc(buffer_size * sizeof(uint8_t))
        if self.buffer is NULL:
            raise MemoryError("Failed to allocate buffer")
        
        self.buffer64 = <uint64_t*>self.buffer
        self.buffer_size = buffer_size
        memset(self.buffer, 0, buffer_size)
        self.position = 0
        self.is_byte_aligned = True
        self.current_byte = 0
        self.bits_in_current_byte = 0

    def __dealloc__(self):
        """Clean up allocated memory."""
        if self.buffer is not NULL:
            free(self.buffer)

    cdef void _ensure_buffer_size(self, unsigned long required_size) nogil:
        """Ensure the buffer is large enough."""
        cdef:
            unsigned long new_size
            uint8_t* new_buffer
            
        if required_size > self.buffer_size:
            new_size = max(required_size, self.buffer_size * 2)
            new_buffer = <uint8_t*>malloc(new_size * sizeof(uint8_t))
            if new_buffer is NULL:
                with gil:
                    raise MemoryError("Failed to allocate buffer")
            memcpy(new_buffer, self.buffer, self.buffer_size)
            memset(&new_buffer[self.buffer_size], 0, new_size - self.buffer_size)
            free(self.buffer)
            self.buffer = new_buffer
            self.buffer64 = <uint64_t*>self.buffer
            self.buffer_size = new_size

    cpdef unsigned long get_position(self):
        """Get current bit position."""
        return self.position

    cdef void _flush_current_byte(self) nogil:
        """Flush the current byte to the buffer if it contains any bits."""
        cdef unsigned long byte_pos
        
        if self.bits_in_current_byte > 0:
            byte_pos = self.position // BITS_PER_BYTE
            self._ensure_buffer_size(byte_pos + 1)
            self.buffer[byte_pos] = self.current_byte
            self.current_byte = 0
            self.bits_in_current_byte = 0

    def write_bit(self, *args, **kwargs):
        """Write a single bit."""
        cdef:
            bint bit
            unsigned long byte_pos = self.position // BITS_PER_BYTE
            uint8_t bit_pos = self.position % BITS_PER_BYTE
            
        # Get bit from either positional or keyword args
        if args:
            bit = args[0]
        else:
            bit = kwargs.get('bit')
        
        if bit_pos == 0:
            self._ensure_buffer_size(byte_pos + 1)
            self.buffer[byte_pos] = 0

        if bit:
            self.buffer[byte_pos] |= BIT_MASKS[bit_pos]
        else:
            self.buffer[byte_pos] &= ~BIT_MASKS[bit_pos]  # Optimized bit clear

        self.position += 1
        self.is_byte_aligned = (self.position % BITS_PER_BYTE == 0)

    def write_bits(self, *args, **kwargs):
        """Write n bits from an unsigned integer."""
        cdef:
            unsigned long byte_pos
            unsigned long i
            uint32_t mask
            uint32_t bits
            unsigned long n
            
        # Get bits and n from either positional or keyword args
        if len(args) == 2:
            bits = args[0]
            n = args[1]
        else:
            bits = kwargs.get('bits', 0)
            n = kwargs.get('n', 0)
            
        if n > 32:
            raise ValueError("Cannot write more than 32 bits at once")

        # Fast path for byte-aligned writes
        if self.is_byte_aligned and n == 32:
            byte_pos = self.position // BITS_PER_BYTE
            self._ensure_buffer_size(byte_pos + 4)
            memcpy(&self.buffer[byte_pos], &bits, 4)
            self.position += 32
            return

        # Fast path for byte-aligned writes of 8, 16, or 24 bits
        if self.is_byte_aligned and n % 8 == 0:
            byte_pos = self.position // BITS_PER_BYTE
            self._ensure_buffer_size(byte_pos + n // 8)
            memcpy(&self.buffer[byte_pos], &bits, n // 8)
            self.position += n
            return

        # Slow path for unaligned writes
        mask = 1 << (n - 1)
        for i in range(n):
            self.write_bit(bool(bits & mask))
            bits <<= 1

    cpdef void write_byte(self, uint8_t byte):
        """Write a single byte."""
        cdef unsigned long byte_pos
        if self.is_byte_aligned:
            byte_pos = self.position // BITS_PER_BYTE
            self._ensure_buffer_size(byte_pos + 1)
            self.buffer[byte_pos] = byte
            self.position += 8
            return
        self.write_bits(byte, BITS_PER_BYTE)

    cpdef void write_bytes(self, bytes bytes_):
        """Write a sequence of bytes."""
        cdef:
            unsigned long n = len(bytes_)
            unsigned long byte_pos = self.position // BITS_PER_BYTE
            unsigned long i
            uint64_t* src
            uint64_t* dst
            char* bytes_ptr = <char*>bytes_
            
        if self.position % BITS_PER_BYTE != 0:
            raise ValueError("Must be byte-aligned to write bytes")

        # Use SIMD operations for large writes
        if n >= PARALLEL_THRESHOLD:
            self._ensure_buffer_size(byte_pos + n)
            src = <uint64_t*>bytes_ptr
            dst = &self.buffer64[byte_pos // 8]
            
            with nogil:
                for i in prange(n // 8):
                    dst[i] = src[i]
                
                # Copy remaining bytes
                if n % 8:
                    memcpy(&self.buffer[byte_pos + (n // 8) * 8], 
                          bytes_ptr + (n // 8) * 8, 
                          n % 8)
        else:
            self._ensure_buffer_size(byte_pos + n)
            memcpy(&self.buffer[byte_pos], bytes_ptr, n)
        
        self.position += n * BITS_PER_BYTE

    def write_uint(self, *args, **kwargs):
        """Write an unsigned 32-bit integer."""
        cdef uint32_t value
        
        # Get value from either positional or keyword args
        if args:
            value = args[0]
        else:
            value = kwargs.get('value', 0)
            
        self.write_bits(value, 32)

    cpdef void align_to_byte(self):
        """Align to the next byte boundary by padding with zeros."""
        cdef unsigned long padding
        if not self.is_byte_aligned:
            padding = BITS_PER_BYTE - (self.position % BITS_PER_BYTE)
            self.write_bits(0, padding)

    cpdef bytes flush(self):
        """Flush stream contents to bytes."""
        self.align_to_byte()
        cdef unsigned long byte_length = (self.position + BITS_PER_BYTE - 1) // BITS_PER_BYTE
        return PyBytes_FromStringAndSize(<char*>self.buffer, byte_length)

    cpdef void close(self):
        """Write the stream to a file and close."""
        if self.path is not None:
            with open(self.path, "wb") as f:
                f.write(self.flush())
        else:
            raise ValueError("No path provided.")

################################################## 