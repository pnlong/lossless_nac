# README
# Phillip Long
# May 11, 2025

# Implementation of Free Lossless Audio Codec (FLAC) for use as a baseline.
# Based off the code provided by Project Nayuki (https://www.nayuki.io/page/simple-flac-implementation).

# IMPORTS
##################################################

import pathlib
import sys
import struct

from os.path import dirname, realpath
import sys
sys.path.insert(0, dirname(realpath(__file__)))
sys.path.insert(0, dirname(dirname(realpath(__file__))))

import utils

##################################################


# CONSTANTS
##################################################



##################################################


# ENCODE
##################################################

def encode(path_input: str, path_output: str):
    """
    Encode a file with the FLAC format.
    
    Parameters
    ----------
    path_input : str
        Absolute filepath to the WAV file that will be encoded.
    path_output : str
        Absolute filepath where the encoded file will be outputted.
    """
    
    # open input path as a file object
    with open(path_input, "rb") as inp:
        
        
    

def main(argv):
    if len(argv) != 3:
        sys.exit(f"Usage: python {argv[0]} InFile.wav OutFile.flac")
    with pathlib.Path(argv[1]).open("rb") as inp:
        with BitOutputStream(pathlib.Path(argv[2]).open("wb")) as out:
            encode_file(inp, out)

def encode_file(inp, out):
    # Read and parse WAV file headers
    def fail_if(cond, msg):
        if cond:
            raise ValueError(msg)
    fail_if(read_fully(inp, 4) != b"RIFF", "Invalid RIFF file header")
    read_little_int(inp, 4)
    fail_if(read_fully(inp, 4) != b"WAVE", "Invalid WAV file header")
    fail_if(read_fully(inp, 4) != b"fmt ", "Unrecognized WAV file chunk")
    fail_if(read_little_int(inp, 4) != 16, "Unsupported WAV file type")
    fail_if(read_little_int(inp, 2) != 0x0001, "Unsupported WAV file codec")
    numchannels = read_little_int(inp, 2)
    fail_if(not (1 <= numchannels <= 8), "Too many (or few) audio channels")
    samplerate = read_little_int(inp, 4)
    fail_if(not (1 <= samplerate < (1 << 20)), "Sample rate too large or invalid")
    read_little_int(inp, 4)
    read_little_int(inp, 2)
    sampledepth = read_little_int(inp, 2)
    fail_if(sampledepth not in (8,16,24,32), "Unsupported sample depth")
    fail_if(read_fully(inp, 4) != b"data", "Unrecognized WAV file chunk")
    sampledatalen = read_little_int(inp, 4)
    fail_if(sampledatalen <= 0 or sampledatalen % (numchannels * (sampledepth // 8)) != 0, "Invalid length of audio sample data")
    
    # Start writing FLAC file header and stream info metadata block
    out.write_int(32, 0x664C6143)
    out.write_int(1, 1)
    out.write_int(7, 0)
    out.write_int(24, 34)
    out.write_int(16, BLOCK_SIZE)
    out.write_int(16, BLOCK_SIZE)
    out.write_int(24, 0)
    out.write_int(24, 0)
    out.write_int(20, samplerate)
    out.write_int(3, numchannels - 1)
    out.write_int(5, sampledepth - 1)
    numsamples = sampledatalen // (numchannels * (sampledepth // 8))
    out.write_int(36, numsamples)
    for _ in range(16):
        out.write_int(8, 0)
    
    # Read raw samples and encode FLAC audio frames
    i = 0
    while numsamples > 0:
        blocksize = min(numsamples, BLOCK_SIZE)
        encode_frame(inp, i, numchannels, sampledepth, samplerate, blocksize, out)
        numsamples -= blocksize
        i += 1

BLOCK_SIZE = 4096


def read_fully(inp, n):
    result = inp.read(n)
    if len(result) < n:
        raise EOFError()
    return result


def read_little_int(inp, n):
    result = 0
    for (i, b) in enumerate(read_fully(inp, n)):
        result |= b << (i * 8)
    return result


def encode_frame(inp, frameindex, numchannels, sampledepth, samplerate, blocksize, out):
    bytespersample = sampledepth // 8
    samples = [[] for _ in range(numchannels)]
    for _ in range(blocksize):
        for chansamples in samples:
            val = read_little_int(inp, bytespersample)
            if sampledepth == 8:
                val -= 128
            else:
                val -= (val >> (sampledepth - 1)) << sampledepth
            chansamples.append(val)
    
    out.reset_crcs()
    out.write_int(14, 0x3FFE)
    out.write_int(1, 0)
    out.write_int(1, 0)
    out.write_int(4, 7)
    out.write_int(4, (14 if samplerate % 10 == 0 else 13))
    out.write_int(4, numchannels - 1)
    out.write_int(3, {8:1, 16:4, 24:6, 32:0}[sampledepth])
    out.write_int(1, 0)
    out.write_int(8, 0xFC | (frameindex >> 30))
    for i in range(24, -1, -6):
        out.write_int(8, 0x80 | ((frameindex >> i) & 0x3F))
    out.write_int(16, blocksize - 1)
    out.write_int(16, samplerate // (10 if samplerate % 10 == 0 else 1))
    out.write_int(8, out.crc8)
    
    for chansamples in samples:
        encode_subframe(chansamples, sampledepth, out)
    out.align_to_byte()
    out.write_int(16, out.crc16)


def encode_subframe(samples, sampledepth, out):
    out.write_int(1, 0)
    out.write_int(6, 1)  # Verbatim coding
    out.write_int(1, 0)
    for x in samples:
        out.write_int(sampledepth, x)

##################################################


# DECODE
##################################################

def main(argv):
    if len(argv) != 3:
        sys.exit(f"Usage: python {argv[0]} InFile.flac OutFile.wav")
    with BitInputStream(pathlib.Path(argv[1]).open("rb")) as inp:
        with pathlib.Path(argv[2]).open("wb") as out:
            decode_file(inp, out)


def decode_file(inp, out):
    # Handle FLAC header and metadata blocks
    if inp.read_uint(32) != 0x664C6143:
        raise ValueError("Invalid magic string")
    samplerate = None
    last = False
    while not last:
        last = inp.read_uint(1) != 0
        type = inp.read_uint(7)
        length = inp.read_uint(24)
        if type == 0:  # Stream info block
            inp.read_uint(16)
            inp.read_uint(16)
            inp.read_uint(24)
            inp.read_uint(24)
            samplerate = inp.read_uint(20)
            numchannels = inp.read_uint(3) + 1
            sampledepth = inp.read_uint(5) + 1
            numsamples = inp.read_uint(36)
            inp.read_uint(128)
        else:
            for i in range(length):
                inp.read_uint(8)
    if samplerate is None:
        raise ValueError("Stream info metadata block absent")
    if sampledepth % 8 != 0:
        raise RuntimeError("Sample depth not supported")
    
    # Start writing WAV file headers
    sampledatalen = numsamples * numchannels * (sampledepth // 8)
    out.write(b"RIFF")
    out.write(struct.pack("<I", sampledatalen + 36))
    out.write(b"WAVE")
    out.write(b"fmt ")
    out.write(struct.pack("<IHHIIHH", 16, 0x0001, numchannels, samplerate,
        samplerate * numchannels * (sampledepth // 8), numchannels * (sampledepth // 8), sampledepth))
    out.write(b"data")
    out.write(struct.pack("<I", sampledatalen))
    
    # Decode FLAC audio frames and write raw samples
    while decode_frame(inp, numchannels, sampledepth, out):
        pass


def decode_frame(inp, numchannels, sampledepth, out):
    # Read a ton of header fields, and ignore most of them
    temp = inp.read_byte()
    if temp == -1:
        return False
    sync = temp << 6 | inp.read_uint(6)
    if sync != 0x3FFE:
        raise ValueError("Sync code expected")
    
    inp.read_uint(1)
    inp.read_uint(1)
    blocksizecode = inp.read_uint(4)
    sampleratecode = inp.read_uint(4)
    chanasgn = inp.read_uint(4)
    inp.read_uint(3)
    inp.read_uint(1)
    
    temp = inp.read_uint(8)
    while temp >= 0b11000000:
        inp.read_uint(8)
        temp = (temp << 1) & 0xFF
    
    if blocksizecode == 1:
        blocksize = 192
    elif 2 <= blocksizecode <= 5:
        blocksize = 576 << blocksizecode - 2
    elif blocksizecode == 6:
        blocksize = inp.read_uint(8) + 1
    elif blocksizecode == 7:
        blocksize = inp.read_uint(16) + 1
    elif 8 <= blocksizecode <= 15:
        blocksize = 256 << (blocksizecode - 8)
    
    if sampleratecode == 12:
        inp.read_uint(8)
    elif sampleratecode in (13, 14):
        inp.read_uint(16)
    
    inp.read_uint(8)
    
    # Decode each channel's subframe, then skip footer
    samples = decode_subframes(inp, blocksize, sampledepth, chanasgn)
    inp.align_to_byte()
    inp.read_uint(16)
    
    # Write the decoded samples
    numbytes = sampledepth // 8
    addend = 128 if sampledepth == 8 else 0
    for i in range(blocksize):
        for j in range(numchannels):
            out.write(struct.pack("<i", samples[j][i] + addend)[ : numbytes])
    return True


def decode_subframes(inp, blocksize, sampledepth, chanasgn):
    if 0 <= chanasgn <= 7:
        return [decode_subframe(inp, blocksize, sampledepth) for _ in range(chanasgn + 1)]
    elif 8 <= chanasgn <= 10:
        temp0 = decode_subframe(inp, blocksize, sampledepth + (1 if (chanasgn == 9) else 0))
        temp1 = decode_subframe(inp, blocksize, sampledepth + (0 if (chanasgn == 9) else 1))
        if chanasgn == 8:
            for i in range(blocksize):
                temp1[i] = temp0[i] - temp1[i]
        elif chanasgn == 9:
            for i in range(blocksize):
                temp0[i] += temp1[i]
        elif chanasgn == 10:
            for i in range(blocksize):
                side = temp1[i]
                right = temp0[i] - (side >> 1)
                temp1[i] = right
                temp0[i] = right + side
        return [temp0, temp1]
    else:
        raise ValueError("Reserved channel assignment")


def decode_subframe(inp, blocksize, sampledepth):
    inp.read_uint(1)
    type = inp.read_uint(6)
    shift = inp.read_uint(1)
    if shift == 1:
        while inp.read_uint(1) == 0:
            shift += 1
    sampledepth -= shift
    
    if type == 0:  # Constant coding
        result = [inp.read_signed_int(sampledepth)] * blocksize
    elif type == 1:  # Verbatim coding
        result = [inp.read_signed_int(sampledepth) for _ in range(blocksize)]
    elif 8 <= type <= 12:
        result = decode_fixed_prediction_subframe(inp, type - 8, blocksize, sampledepth)
    elif 32 <= type <= 63:
        result = decode_linear_predictive_coding_subframe(inp, type - 31, blocksize, sampledepth)
    else:
        raise ValueError("Reserved subframe type")
    return [(v << shift) for v in result]


def decode_fixed_prediction_subframe(inp, predorder, blocksize, sampledepth):
    result = [inp.read_signed_int(sampledepth) for _ in range(predorder)]
    decode_residuals(inp, blocksize, result)
    restore_linear_prediction(result, FIXED_PREDICTION_COEFFICIENTS[predorder], 0)
    return result

FIXED_PREDICTION_COEFFICIENTS = (
    (),
    (1,),
    (2, -1),
    (3, -3, 1),
    (4, -6, 4, -1),
)


def decode_linear_predictive_coding_subframe(inp, lpcorder, blocksize, sampledepth):
    result = [inp.read_signed_int(sampledepth) for _ in range(lpcorder)]
    precision = inp.read_uint(4) + 1
    shift = inp.read_signed_int(5)
    coefs = [inp.read_signed_int(precision) for _ in range(lpcorder)]
    decode_residuals(inp, blocksize, result)
    restore_linear_prediction(result, coefs, shift)
    return result


def decode_residuals(inp, blocksize, result):
    method = inp.read_uint(2)
    if method >= 2:
        raise ValueError("Reserved residual coding method")
    parambits = [4, 5][method]
    escapeparam = [0xF, 0x1F][method]
    
    partitionorder = inp.read_uint(4)
    numpartitions = 1 << partitionorder
    if blocksize % numpartitions != 0:
        raise ValueError("Block size not divisible by number of Rice partitions")
    
    for i in range(numpartitions):
        count = blocksize >> partitionorder
        if i == 0:
            count -= len(result)
        param = inp.read_uint(parambits)
        if param < escapeparam:
            result.extend(inp.read_rice_signed_int(param) for _ in range(count))
        else:
            numbits = inp.read_uint(5)
            result.extend(inp.read_signed_int(numbits) for _ in range(count))


def restore_linear_prediction(result, coefs, shift):
    for i in range(len(coefs), len(result)):
        result[i] += sum((result[i - 1 - j] * c) for (j, c) in enumerate(coefs)) >> shift

##################################################


# MAIN METHOD
##################################################

if __name__ == "__main__":
    
    pass

##################################################