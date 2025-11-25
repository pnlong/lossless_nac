"""Implements a lossless compressor with FLAC."""

from language_modeling_is_compression import constants_audio


if constants_audio.USE_PYDUB_FOR_FLAC:

  import pydub
  import audioop
  import io
  
  def compress(
      data: bytes,
      bit_depth: int = constants_audio.BIT_DEPTH,
      sample_rate: int = constants_audio.SAMPLE_RATE,
  ) -> bytes:
    """Returns data compressed with the FLAC codec.

    Args:
      data: Assumes 1 channel (mono or pseudo-mono) audio.
      bit_depth: Bit depth of the audio data (8, 16, or 24).
      sample_rate: Sample rate of the audio data (Hz).
    """
    sample = pydub.AudioSegment(
        data=data,
        channels=1, # assume mono or pseudo-mono audio
        sample_width=bit_depth // 8, # sample width (number of bytes per sample)
        frame_rate=sample_rate, # sample rate
    )
    return sample.export(
        format='flac',
        parameters=['-compression_level', '12'],
    ).read()

  def decompress(
      data: bytes,
      bit_depth: int = constants_audio.BIT_DEPTH,
      sample_rate: int = constants_audio.SAMPLE_RATE,
  ) -> bytes:
    """Decompresses `data` losslessly using the FLAC codec.

    Args:
      data: The data to be decompressed. Assumes 1 channel (mono or pseudo-mono) audio.
      bit_depth: Bit depth of the audio data (8, 16, or 24).
      sample_rate: Sample rate of the audio data (Hz).

    Returns:
      The decompressed data. Assumes 1 channel (mono or pseudo-mono) audio.
    """
    sample = pydub.AudioSegment.from_file(io.BytesIO(data), format='flac')
    new_width = bit_depth // 8
    bias = 2 ** (bit_depth - 1)
    return audioop.bias(audioop.lin2lin(sample.raw_data, 2, new_width), new_width, bias)

else:

  import numpy as np
  import subprocess

  # FLAC format map for ffmpeg, specifies the PCM format for ffmpeg to use.
  _FLAC_FMT_MAP = { # Note: 'u8' = unsigned 8-bit, 's16le' = signed 16-bit little endian, etc.
      8: 'u8', # does not need to specify endianness since it is a single byte
      16: 'u16' + ('le' if np.little_endian else 'be'),
      24: 'u24' + ('le' if np.little_endian else 'be'),
  }

  def compress(
      data: bytes,
      bit_depth: int = constants_audio.BIT_DEPTH,
      sample_rate: int = constants_audio.SAMPLE_RATE,
  ) -> bytes:
    """Compresses raw PCM audio bytes into FLAC format using ffmpeg.

    Args:
      data: Raw PCM audio bytes (mono).
      bit_depth: Bit depth of the audio data (8, 16, or 24).
      sample_rate: Sample rate of the audio data (Hz).

    Returns:
      FLAC-compressed audio data as bytes.
    """
    assert bit_depth in constants_audio.VALID_BIT_DEPTHS, f"Invalid bit depth: {bit_depth}. Valid bit depths are {constants_audio.VALID_BIT_DEPTHS}."
    process = subprocess.run(
      args=[
          'ffmpeg',
          '-f', _FLAC_FMT_MAP[bit_depth], # input PCM format, map bit depth to ffmpeg format string
          '-ar', str(sample_rate), # sample rate
          '-ac', '1', # mono channel
          '-i', 'pipe:0', # read input from stdin
          '-compression_level', '12', # max FLAC compression
          '-f', 'flac', # output format
          'pipe:1', # write FLAC bytes to stdout
          '-loglevel', 'error', # suppress logs unless errors occur
      ],
      input=data,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      check=True,
    ) # compressed data is written to stdout as bytes
    return process.stdout


  def decompress(
      data: bytes,
      bit_depth: int = constants_audio.BIT_DEPTH,
      sample_rate: int = constants_audio.SAMPLE_RATE,
  ) -> bytes:
    """Decompresses `data` losslessly using the FLAC codec.

    Args:
      data: The data to be decompressed. Assumes 1 channel (mono or pseudo-mono) audio.
      bit_depth: Bit depth of the audio data (8, 16, or 24).
      sample_rate: Sample rate of the audio data (Hz).

    Returns:
      The decompressed data. Assumes 1 channel (mono or pseudo-mono) audio.
    """
    assert bit_depth in constants_audio.VALID_BIT_DEPTHS, f"Invalid bit depth: {bit_depth}. Valid bit depths are {constants_audio.VALID_BIT_DEPTHS}."
    process = subprocess.run(
      args=[
          'ffmpeg',
          '-i', 'pipe:0', # read FLAC from stdin
          '-f', _FLAC_FMT_MAP[bit_depth], # output raw PCM in desired format, map bit depth to ffmpeg format string
          '-ac', '1', # mono channel
          '-ar', str(sample_rate), # sample rate
          '-map', '0:a:0', # take first audio stream
          'pipe:1', # write raw PCM to stdout
          '-loglevel', 'error',
      ],
      input=data,
      stdout=subprocess.PIPE,
      stderr=subprocess.PIPE,
      check=True,
    ) # decompressed data is written to stdout as bytes
    return process.stdout
