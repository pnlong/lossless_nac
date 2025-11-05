"""Defines project-wide constants for audio."""

# default bit depth
BIT_DEPTH = 16
VALID_BIT_DEPTHS = {8, 16, 24}

# default sample rate
SAMPLE_RATE = 44100

# for chunking audio data
RANDOMIZE_CHUNKS = True
CHUNKS_PER_SAMPLE = 10

# default llama model
DEFAULT_LLAMA_MODEL = 'llama-2-7b'
VALID_LLAMA_MODELS = ['llama-2-7b', 'llama-2-13b', 'llama-2-70b']
LLAMA_USE_TOP_K = True
TOP_K = 100 # top-k next token log-probabilities
QUANTIZE_LLAMA_MODEL = False
POST_TOKENIZATION_LENGTH_BYTES = 4 # number of bytes to store the post tokenization length
POST_TOKENIZATION_LENGTH_ENDIANNESS = 'little' # endianness of the post tokenization length

# whether to merge LMIC's data generators with our custom ones
MERGE_LMIC_DATA_GENERATOR_FN_DICT = False

# whether to use pydub for FLAC compression
USE_PYDUB_FOR_FLAC = False # pydub doesn't support variable bit depth

# whether to use slow lossless compression for evals
USE_SLOW_LOSSLESS_COMPRESSION_FOR_EVALS = False

# use tqdm for progress bars
USE_TQDM = True # enabled because we are writing to log files

# filepaths (general)
AUDIO_DATA_DIR = "/graft3/datasets/pnlong/lnac/sashimi/data"

# MUSDB18 Mono
MUSDB18MONO_DATA_DIR = f"{AUDIO_DATA_DIR}/musdb18mono"
MUSDB18MONO_MIXES_ONLY = False
MUSDB18MONO_PARTITION = "all" # "train" or "valid"

# MUSDB18 Stereo
MUSDB18STEREO_DATA_DIR = f"{AUDIO_DATA_DIR}/musdb18stereo"
MUSDB18STEREO_MIXES_ONLY = False
MUSDB18STEREO_PARTITION = "all" # "train" or "valid"
