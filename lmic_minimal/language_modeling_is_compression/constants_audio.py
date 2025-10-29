"""Defines project-wide constants for audio."""

# default bit depth
BIT_DEPTH = 16
VALID_BIT_DEPTHS = {8, 16, 24}

# default sample rate
SAMPLE_RATE = 44100

# default llama model
LLAMA_MODEL = 'llama-2-7b'
VALID_LLAMA_MODELS = {'llama-2-7b', 'llama-2-13b', 'llama-2-70b'}
TOP_K = 100 # top-k next token log-probabilities
QUANTIZE_LLAMA_MODEL = False

# whether to merge LMIC's data generators with our custom ones
MERGE_LMIC_DATA_GENERATOR_FN_DICT = False

# whether to use pydub for FLAC compression
USE_PYDUB_FOR_FLAC = False

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
