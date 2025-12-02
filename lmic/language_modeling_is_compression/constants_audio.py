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
LLAMA_USE_TOP_K = False
TOP_K = 100 # top-k next token log-probabilities
QUANTIZE_LLAMA_MODEL = False
POST_TOKENIZATION_LENGTH_BYTES = 4 # number of bytes to store the post tokenization length
POST_TOKENIZATION_LENGTH_ENDIANNESS = 'little' # endianness of the post tokenization length

# print loss and bits per byte for each chunk with Llama
OUTPUT_LOSS_AND_BPB_TO_VERIFY_COMPRESSION = True

# whether to use pydub for FLAC compression
USE_PYDUB_FOR_FLAC = False # pydub doesn't support variable bit depth

# whether to use slow lossless compression for evals
USE_SLOW_LOSSLESS_COMPRESSION_FOR_EVALS = False

# use tqdm for progress bars
USE_TQDM = True # enabled because we are writing to log files

# output filepath for evaluation results
EVAL_OUTPUT_FILEPATH = "/home/pnlong/lnac/lmic/lmic_eval_results.csv"
LOSS_BPB_OUTPUT_FILEPATH = "/home/pnlong/lnac/lmic/lmic_eval_loss_bpb_results.csv"

# filepaths (general)
AUDIO_DATA_DIR = "/graft3/datasets/pnlong/lnac/sashimi/data"

# MUSDB18 Mono
MUSDB18MONO_DATA_DIR = f"{AUDIO_DATA_DIR}/musdb18mono" # yggdrasil

# MUSDB18 Stereo
MUSDB18STEREO_DATA_DIR = f"{AUDIO_DATA_DIR}/musdb18stereo" # yggdrasil

# LibriSpeech
LIBRISPEECH_SPLIT = "dev-clean" # "dev-clean" or "train-clean-100"
LIBRISPEECH_DATA_DIR = f"{AUDIO_DATA_DIR}/librispeech/LibriSpeech/{LIBRISPEECH_SPLIT}" # yggdrasil

# LJSpeech
LJSPEECH_DATA_DIR = f"{AUDIO_DATA_DIR}/ljspeech" # yggdrasil

# Epidemic Sound
EPIDEMIC_SOUND_DATA_DIR = "/graft1/datasets/kechen/epidemic/epidemic_sound" # pando

# VCTK (speech)
VCTK_DATA_DIR = "/graft2/datasets/znovack/VCTK-Corpus-0.92/wav48_silence_trimmed" # pando

# Torrent Data 16-bit
TORRENT_DATA_DATA_DIR = "/graft3/datasets/znovack/trilobyte" # yggdrasil

# Birdvox bioacoustic data
BIRDVOX_DATA_DIR = "/mnt/arrakis_data/pnlong/lnac/birdvox/unit06" # yggdrasil

# Beethoven Piano Sonatas
BEETHOVEN_DATA_DIR = "/mnt/arrakis_data/pnlong/lnac/beethoven" # yggdrasil

# YouTubeMix Audio Dataset
YOUTUBE_MIX_DATA_DIR = "/mnt/arrakis_data/pnlong/lnac/youtube_mix" # yggdrasil

# SC09 Speech Dataset
SC09_DATA_DIR = "/mnt/arrakis_data/pnlong/lnac/sc09" # yggdrasil