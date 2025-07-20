# README
# Phillip Long
# July 16, 2025

# This script includes the command for running the mix_and_match.py script for all lossless compressors and entropy coders.
# It is used to evaluate the compression rate of the lossless compressors and entropy coders.

# PARAMETERS
##################################################

# entropy coders
EC_VERBATIM_PARAMETERS='{}'
EC_NAIVE_RICE_PARAMETERS='{"k": 12}'
EC_ADAPTIVE_RICE_PARAMETERS='{}'
EC_FLAC_RICE_PARAMETERS='{}'

# lossless compressors
LC_NAIVE_LPC_PARAMETERS='{"order": 16, "jobs": 10}'
LC_ADAPTIVE_LPC_PARAMETERS='{"jobs": 10}'
LC_FLAC_LPC_PARAMETERS='{"jobs": 10}'
LC_NAIVE_DAC_PARAMETERS='{"codebook_level": 4, "device": "cuda:0", "batch_size": 128}'
LC_ADAPTIVE_DAC_PARAMETERS='{"device": "cuda:3"}'

# block size
BLOCK_SIZE=4096

# number of samples
N_SAMPLES=10

# number of seconds
N_SECONDS=20

##################################################


# VERBATIM
##################################################

# naive lpc, verbatim
python "/home/pnlong/lnac/m&m/mix_and_match.py" --lossless_compressor "naive_lpc" --entropy_coder "verbatim" --lossless_compressor_parameters "${LC_NAIVE_LPC_PARAMETERS}" --entropy_coder_parameters "${EC_VERBATIM_PARAMETERS}" --block_size "${BLOCK_SIZE}" --n_samples "${N_SAMPLES}" --n_seconds "${N_SECONDS}" --mixes_only

# adaptive lpc, verbatim
python "/home/pnlong/lnac/m&m/mix_and_match.py" --lossless_compressor "adaptive_lpc" --entropy_coder "verbatim" --lossless_compressor_parameters "${LC_ADAPTIVE_LPC_PARAMETERS}" --entropy_coder_parameters "${EC_VERBATIM_PARAMETERS}" --block_size "${BLOCK_SIZE}" --n_samples "${N_SAMPLES}" --n_seconds "${N_SECONDS}" --mixes_only

# flac lpc, verbatim
python "/home/pnlong/lnac/m&m/mix_and_match.py" --lossless_compressor "flac_lpc" --entropy_coder "verbatim" --lossless_compressor_parameters "${LC_FLAC_LPC_PARAMETERS}" --entropy_coder_parameters "${EC_VERBATIM_PARAMETERS}" --block_size "${BLOCK_SIZE}" --n_samples "${N_SAMPLES}" --n_seconds "${N_SECONDS}" --mixes_only

# naive dac, verbatim
python "/home/pnlong/lnac/m&m/mix_and_match.py" --lossless_compressor "naive_dac" --entropy_coder "verbatim" --lossless_compressor_parameters "${LC_NAIVE_DAC_PARAMETERS}" --entropy_coder_parameters "${EC_VERBATIM_PARAMETERS}" --block_size "${BLOCK_SIZE}" --n_samples "${N_SAMPLES}" --n_seconds "${N_SECONDS}" --mixes_only

# adaptive dac, verbatim
python "/home/pnlong/lnac/m&m/mix_and_match.py" --lossless_compressor "adaptive_dac" --entropy_coder "verbatim" --lossless_compressor_parameters "${LC_ADAPTIVE_DAC_PARAMETERS}" --entropy_coder_parameters "${EC_VERBATIM_PARAMETERS}" --block_size "${BLOCK_SIZE}" --n_samples "${N_SAMPLES}" --n_seconds "${N_SECONDS}" --mixes_only

##################################################


# NAIVE RICE
##################################################

# naive lpc, naive rice
python "/home/pnlong/lnac/m&m/mix_and_match.py" --lossless_compressor "naive_lpc" --entropy_coder "naive_rice" --lossless_compressor_parameters "${LC_NAIVE_LPC_PARAMETERS}" --entropy_coder_parameters "${EC_NAIVE_RICE_PARAMETERS}" --block_size "${BLOCK_SIZE}" --n_samples "${N_SAMPLES}" --n_seconds "${N_SECONDS}" --mixes_only

# adaptive lpc, naive rice
python "/home/pnlong/lnac/m&m/mix_and_match.py" --lossless_compressor "adaptive_lpc" --entropy_coder "naive_rice" --lossless_compressor_parameters "${LC_ADAPTIVE_LPC_PARAMETERS}" --entropy_coder_parameters "${EC_NAIVE_RICE_PARAMETERS}" --block_size "${BLOCK_SIZE}" --n_samples "${N_SAMPLES}" --n_seconds "${N_SECONDS}" --mixes_only

# flac lpc, naive rice
python "/home/pnlong/lnac/m&m/mix_and_match.py" --lossless_compressor "flac_lpc" --entropy_coder "naive_rice" --lossless_compressor_parameters "${LC_FLAC_LPC_PARAMETERS}" --entropy_coder_parameters "${EC_NAIVE_RICE_PARAMETERS}" --block_size "${BLOCK_SIZE}" --n_samples "${N_SAMPLES}" --n_seconds "${N_SECONDS}" --mixes_only

# naive dac, naive rice
python "/home/pnlong/lnac/m&m/mix_and_match.py" --lossless_compressor "naive_dac" --entropy_coder "naive_rice" --lossless_compressor_parameters "${LC_NAIVE_DAC_PARAMETERS}" --entropy_coder_parameters "${EC_NAIVE_RICE_PARAMETERS}" --block_size "${BLOCK_SIZE}" --n_samples "${N_SAMPLES}" --n_seconds "${N_SECONDS}" --mixes_only

# adaptive dac, naive rice
python "/home/pnlong/lnac/m&m/mix_and_match.py" --lossless_compressor "adaptive_dac" --entropy_coder "naive_rice" --lossless_compressor_parameters "${LC_ADAPTIVE_DAC_PARAMETERS}" --entropy_coder_parameters "${EC_NAIVE_RICE_PARAMETERS}" --block_size "${BLOCK_SIZE}" --n_samples "${N_SAMPLES}" --n_seconds "${N_SECONDS}" --mixes_only

##################################################


# ADAPTIVE RICE
##################################################

# naive lpc, adaptive rice
python "/home/pnlong/lnac/m&m/mix_and_match.py" --lossless_compressor "naive_lpc" --entropy_coder "adaptive_rice" --lossless_compressor_parameters "${LC_NAIVE_LPC_PARAMETERS}" --entropy_coder_parameters "${EC_ADAPTIVE_RICE_PARAMETERS}" --block_size "${BLOCK_SIZE}" --n_samples "${N_SAMPLES}" --n_seconds "${N_SECONDS}" --mixes_only

# adaptive lpc, adaptive rice
python "/home/pnlong/lnac/m&m/mix_and_match.py" --lossless_compressor "adaptive_lpc" --entropy_coder "adaptive_rice" --lossless_compressor_parameters "${LC_ADAPTIVE_LPC_PARAMETERS}" --entropy_coder_parameters "${EC_ADAPTIVE_RICE_PARAMETERS}" --block_size "${BLOCK_SIZE}" --n_samples "${N_SAMPLES}" --n_seconds "${N_SECONDS}" --mixes_only

# flac lpc, adaptive rice
python "/home/pnlong/lnac/m&m/mix_and_match.py" --lossless_compressor "flac_lpc" --entropy_coder "adaptive_rice" --lossless_compressor_parameters "${LC_FLAC_LPC_PARAMETERS}" --entropy_coder_parameters "${EC_ADAPTIVE_RICE_PARAMETERS}" --block_size "${BLOCK_SIZE}" --n_samples "${N_SAMPLES}" --n_seconds "${N_SECONDS}" --mixes_only

# naive dac, adaptive rice
python "/home/pnlong/lnac/m&m/mix_and_match.py" --lossless_compressor "naive_dac" --entropy_coder "adaptive_rice" --lossless_compressor_parameters "${LC_NAIVE_DAC_PARAMETERS}" --entropy_coder_parameters "${EC_ADAPTIVE_RICE_PARAMETERS}" --block_size "${BLOCK_SIZE}" --n_samples "${N_SAMPLES}" --n_seconds "${N_SECONDS}" --mixes_only

# adaptive dac, adaptive rice
python "/home/pnlong/lnac/m&m/mix_and_match.py" --lossless_compressor "adaptive_dac" --entropy_coder "adaptive_rice" --lossless_compressor_parameters "${LC_ADAPTIVE_DAC_PARAMETERS}" --entropy_coder_parameters "${EC_ADAPTIVE_RICE_PARAMETERS}" --block_size "${BLOCK_SIZE}" --n_samples "${N_SAMPLES}" --n_seconds "${N_SECONDS}" --mixes_only

##################################################


# FLAC RICE
##################################################

# naive lpc, flac rice
python "/home/pnlong/lnac/m&m/mix_and_match.py" --lossless_compressor "naive_lpc" --entropy_coder "flac_rice" --lossless_compressor_parameters "${LC_NAIVE_LPC_PARAMETERS}" --entropy_coder_parameters "${EC_FLAC_RICE_PARAMETERS}" --block_size "${BLOCK_SIZE}" --n_samples "${N_SAMPLES}" --n_seconds "${N_SECONDS}" --mixes_only

# adaptive lpc, flac rice
python "/home/pnlong/lnac/m&m/mix_and_match.py" --lossless_compressor "adaptive_lpc" --entropy_coder "flac_rice" --lossless_compressor_parameters "${LC_ADAPTIVE_LPC_PARAMETERS}" --entropy_coder_parameters "${EC_FLAC_RICE_PARAMETERS}" --block_size "${BLOCK_SIZE}" --n_samples "${N_SAMPLES}" --n_seconds "${N_SECONDS}" --mixes_only

# flac lpc, flac rice
python "/home/pnlong/lnac/m&m/mix_and_match.py" --lossless_compressor "flac_lpc" --entropy_coder "flac_rice" --lossless_compressor_parameters "${LC_FLAC_LPC_PARAMETERS}" --entropy_coder_parameters "${EC_FLAC_RICE_PARAMETERS}" --block_size "${BLOCK_SIZE}" --n_samples "${N_SAMPLES}" --n_seconds "${N_SECONDS}" --mixes_only

# naive dac, flac rice
python "/home/pnlong/lnac/m&m/mix_and_match.py" --lossless_compressor "naive_dac" --entropy_coder "flac_rice" --lossless_compressor_parameters "${LC_NAIVE_DAC_PARAMETERS}" --entropy_coder_parameters "${EC_FLAC_RICE_PARAMETERS}" --block_size "${BLOCK_SIZE}" --n_samples "${N_SAMPLES}" --n_seconds "${N_SECONDS}" --mixes_only

# adaptive dac, flac rice
python "/home/pnlong/lnac/m&m/mix_and_match.py" --lossless_compressor "adaptive_dac" --entropy_coder "flac_rice" --lossless_compressor_parameters "${LC_ADAPTIVE_DAC_PARAMETERS}" --entropy_coder_parameters "${EC_FLAC_RICE_PARAMETERS}" --block_size "${BLOCK_SIZE}" --n_samples "${N_SAMPLES}" --n_seconds "${N_SECONDS}" --mixes_only

##################################################