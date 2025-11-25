#!/bin/bash
# bash lmic_eval.sh

# Default values
SOFTWARE="/home/pnlong/lnac/lmic/language_modeling_is_compression/compress_audio.py" # software path
OUTPUT="/home/pnlong/lnac/lmic/lmic_eval_results.csv" # output filepath
COMPRESSOR="llama-2-7b"
CHUNK_SIZE=2048
NUM_CHUNKS=1000
BIT_DEPTH=""
IS_MU_LAW=""

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --compressor COMPRESSOR    Compressor to use (default: ${COMPRESSOR})
    --chunk_size CHUNK_SIZE    Chunk size (number of bytes), default: ${CHUNK_SIZE}
    --num_chunks NUM_CHUNKS    Number of chunks, default: ${NUM_CHUNKS}
    --bit_depth BIT_DEPTH      Bit depth (8, 16, or 24), if not provided, the bit depth is determined by the dataset
    --is_mu_law IS_MU_LAW      Whether to use mu-law encoding, if not provided, the is_mu_law is determined by the dataset
    -h, --help                 Show this help message
EOF
}

# Parse command line arguments using getopt
OPTS=$(getopt -o "h" --long compressor:,chunk_size:,num_chunks:,bit_depth:,is_mu_law:,help -- "$@")
if [ $? -ne 0 ]; then
    echo "Error: Failed to parse options"
    usage
    exit 1
fi

eval set -- "$OPTS"

while true; do
    case "$1" in
        --compressor)
            COMPRESSOR="$2"
            shift 2
            ;;
        --chunk_size)
            CHUNK_SIZE="$2"
            shift 2
            ;;
        --num_chunks)
            NUM_CHUNKS="$2"
            shift 2
            ;;
        --bit_depth)
            BIT_DEPTH="$2"
            shift 2
            ;;
        --is_mu_law)
            IS_MU_LAW="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate compressor
if ! [[ "$COMPRESSOR" =~ ^(llama-2-7b|llama-2-13b|llama-2-70b)$ ]]; then
    echo "Error: --compressor must be one of llama-2-7b, llama-2-13b, or llama-2-70b"
    exit 1
fi

# Validate chunk size
if ! [[ "$CHUNK_SIZE" =~ ^[0-9]+$ ]]; then
    echo "Error: --chunk_size must be a positive integer"
    exit 1
fi

# Validate number of chunks
if ! [[ "$NUM_CHUNKS" =~ ^[0-9]+$ ]]; then
    echo "Error: --num_chunks must be a positive integer"
    exit 1
fi

# Build common arguments
common_args=(
    "--output_filepath" "${OUTPUT}"
    "--compressor" "${COMPRESSOR}"
    "--chunk_size" "${CHUNK_SIZE}"
    "--num_chunks" "${NUM_CHUNKS}"
)
[[ -n "$BIT_DEPTH" ]] && common_args+=("--bit_depth" "${BIT_DEPTH}")
[[ -n "$IS_MU_LAW" ]] && common_args+=("--is_mu_law" "${IS_MU_LAW}")

# musdb18mono and musdb18stereo
for subset in "_mixes" "_stems" ""; do
    python "${SOFTWARE}" --dataset "musdb18mono${subset}" "${common_args[@]}"
    python "${SOFTWARE}" --dataset "musdb18stereo${subset}" "${common_args[@]}"
done

# librispeech
python "${SOFTWARE}" --dataset "librispeech" "${common_args[@]}"

# ljspeech
python "${SOFTWARE}" --dataset "ljspeech" "${common_args[@]}"

# epidemic
python "${SOFTWARE}" --dataset "epidemic" "${common_args[@]}"

# vctk
python "${SOFTWARE}" --dataset "vctk" "${common_args[@]}"

# torrent 16-bit and 24-bit
for torrent_subset in "_pro" "_amateur" "_freeload" ""; do
    python "${SOFTWARE}" --dataset "torrent16b${torrent_subset}" "${common_args[@]}"
    python "${SOFTWARE}" --dataset "torrent24b${torrent_subset}" "${common_args[@]}"
done

# birdvox
python "${SOFTWARE}" --dataset "birdvox" "${common_args[@]}"

# beethoven piano sonatas
python "${SOFTWARE}" --dataset "beethoven" "${common_args[@]}"

# youtube mix
python "${SOFTWARE}" --dataset "youtube_mix" "${common_args[@]}"

# sc09 speech
python "${SOFTWARE}" --dataset "sc09" "${common_args[@]}"