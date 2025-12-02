#!/bin/bash
# bash lmic_eval.sh

set -e

# Default values
SOFTWARE="/home/pnlong/lnac/lmic/language_modeling_is_compression/compress_audio.py" # software path
OUTPUT="/home/pnlong/lnac/lmic/lmic_eval_results.csv" # output filepath
LOSS_BPB_OUTPUT="/home/pnlong/lnac/lmic/lmic_eval_loss_bpb_results.csv" # output filepath for loss and bits per byte data
COMPRESSOR="llama-2-7b" # compressor to use
CHUNK_SIZE=2048 # chunk size to use, in bytes
NUM_CHUNKS=1000 # number of chunks to compress
BIT_DEPTH="" # bit depth to use, if not provided, the bit depth is determined by the dataset
IS_MU_LAW="" # whether to use mu-law encoding, if not provided, the is_mu_law is determined by the dataset
USE_SLOW_LOSSLESS_COMPRESSION="" # whether to use slow lossless compression
MACHINE="yggdrasil" # machine to use (yggdrasil or pando)
BATCH=0 # batch number (0-3 for yggdrasil, 0 for pando)

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    --compressor COMPRESSOR             Compressor to use (default: ${COMPRESSOR})
    --chunk_size CHUNK_SIZE             Chunk size (number of bytes), default: ${CHUNK_SIZE}
    --num_chunks NUM_CHUNKS             Number of chunks, default: ${NUM_CHUNKS}
    --bit_depth BIT_DEPTH               Bit depth (8, 16, or 24), if not provided, the bit depth is determined by the dataset
    --is_mu_law IS_MU_LAW               Whether to use mu-law encoding, if not provided, the is_mu_law is determined by the dataset
    --use_slow_lossless_compression     Whether to use slow lossless compression
    --machine MACHINE                   Machine to use (yggdrasil or pando, default: ${MACHINE})
    --batch BATCH                       Batch number (0-3 for yggdrasil, 0 for pando, default: ${BATCH})
    -h, --help                          Show this help message
EOF
}

# Parse command line arguments using getopt
OPTS=$(getopt -o "h" --long compressor:,chunk_size:,num_chunks:,bit_depth:,is_mu_law:,use_slow_lossless_compression,machine:,batch:,help -- "$@")
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
        --use_slow_lossless_compression)
            USE_SLOW_LOSSLESS_COMPRESSION="1"
            shift
            ;;
        --machine)
            MACHINE="$2"
            shift 2
            ;;
        --batch)
            BATCH="$2"
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

# Validate machine
if ! [[ "$MACHINE" =~ ^(yggdrasil|pando)$ ]]; then
    echo "Error: Unknown machine: ${MACHINE}. Must be 'yggdrasil' or 'pando'"
    exit 1
fi

# Validate batch
if ! [[ "$BATCH" =~ ^[0-9]+$ ]]; then
    echo "Error: --batch must be a non-negative integer"
    exit 1
fi

# Build common arguments
common_args=(
    "--output_filepath" "${OUTPUT}"
    "--loss_bpb_output_filepath" "${LOSS_BPB_OUTPUT}"
    "--compressor" "${COMPRESSOR}"
    "--chunk_size" "${CHUNK_SIZE}"
    "--num_chunks" "${NUM_CHUNKS}"
)
[[ -n "$BIT_DEPTH" ]] && common_args+=("--bit_depth" "${BIT_DEPTH}")
[[ -n "$IS_MU_LAW" ]] && common_args+=("--is_mu_law" "${IS_MU_LAW}")
[[ -n "$USE_SLOW_LOSSLESS_COMPRESSION" ]] && common_args+=("--use_slow_lossless_compression")

# Run datasets based on machine and batch

# pando datasets
if [ "$MACHINE" == "pando" ]; then

    # batch 0
    if [ "$BATCH" == "0" ]; then
        # epidemic sound
        python "${SOFTWARE}" --dataset "epidemic" "${common_args[@]}"

        # vctk
        python "${SOFTWARE}" --dataset "vctk" "${common_args[@]}"

    # further batches are not supported for pando
    else
        echo "Error: Invalid batch index ${BATCH} for machine: ${MACHINE}. Valid batch indices are: 0"
        exit 1
    fi

# yggdrasil datasets
elif [ "$MACHINE" == "yggdrasil" ]; then

    # batch 0
    if [ "$BATCH" == "0" ]; then

        # musdb18mono
        python "${SOFTWARE}" --dataset "musdb18mono_mixes" "${common_args[@]}"
        python "${SOFTWARE}" --dataset "musdb18mono_stems" "${common_args[@]}"
        python "${SOFTWARE}" --dataset "musdb18mono" "${common_args[@]}"

        # musdb18stereo
        python "${SOFTWARE}" --dataset "musdb18stereo_mixes" "${common_args[@]}"
        python "${SOFTWARE}" --dataset "musdb18stereo_stems" "${common_args[@]}"
    
    # batch 1
    elif [ "$BATCH" == "1" ]; then

        # musdb18stereo (continued)
        python "${SOFTWARE}" --dataset "musdb18stereo" "${common_args[@]}"

        # librispeech
        python "${SOFTWARE}" --dataset "librispeech" "${common_args[@]}"

        # ljspeech
        python "${SOFTWARE}" --dataset "ljspeech" "${common_args[@]}"

        # torrent 16-bit
        python "${SOFTWARE}" --dataset "torrent16b_pro" "${common_args[@]}"
        python "${SOFTWARE}" --dataset "torrent16b_amateur" "${common_args[@]}"
    
    # batch 2
    elif [ "$BATCH" == "2" ]; then

        # torrent 16-bit (continued)
        python "${SOFTWARE}" --dataset "torrent16b_freeload" "${common_args[@]}"
        python "${SOFTWARE}" --dataset "torrent16b" "${common_args[@]}"
        
        # torrent 24-bit
        python "${SOFTWARE}" --dataset "torrent24b_pro" "${common_args[@]}"
        python "${SOFTWARE}" --dataset "torrent24b_amateur" "${common_args[@]}"
        python "${SOFTWARE}" --dataset "torrent24b_freeload" "${common_args[@]}"
    
    # batch 3
    elif [ "$BATCH" == "3" ]; then

        # torrent 24-bit (continued)
        python "${SOFTWARE}" --dataset "torrent24b" "${common_args[@]}"

        # birdvox
        python "${SOFTWARE}" --dataset "birdvox" "${common_args[@]}"

        # beethoven piano sonatas
        python "${SOFTWARE}" --dataset "beethoven" "${common_args[@]}"

        # youtube mix
        python "${SOFTWARE}" --dataset "youtube_mix" "${common_args[@]}"

        # sc09 speech
        python "${SOFTWARE}" --dataset "sc09" "${common_args[@]}"
    
    # further batches are not supported for yggdrasil
    else
        echo "Error: Invalid batch index ${BATCH} for machine: ${MACHINE}. Valid batch indices are: 0, 1, 2, 3"
        exit 1
    fi

# unknown machine
else
    echo "Error: Unknown machine: ${MACHINE}. Must be 'yggdrasil' or 'pando'"
    exit 1
fi