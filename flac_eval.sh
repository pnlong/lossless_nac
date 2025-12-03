#!/bin/bash
# for compression_level in $(seq 0 8); do bash flac_eval.sh --compression-level ${compression_level}; done
# for compression_level in $(seq 0 8); do bash flac_eval.sh --compression-level ${compression_level} --disable-constant-subframes --disable-fixed-subframes --disable-verbatim-subframes; done

set -e

# Default values
SOFTWARE="/home/pnlong/lnac/flac_eval.py" # software path
OUTPUT="/home/pnlong/lnac/flac_eval_results.csv" # output filepath
COMPRESSION_LEVEL=5 # flac compression level
DISABLE_CONSTANT_SUBFRAMES=""
DISABLE_FIXED_SUBFRAMES=""
DISABLE_VERBATIM_SUBFRAMES=""
MACHINE="yggdrasil" # machine to use (yggdrasil or pando)

# Usage function
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Options:
    -c, --compression-level LEVEL    FLAC compression level (0-8, default: ${COMPRESSION_LEVEL})
    --disable-constant-subframes Disable constant subframes for FLAC
    --disable-fixed-subframes    Disable fixed subframes for FLAC
    --disable-verbatim-subframes Disable verbatim subframes for FLAC
    --machine MACHINE            Machine to use (yggdrasil or pando, default: ${MACHINE})
    -h, --help                   Show this help message

Examples:
    $0                                    # Use default compression level ${COMPRESSION_LEVEL}
    $0 --compression-level 8              # Use compression level 8
    $0 --compression-level 5 --disable-constant-subframes --disable-fixed-subframes
EOF
}

# Parse command line arguments using getopt
OPTS=$(getopt -o "h" --long compression-level:,disable-constant-subframes,disable-fixed-subframes,disable-verbatim-subframes,machine:,help -- "$@")
if [ $? -ne 0 ]; then
    echo "Error: Failed to parse options"
    usage
    exit 1
fi

eval set -- "$OPTS"

while true; do
    case "$1" in
        -c|--compression-level)
            COMPRESSION_LEVEL="$2"
            shift 2
            ;;
        --disable-constant-subframes)
            DISABLE_CONSTANT_SUBFRAMES="--disable_constant_subframes"
            shift
            ;;
        --disable-fixed-subframes)
            DISABLE_FIXED_SUBFRAMES="--disable_fixed_subframes"
            shift
            ;;
        --disable-verbatim-subframes)
            DISABLE_VERBATIM_SUBFRAMES="--disable_verbatim_subframes"
            shift
            ;;
        --machine)
            MACHINE="$2"
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

# Validate compression level
if ! [[ "$COMPRESSION_LEVEL" =~ ^[0-8]$ ]]; then
    echo "Error: --compression-level must be between 0 and 8"
    exit 1
fi

# Validate machine
if ! [[ "$MACHINE" =~ ^(yggdrasil|pando)$ ]]; then
    echo "Error: Unknown machine: ${MACHINE}. Must be 'yggdrasil' or 'pando'"
    exit 1
fi

# Build common arguments
common_args=(
    "--output_filepath" "${OUTPUT}"
    "--flac_compression_level" "${COMPRESSION_LEVEL}"
)
[[ -n "$DISABLE_CONSTANT_SUBFRAMES" ]] && common_args+=("$DISABLE_CONSTANT_SUBFRAMES")
[[ -n "$DISABLE_FIXED_SUBFRAMES" ]] && common_args+=("$DISABLE_FIXED_SUBFRAMES")
[[ -n "$DISABLE_VERBATIM_SUBFRAMES" ]] && common_args+=("$DISABLE_VERBATIM_SUBFRAMES")

# Run datasets based on machine

# pando datasets
if [ "$MACHINE" == "pando" ]; then

    # epidemic sound
    python "${SOFTWARE}" --dataset "epidemic" "${common_args[@]}"

    # vctk
    python "${SOFTWARE}" --dataset "vctk" "${common_args[@]}"

# yggdrasil datasets
elif [ "$MACHINE" == "yggdrasil" ]; then

    # musdb18mono and musdb18stereo
    for subset in "_mixes" "_stems" ""; do
        python "${SOFTWARE}" --dataset "musdb18mono${subset}" "${common_args[@]}"
        python "${SOFTWARE}" --dataset "musdb18stereo${subset}" "${common_args[@]}"
    done

    # librispeech
    python "${SOFTWARE}" --dataset "librispeech" "${common_args[@]}"

    # ljspeech
    python "${SOFTWARE}" --dataset "ljspeech" "${common_args[@]}"

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

# unknown machine
else
    echo "Error: Unknown machine: ${MACHINE}. Must be 'yggdrasil' or 'pando'"
    exit 1
fi