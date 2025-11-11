#!/bin/bash

software_path="/home/pnlong/lnac/flac_eval.py"
output_filepath="/home/pnlong/lnac/flac_eval_results.csv"
flac_compression_level=${1}

# musdb18mono and musdb18stereo
# for bit_depth in 8 16; do
#     for mixes_only in "_mixes" ""; do
#         for partition in "_train" "_valid" ""; do
#             python "${software_path}" --dataset "musdb18mono${mixes_only}${partition}" --bit_depth "${bit_depth}" --output_filepath "${output_filepath}" --flac_compression_level "${flac_compression_level}"
#             python "${software_path}" --dataset "musdb18stereo${mixes_only}${partition}" --bit_depth "${bit_depth}" --output_filepath "${output_filepath}" --flac_compression_level "${flac_compression_level}"
#         done
#     done
# done
for bit_depth in 8 16; do
    for mixes_only in "_mixes" ""; do
        python "${software_path}" --dataset "musdb18mono${mixes_only}" --bit_depth "${bit_depth}" --output_filepath "${output_filepath}" --flac_compression_level "${flac_compression_level}"
        python "${software_path}" --dataset "musdb18stereo${mixes_only}" --bit_depth "${bit_depth}" --output_filepath "${output_filepath}" --flac_compression_level "${flac_compression_level}"
    done
done

# librispeech
for bit_depth in 8 16; do
    python "${software_path}" --dataset "librispeech" --bit_depth "${bit_depth}" --output_filepath "${output_filepath}" --flac_compression_level "${flac_compression_level}"
done

# ljspeech
for bit_depth in 8 16; do
    python "${software_path}" --dataset "ljspeech" --bit_depth "${bit_depth}" --output_filepath "${output_filepath}" --flac_compression_level "${flac_compression_level}"
done

# epidemic
for bit_depth in 8 16; do
    python "${software_path}" --dataset "epidemic" --bit_depth "${bit_depth}" --output_filepath "${output_filepath}" --flac_compression_level "${flac_compression_level}"
done

# vctk
for bit_depth in 8 16; do
    python "${software_path}" --dataset "vctk" --bit_depth "${bit_depth}" --output_filepath "${output_filepath}" --flac_compression_level "${flac_compression_level}"
done

# torrent 16-bit
for bit_depth in 8 16; do
    for torrent_subset in "_pro" "_amateur" "_freeload" ""; do
        python "${software_path}" --dataset "torrent16b${torrent_subset}" --bit_depth "${bit_depth}" --output_filepath "${output_filepath}" --flac_compression_level "${flac_compression_level}"
    done
done

# torrent 24-bit
for bit_depth in 8 16 24; do
    for torrent_subset in "_pro" "_amateur" "_freeload" ""; do
        python "${software_path}" --dataset "torrent24b${torrent_subset}" --bit_depth "${bit_depth}" --output_filepath "${output_filepath}" --flac_compression_level "${flac_compression_level}"
    done
done

# birdvox
for bit_depth in 8 16; do
    python "${software_path}" --dataset "birdvox" --bit_depth "${bit_depth}" --output_filepath "${output_filepath}" --flac_compression_level "${flac_compression_level}"
done