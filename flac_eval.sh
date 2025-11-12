#!/bin/bash

software="/home/pnlong/lnac/flac_eval.py" # software path
output="/home/pnlong/lnac/flac_eval_results.csv" # output filepath
fcl=${1} # flac compression level

# musdb18mono and musdb18stereo
# for bit_depth in 8 16; do
#     for mixes_only in "_mixes" ""; do
#         for partition in "_train" "_valid" ""; do
#             python "${software}" --dataset "musdb18mono${mixes_only}${partition}" --bit_depth "${bit_depth}" --output_filepath "${output}" --flac_compression_level "${fcl}"
#             python "${software}" --dataset "musdb18stereo${mixes_only}${partition}" --bit_depth "${bit_depth}" --output_filepath "${output}" --flac_compression_level "${fcl}"
#         done
#     done
# done
# for bit_depth in 8 16; do
#     for mixes_only in "_mixes" ""; do
#         python "${software}" --dataset "musdb18mono${mixes_only}" --bit_depth "${bit_depth}" --output_filepath "${output}" --flac_compression_level "${fcl}"
#         python "${software}" --dataset "musdb18stereo${mixes_only}" --bit_depth "${bit_depth}" --output_filepath "${output}" --flac_compression_level "${fcl}"
#     done
# done
for mixes_only in "_mixes" ""; do
    python "${software}" --dataset "musdb18mono${mixes_only}" --output_filepath "${output}" --flac_compression_level "${fcl}"
    python "${software}" --dataset "musdb18stereo${mixes_only}" --output_filepath "${output}" --flac_compression_level "${fcl}"
done

# librispeech
# for bit_depth in 8 16; do
#     python "${software}" --dataset "librispeech" --bit_depth "${bit_depth}" --output_filepath "${output}" --flac_compression_level "${fcl}"
# done
python "${software}" --dataset "librispeech" --output_filepath "${output}" --flac_compression_level "${fcl}"

# ljspeech
# for bit_depth in 8 16; do
#     python "${software}" --dataset "ljspeech" --bit_depth "${bit_depth}" --output_filepath "${output}" --flac_compression_level "${fcl}"
# done
python "${software}" --dataset "ljspeech" --output_filepath "${output}" --flac_compression_level "${fcl}"

# epidemic
# for bit_depth in 8 16; do
#     python "${software}" --dataset "epidemic" --bit_depth "${bit_depth}" --output_filepath "${output}" --flac_compression_level "${fcl}"
# done
python "${software}" --dataset "epidemic" --output_filepath "${output}" --flac_compression_level "${fcl}"

# vctk
# for bit_depth in 8 16; do
#     python "${software}" --dataset "vctk" --bit_depth "${bit_depth}" --output_filepath "${output}" --flac_compression_level "${fcl}"
# done
python "${software}" --dataset "vctk" --output_filepath "${output}" --flac_compression_level "${fcl}"

# torrent 16-bit
# for bit_depth in 8 16; do
#     for torrent_subset in "_pro" "_amateur" "_freeload" ""; do
#         python "${software}" --dataset "torrent16b${torrent_subset}" --bit_depth "${bit_depth}" --output_filepath "${output}" --flac_compression_level "${fcl}"
#     done
# done
for torrent_subset in "_pro" "_amateur" "_freeload" ""; do
    python "${software}" --dataset "torrent16b${torrent_subset}" --output_filepath "${output}" --flac_compression_level "${fcl}"
done

# torrent 24-bit
# for bit_depth in 8 16 24; do
#     for torrent_subset in "_pro" "_amateur" "_freeload" ""; do
#         python "${software}" --dataset "torrent24b${torrent_subset}" --bit_depth "${bit_depth}" --output_filepath "${output}" --flac_compression_level "${fcl}"
#     done
# done
for torrent_subset in "_pro" "_amateur" "_freeload" ""; do
    python "${software}" --dataset "torrent24b${torrent_subset}" --output_filepath "${output}" --flac_compression_level "${fcl}"
done

# birdvox
# for bit_depth in 8 16; do
#     python "${software}" --dataset "birdvox" --bit_depth "${bit_depth}" --output_filepath "${output}" --flac_compression_level "${fcl}"
# done
python "${software}" --dataset "birdvox" --output_filepath "${output}" --flac_compression_level "${fcl}"