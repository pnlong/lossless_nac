#!/bin/bash

software="/home/pnlong/lnac/flac_eval.py" # software path
output="/home/pnlong/lnac/flac_eval_results.csv" # output filepath
fcl=${1} # flac compression level

# musdb18mono and musdb18stereo
for mixes_only in "_mixes" ""; do
    python "${software}" --dataset "musdb18mono${mixes_only}" --output_filepath "${output}" --flac_compression_level "${fcl}"
    python "${software}" --dataset "musdb18stereo${mixes_only}" --output_filepath "${output}" --flac_compression_level "${fcl}"
done

# librispeech
python "${software}" --dataset "librispeech" --output_filepath "${output}" --flac_compression_level "${fcl}"

# ljspeech
python "${software}" --dataset "ljspeech" --output_filepath "${output}" --flac_compression_level "${fcl}"

# epidemic
python "${software}" --dataset "epidemic" --output_filepath "${output}" --flac_compression_level "${fcl}"

# vctk
python "${software}" --dataset "vctk" --output_filepath "${output}" --flac_compression_level "${fcl}"

# torrent 16-bit and 24-bit
for torrent_subset in "_pro" "_amateur" "_freeload" ""; do
    python "${software}" --dataset "torrent16b${torrent_subset}" --output_filepath "${output}" --flac_compression_level "${fcl}"
    python "${software}" --dataset "torrent24b${torrent_subset}" --output_filepath "${output}" --flac_compression_level "${fcl}"
done

# birdvox
python "${software}" --dataset "birdvox" --output_filepath "${output}" --flac_compression_level "${fcl}"