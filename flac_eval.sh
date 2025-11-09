#!/bin/bash

software_path="/home/pnlong/lnac/flac_eval.py"
output_filepath="/home/pnlong/lnac/flac_eval_results.csv"

# musdb18mono and musdb18stereo
for bit_depth in 8 16 24; do
    for mixes_only in "_mixes" ""; do
        for partition in "_train" "_valid" ""; do
            python "${software_path}" --dataset "musdb18mono${mixes_only}${partition}" --bit_depth "${bit_depth}" --output_filepath "${output_filepath}"
            python "${software_path}" --dataset "musdb18stereo${mixes_only}${partition}" --bit_depth "${bit_depth}" --output_filepath "${output_filepath}"
        done
    done
done

# librispeech
for bit_depth in 8 16 24; do
    python "${software_path}" --dataset "librispeech" --bit_depth "${bit_depth}" --output_filepath "${output_filepath}"
done

# ljspeech
for bit_depth in 8 16 24; do
    python "${software_path}" --dataset "ljspeech" --bit_depth "${bit_depth}" --output_filepath "${output_filepath}"
done

# epidemic
for bit_depth in 8 16 24; do
    python "${software_path}" --dataset "epidemic" --bit_depth "${bit_depth}" --output_filepath "${output_filepath}"
done

# vctk
for bit_depth in 8 16 24; do
    python "${software_path}" --dataset "vctk" --bit_depth "${bit_depth}" --output_filepath "${output_filepath}"
done

# torrent
for native_bit_depth in 16 24; do
    for bit_depth in 8 16 24; do
        for torrent_subset in "_pro" "_amateur" "_freeload" ""; do
            python "${software_path}" --dataset "torrent${native_bit_depth}b${torrent_subset}" --bit_depth "${bit_depth}" --output_filepath "${output_filepath}"
        done
    done
done

# birdvox
for bit_depth in 8 16 24; do
    python "${software_path}" --dataset "birdvox" --bit_depth "${bit_depth}" --output_filepath "${output_filepath}"
done