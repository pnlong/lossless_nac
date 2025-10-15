#!/bin/bash
# README
# Phillip Long
# October 14, 2025

# This script includes commands for running zero-shot.py with different experiments

# paths
software_path="/home/pnlong/lnac/lmic/zero_shot.py"
mono_audio_dir="/graft4/datasets/pnlong/lnac/sashimi/data/musdb18mono"
stereo_audio_dir="/graft4/datasets/pnlong/lnac/sashimi/data/musdb18stereo"
llama_7b_path="/graft4/checkpoints/pnlong/lnac/lmic/llama/Llama-2-7b"
llama_13b_path="/graft4/checkpoints/pnlong/lnac/lmic/llama/Llama-2-13b"
llama_70b_path="/graft4/checkpoints/pnlong/lnac/lmic/llama/Llama-2-70b"
outputs_dir="/home/pnlong/lnac/lmic/language_modeling_is_compression/outputs"
max_length=2048
stereo_blocking_n=1024
batch_size=2
gpu=0

# mono
for bit_depth in 8 16; do
    echo "Mono, ${bit_depth} bit:"
    echo CUDA_VISIBLE_DEVICES="${gpu}" python "${software_path}" --gpu --audio_dir "${mono_audio_dir}" --model_path "${llama_7b_path}" --bit_depth "${bit_depth}" --output_file "${outputs_dir}/mono_llama2-7b_${bit_depth}bit_${max_length}.out" --max_length "${max_length}" --batch_size "${batch_size}"
    echo CUDA_VISIBLE_DEVICES="${gpu}" python "${software_path}" --gpu --audio_dir "${mono_audio_dir}" --model_path "${llama_13b_path}" --bit_depth "${bit_depth}" --output_file "${outputs_dir}/mono_llama2-13b_${bit_depth}bit_${max_length}.out" --max_length "${max_length}" --batch_size "${batch_size}"
    echo CUDA_VISIBLE_DEVICES="${gpu}" python "${software_path}" --gpu --audio_dir "${mono_audio_dir}" --model_path "${llama_70b_path}" --bit_depth "${bit_depth}" --output_file "${outputs_dir}/mono_llama2-70b_${bit_depth}bit_${max_length}.out" --max_length "${max_length}" --batch_size "${batch_size}"
    echo
done

# stereo
for bit_depth in 8 16; do
    echo "Stereo, ${bit_depth} bit:"
    echo CUDA_VISIBLE_DEVICES="${gpu}" python "${software_path}" --gpu --audio_dir "${stereo_audio_dir}" --model_path "${llama_7b_path}" --bit_depth "${bit_depth}" --output_file "${outputs_dir}/stereo_llama2-7b_${bit_depth}bit_${max_length}_blocking-${stereo_blocking_n}.out" --max_length "${max_length}" --batch_size "${batch_size}" --stereo_blocking_n "${stereo_blocking_n}"
    echo CUDA_VISIBLE_DEVICES="${gpu}" python "${software_path}" --gpu --audio_dir "${stereo_audio_dir}" --model_path "${llama_13b_path}" --bit_depth "${bit_depth}" --output_file "${outputs_dir}/stereo_llama2-13b_${bit_depth}bit_${max_length}_blocking-${stereo_blocking_n}.out" --max_length "${max_length}" --batch_size "${batch_size}" --stereo_blocking_n "${stereo_blocking_n}"
    echo CUDA_VISIBLE_DEVICES="${gpu}" python "${software_path}" --gpu --audio_dir "${stereo_audio_dir}" --model_path "${llama_70b_path}" --bit_depth "${bit_depth}" --output_file "${outputs_dir}/stereo_llama2-70b_${bit_depth}bit_${max_length}_blocking-${stereo_blocking_n}.out" --max_length "${max_length}" --batch_size "${batch_size}" --stereo_blocking_n "${stereo_blocking_n}"
    echo
done