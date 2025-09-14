#!/bin/bash

# experiments to run

line="----------------------------------------"

# PARAMETERS
##################################################

batch_size=8
sample_len=4096
cuda_visible_devices="0,1"
n_devices=2
wandb_group="rambutan"

##################################################


# MONO
##################################################

echo "MONO"

echo ${line}
echo "8 bit:"

# 8 bit
echo CUDA_VISIBLE_DEVICES=${cuda_visible_devices} python -m train experiment=audio/sashimi-musdb18mono wandb.group=${wandb_group} +wandb.name=musdb18mono-8bit-${sample_len}sl dataset.bits=8 dataset.sample_len=${sample_len} trainer.devices=${n_devices} loader.batch_size=${batch_size}

# DML, 8 bit
echo CUDA_VISIBLE_DEVICES=${cuda_visible_devices} python -m train experiment=audio/sashimi-dml-musdb18mono wandb.group=${wandb_group} +wandb.name=musdb18mono-dml-8bit-${sample_len}sl dataset.bits=8 dataset.sample_len=${sample_len} trainer.devices=${n_devices} loader.batch_size=${batch_size}

echo ${line}
echo "16 bit:"

# 16 bit
echo CUDA_VISIBLE_DEVICES=${cuda_visible_devices} python -m train experiment=audio/sashimi-musdb18mono wandb.group=${wandb_group} +wandb.name=musdb18mono-16bit-${sample_len}sl dataset.bits=16 dataset.sample_len=${sample_len} trainer.devices=${n_devices} loader.batch_size=${batch_size}

# DML, 16 bit
echo CUDA_VISIBLE_DEVICES=${cuda_visible_devices} python -m train experiment=audio/sashimi-dml-musdb18mono wandb.group=${wandb_group} +wandb.name=musdb18mono-dml-16bit-${sample_len}sl dataset.bits=16 dataset.sample_len=${sample_len} trainer.devices=${n_devices} loader.batch_size=${batch_size}

echo ${line}

##################################################

echo

# STEREO
##################################################

echo "STEREO"

echo ${line}
echo "8 bit:"

# 8 bit
echo CUDA_VISIBLE_DEVICES=${cuda_visible_devices} python -m train experiment=audio/sashimi-musdb18stereo wandb.group=${wandb_group} +wandb.name=musdb18stereo-8bit-${sample_len}sl dataset.bits=8 dataset.sample_len=${sample_len} trainer.devices=${n_devices} loader.batch_size=${batch_size}

# DML, 8 bit
echo CUDA_VISIBLE_DEVICES=${cuda_visible_devices} python -m train experiment=audio/sashimi-dml-musdb18stereo wandb.group=${wandb_group} +wandb.name=musdb18stereo-dml-8bit-${sample_len}sl dataset.bits=8 dataset.sample_len=${sample_len} trainer.devices=${n_devices} loader.batch_size=${batch_size}

echo ${line}
echo "16 bit:"

# 16 bit
echo CUDA_VISIBLE_DEVICES=${cuda_visible_devices} python -m train experiment=audio/sashimi-musdb18stereo wandb.group=${wandb_group} +wandb.name=musdb18stereo-16bit-${sample_len}sl dataset.bits=16 dataset.sample_len=${sample_len} trainer.devices=${n_devices} loader.batch_size=${batch_size}

# DML, 16 bit
echo CUDA_VISIBLE_DEVICES=${cuda_visible_devices} python -m train experiment=audio/sashimi-dml-musdb18stereo wandb.group=${wandb_group} +wandb.name=musdb18stereo-dml-16bit-${sample_len}sl dataset.bits=16 dataset.sample_len=${sample_len} trainer.devices=${n_devices} loader.batch_size=${batch_size}

echo ${line}

##################################################