#!/bin/bash

# experiments to run

double_line="========================================"
line="----------------------------------------"

# PARAMETERS
##################################################

batch_size_mono=4
batch_size_stereo=2
sample_len=8192
cuda_visible_devices="0,1"
n_devices=2
wandb_group="longan"
max_epochs=2000

##################################################


# MONO
##################################################

echo ${double_line}
# echo
echo "MONO"
# echo
echo ${line}

# 8 bit
# echo
echo "8 bit, categorical:"
echo CUDA_VISIBLE_DEVICES=${cuda_visible_devices} python -m train experiment=audio/sashimi-musdb18mono wandb.group=${wandb_group} +wandb.name=musdb18mono-8bit-${sample_len}sl dataset.bits=8 dataset.sample_len=${sample_len} loader.batch_size=${batch_size_mono} trainer.max_epochs=${max_epochs} trainer.devices=${n_devices}

# DML, 8 bit
echo
echo "8 bit, DML:"
echo CUDA_VISIBLE_DEVICES=${cuda_visible_devices} python -m train experiment=audio/sashimi-dml-musdb18mono wandb.group=${wandb_group} +wandb.name=musdb18mono-dml-8bit-${sample_len}sl dataset.bits=8 dataset.sample_len=${sample_len} loader.batch_size=${batch_size_mono} trainer.max_epochs=${max_epochs} trainer.devices=${n_devices}

# echo
echo ${line}

# 16 bit
# echo
echo "16 bit, categorical:"
echo CUDA_VISIBLE_DEVICES=${cuda_visible_devices} python -m train experiment=audio/sashimi-musdb18mono wandb.group=${wandb_group} +wandb.name=musdb18mono-16bit-${sample_len}sl dataset.bits=16 dataset.sample_len=${sample_len} loader.batch_size=${batch_size_mono} trainer.max_epochs=${max_epochs} trainer.devices=${n_devices}

# DML, 16 bit
echo
echo "16 bit, DML:"
echo CUDA_VISIBLE_DEVICES=${cuda_visible_devices} python -m train experiment=audio/sashimi-dml-musdb18mono wandb.group=${wandb_group} +wandb.name=musdb18mono-dml-16bit-${sample_len}sl dataset.bits=16 dataset.sample_len=${sample_len} loader.batch_size=${batch_size_mono} trainer.max_epochs=${max_epochs} trainer.devices=${n_devices}

# echo
echo ${double_line}

##################################################

printf "\n\n\n"

# STEREO
##################################################

echo ${double_line}
# echo
echo "STEREO"
# echo
echo ${line}

# 8 bit
# echo
echo "8 bit, categorical:"
echo CUDA_VISIBLE_DEVICES=${cuda_visible_devices} python -m train experiment=audio/sashimi-musdb18stereo wandb.group=${wandb_group} +wandb.name=musdb18stereo-8bit-${sample_len}sl dataset.bits=8 dataset.sample_len=${sample_len} loader.batch_size=${batch_size_stereo} trainer.max_epochs=${max_epochs} trainer.devices=${n_devices}

# DML, 8 bit
echo
echo "8 bit, DML:"
echo CUDA_VISIBLE_DEVICES=${cuda_visible_devices} python -m train experiment=audio/sashimi-dml-musdb18stereo wandb.group=${wandb_group} +wandb.name=musdb18stereo-dml-8bit-${sample_len}sl dataset.bits=8 dataset.sample_len=${sample_len} loader.batch_size=${batch_size_stereo} trainer.max_epochs=${max_epochs} trainer.devices=${n_devices}

# echo
echo ${line}

# 16 bit
# echo
echo "16 bit, categorical:"
echo CUDA_VISIBLE_DEVICES=${cuda_visible_devices} python -m train experiment=audio/sashimi-musdb18stereo wandb.group=${wandb_group} +wandb.name=musdb18stereo-16bit-${sample_len}sl dataset.bits=16 dataset.sample_len=${sample_len} loader.batch_size=${batch_size_stereo} trainer.max_epochs=${max_epochs} trainer.devices=${n_devices}

# DML, 16 bit
echo
echo "16 bit, DML:"
echo CUDA_VISIBLE_DEVICES=${cuda_visible_devices} python -m train experiment=audio/sashimi-dml-musdb18stereo wandb.group=${wandb_group} +wandb.name=musdb18stereo-dml-16bit-${sample_len}sl dataset.bits=16 dataset.sample_len=${sample_len} loader.batch_size=${batch_size_stereo} trainer.max_epochs=${max_epochs} trainer.devices=${n_devices}

# echo
echo ${double_line}

##################################################