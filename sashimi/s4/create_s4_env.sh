#!/bin/bash
# create new s4 environment
mamba create -n s4 python=3.9
mamba activate s4
mamba install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.6 mkl=2022.* -c pytorch -c nvidia
mamba install -c conda-forge numpy scipy pandas scikit-learn matplotlib tqdm rich lit pytorch-lightning=2.0.4 hydra-core omegaconf wandb einops cmake transformers datasets sktime numba gluonts timm=0.5.4
pip install torchtext==0.14.1 --no-dependencies
mamba remove rich pytorch-lightning
pip install rich==13.3.5 pytorch-lightning==2.0.4 --no-dependencies
pip install lightning-utilities --no-dependencies
pip install torchmetrics --no-dependencies
mamba remove --force pytorch pytorch-lightning torchvision torchaudio torchtext rich transformers
mamba install -c pytorch -c nvidia pytorch=2.1 pytorch-cuda=11.8 torchvision torchaudio torchtext
mamba install -c conda-forge pytorch-lightning=2.1 rich pygments transformers lightning-utilities
mamba install natsort
mamba remove pytorch-lightning rich
mamba install -c conda-forge pytorch-lightning=2.1.0 rich=13.3.5