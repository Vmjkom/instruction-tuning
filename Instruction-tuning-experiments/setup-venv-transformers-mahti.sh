#!/bin/bash
#SBATCH --job-name=env_setup  # Job name
#SBATCH --account=project_2007628  # Project for billing
#SBATCH --time=00:10:00       # Run time (d-hh:mm:ss)
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:a100:1,nvme:950
#SBATCH --output=logs/%j.out # Name of stdout output file
#SBATCH --error=logs/%j.err  # Name of stderr error file

# Set up virtual environment for Megatron-DeepSpeed pretrain_gpt.py.

# This script creates the directories venv and apex. If either of
# these exists, ask to delete.

module use /appl/local/csc/modulefiles/
module load pytorch/2.0

# Create and activate venv
python -m venv --system-site-packages .venv
source .venv/bin/activate

# Upgrade pip etc.
python -m pip install --upgrade pip setuptools wheel

# Install pip packages
#python -m pip install --upgrade torch==1.13.1+rocm5.2 --extra-index-url https://download.pytorch.org/whl/rocm5.2
python -m pip install --upgrade numpy datasets evaluate accelerate scikit-learn nltk
python -m pip install --upgrade transformers
python -m pip install --upgrade trl
python -m pip install --upgrade deepspeed
python -m pip install --upgrade tensorboard
python -m pip install --upgrade peft
python -m pip install --upgrade seqeval
