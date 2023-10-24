#!/bin/bash
#SBATCH --job-name=env_setup  # Job name
#SBATCH --output=logs/%j.out # Name of stdout output file
#SBATCH --error=logs/%j.err  # Name of stderr error file
#SBATCH --partition=dev-g  # Partition (queue) name
#SBATCH --nodes=1              # Total number of nodes
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 128 total (16x8)
#SBATCH --gpus-per-node=8
##SBATCH --mem=0
##SBATCH --exclusive=user
##SBATCH --hint=nomultithread
#SBATCH --time=00:10:00       # Run time (d-hh:mm:ss)
#SBATCH --account=project_2007628  # Project for billing
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

# Set up virtual environment for Megatron-DeepSpeed pretrain_gpt.py.

# This script creates the directories venv and apex. If either of
# these exists, ask to delete.

# Load modules
module load cray-python
module load LUMI/22.08 partition/G rocm5.4.2

#module use /pfs/lustrep2/projappl/project_462000125/samantao-public/mymodules
#module load aws-ofi-rccl/rocm-5.2.3

module use /appl/local/csc/modulefiles/
module load pytorch

# Create and activate venv
python -m venv --system-site-packages .venv
source .venv/bin/activate

# Upgrade pip etc.
python -m pip install --upgrade pip setuptools wheel

# Install pip packages
#python -m pip install --upgrade torch==1.13.1+rocm5.2 --extra-index-url https://download.pytorch.org/whl/rocm5.2
python -m pip install --upgrade numpy datasets evaluate accelerate scikit-learn nltk
# python -m pip install --upgrade /scratch/project_462000319/poyhnent/tietoevry/transformers
python -m pip install --upgrade /scratch/project_462000319/zosaelai2/transformers
python -m pip install --upgrade deepspeed==0.10.3
python -m pip install --upgrade tensorboard
python -m pip install --upgrade peft
python -m pip install --upgrade seqeval
