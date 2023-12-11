#!/bin/bash
#SBATCH --job-name=env_setup 
#SBATCH --partition=small-g  
#SBATCH --nodes=1              
#SBATCH --ntasks-per-node=1     
#SBATCH --gpus-per-node=1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=10
#SBATCH --time=00:10:00
#SBATCH --account=project_462000241
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

mkdir -p logs

# Load modules
module load LUMI/22.08
module use /appl/local/csc/modulefiles/
module load pytorch/2.0
#Fix for the bf16 amd compability
export SING_IMAGE=/appl/local/csc/soft/ai/images/pytorch_2.0.1_rocm_ubuntu_hf_bf16_fix.sif

# Create and activate venv
python -m venv --system-site-packages ../.venv
source ../.venv/bin/activate

# Install pip packages
python -m pip install --upgrade peft
python -m pip install --upgrade seqeval