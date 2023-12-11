#!/bin/bash
#SBATCH --job-name=poro_sft  # Job name
#SBATCH --output=logs/%j.out # Name of stdout output file
#SBATCH --error=logs/%j.err  # Name of stderr error file
#SBATCH --partition=gpu  # Partition (queue) name
#SBATCH --nodes=1              # Total number of nodes
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 128 total (16x8)
#SBATCH --gpus-per-node=8
##SBATCH --mem=0
##SBATCH --exclusive=user
##SBATCH --hint=nomultithread
#SBATCH --time=4:00:00       # Run time (d-hh:mm:ss)
#SBATCH --account=project_2007628  # Project for billing
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

# Hoping to resolve "Cassini Event Queue overflow detected." errors
# export FI_CXI_DEFAULT_CQ_SIZE=262144    # default 131072

# export NCCL_SOCKET_IFNAME=hsn

# export TRANSFORMERS_CACHE=/scratch/project_462000319/.cache
export TRANSFORMERS_CACHE=/scratch/project_2007628/transformers_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/scratch/project_2007628/silo/zosaelai2"

module --force purge
module load pytorch/1.13

source /scratch/project_2007628/zosaelai2/venv/bin/activate

echo "$(python -c 'import torch; print(torch.cuda.is_available())')"

deepspeed --num_gpus=8 huggingface-finetune.py

echo "END $SLURM_JOBID: $(date)"
