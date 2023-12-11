#!/bin/bash
#SBATCH --job-name=poro_sft  # Job name
#SBATCH --account=project_2007628  # Project for billing
#SBATCH --time=01:00:00       # Run time (d-hh:mm:ss)
#SBATCH --partition=gpusmall
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=128G
#SBATCH --output=logs/%j.out # Name of stdout output file
#SBATCH --error=logs/%j.err  # Name of stderr error file

# Hoping to resolve "Cassini Event Queue overflow detected." errors
# export FI_CXI_DEFAULT_CQ_SIZE=262144    # default 131072

# export NCCL_SOCKET_IFNAME=hsn

#DEBUG
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=INIT,GRAPH,ENV
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

export TRANSFORMERS_CACHE=/scratch/project_2007628/transformers_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/scratch/project_2007628/zosaelai2"

module use /appl/local/csc/modulefiles/
module load pytorch/2.0

source /scratch/project_2007628/zosaelai2/.venv/bin/activate

echo "$(python -c 'import torch; print(torch.cuda.is_available())')"

deepspeed --num_gpus=2 huggingface-finetune.py

echo "END $SLURM_JOBID: $(date)"
