#!/bin/bash
#SBATCH --job-name=debug_poro_sft  # Job name
#SBATCH --output=logs/%j.out # Name of stdout output file
#SBATCH --error=logs/%j.err  # Name of stderr error file
#SBATCH --partition=dev-g  # Partition (queue) name
#SBATCH --nodes=1              # Total number of nodes
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 128 total (16x8)
#SBATCH --cpus-per-task=56
#SBATCH --gpus-per-node=8
#SBATCH --mem=100G
#SBATCH --time=2:00:00       # Run time (d-hh:mm:ss)
#SBATCH --account=project_462000241  # Project for billing
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

rm -f logs/latest.out logs/latest.err
ln -s $SLURM_JOB_NAME-$SLURM_JOB_ID.out logs/latest.out
ln -s $SLURM_JOB_NAME-$SLURM_JOB_ID.err logs/latest.err

# Hoping to resolve "Cassini Event Queue overflow detected." errors
export FI_CXI_DEFAULT_CQ_SIZE=262144    # default 131072

export TRANSFORMERS_CACHE=/scratch/project_462000319/transformers_cache
export TOKENIZERS_PARALLELISM=false

module load LUMI/22.08 partition/G
module use /appl/local/csc/modulefiles/
module load pytorch
export SING_IMAGE=/appl/local/csc/soft/ai/images/pytorch_2.0.1_rocm_ubuntu_hf_bf16_fix.sif

source /projappl/project_462000319/villekom/instruction-tuning/Instruction-tuning-experiments/.venv/bin/activate
export PYTHONPATH=/projappl/project_462000319/villekom/instruction-tuning/Instruction-tuning-experiments/.venv/lib/python3.10/site-packages
OMP_NUM_THREADS=2
torchrun --nproc-per-node=$SLURM_GPUS_ON_NODE --standalone huggingface-finetune.py

echo "END $SLURM_JOBID: $(date)"
