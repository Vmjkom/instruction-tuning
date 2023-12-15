#!/bin/bash
#SBATCH --job-name=poro_sft  # Job name
#SBATCH --account=project_2007628  # Project for billing
#SBATCH --time=08:00:00       # Run time (d-hh:mm:ss)
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

deepspeed --num_gpus=4 huggingface-finetune.py \
        --model "/scratch/project_462000319/zosaelai2/models/33B_torch_step166752_bfloat16" \
        --tokenizer "/scratch/project_462000319/tokenizers/tokenizer_v6_fixed_fin" \
        --training_data "oasst" \
        --lang "en" \
        --task "sft" \
        --num_train_epochs 2 \
        --learning_rate 2e-5 \
        --deepspeed_config "./ds-configs/oa_zero3_config_sft_warmuplr.json"

echo "END $SLURM_JOBID: $(date)"
