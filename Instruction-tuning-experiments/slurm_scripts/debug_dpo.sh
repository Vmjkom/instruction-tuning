#!/bin/bash
#SBATCH --job-name=debug_mnode_dpo
#SBATCH --account=project_462000241
#SBATCH --partition=dev-g
#SBATCH --cpus-per-task=56
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=480G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

rm -f logs/debug_latest.out logs/debug_latest.err
ln -s $SLURM_JOB_NAME-$SLURM_JOB_ID.out logs/debug_latest.out
ln -s $SLURM_JOB_NAME-$SLURM_JOB_ID.err logs/debug_latest.err

echo "JOB NAME" $SLURM_JOB_NAME
echo "PARTITION" $SLURM_JOB_PARTITION
echo "NNODES" $SLURM_NNODES
echo "NCPUS" $SLURM_CPUS_PER_TASK
echo "NODES" $SLURM_NODELIST

PYTHON_VENV=/projappl/project_462000319/villekom/instruction-tuning/Instruction-tuning-experiments/.venv

module purge
module load LUMI/22.08 partition/G
module use /appl/local/csc/modulefiles
module load pytorch/2.0
source $PYTHON_VENV/bin/activate

#export TORCH_EXTENSIONS_DIR=/tmp/$USER/torch_extensions
export PYTORCH_ROCM_ARCH=gfx90a

#Distributed variables
#export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
#master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
#export MASTER_ADDR=$master_addr
export LOCAL_RANK=$SLURM_LOCALID
export RANK=$SLURM_PROCID
#export WORLD_SIZE=$((SLURM_GPUS_ON_NODE*SLURM_NNODES))
export RDZV_HOST=$(hostname)
export RDZV_PORT=29400
#echo master_addr

export TOKENIZERS_PARALLELISM=true
#LOGGING
export TORCH_DISTRIBUTED_DEBUG=DETAIL
#export TRANSFORMERS_VERBOSITY=error
#export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

export OMP_NUM_THREADS=1
echo "Pwd $(pwd)"
#srun -l python3 -m torch.distributed.run --nnodes=$SLURM_NNODES \
#        --nproc-per-node=$SLURM_GPUS_ON_NODE \
#        --node_rank $LOCAL_RANK \
#        --rdzv-id=$SLURM_JOB_ID \
#        --rdzv-backend=c10d \
#        --rdzv-endpoint="$RDZV_HOST:$RDZV_PORT" \
#        train_dpo.py \
#        --model "/scratch/project_462000319/zosaelai2/models/sft_finetuned/merged-33B_torch_step70128_bfloat16-oasst-dolly-both-6epochs" \
#        --tokenizer "/scratch/project_462000319/tokenizers/tokenizer_v6_fixed_fin" \
#        --training_data "oasst" \
#        --lang "en" \
#        --num_train_epochs 1 \
#        --deepspeed_config "/projappl/project_462000319/villekom/instruction-tuning/Instruction-tuning-experiments/ds-configs/oa_deepspeed_rl_zero3.json" \


srun -l python3 -m torch.distributed.run --standalone \
        --nproc-per-node=$SLURM_GPUS_ON_NODE \
        train_dpo.py \
        --model "/scratch/project_462000319/zosaelai2/models/sft_finetuned/merged-33B_torch_step70128_bfloat16-oasst-dolly-both-6epochs" \
        --tokenizer "/scratch/project_462000319/tokenizers/tokenizer_v6_fixed_fin" \
        --training_data "oasst" \
        --lang "en" \
        --num_train_epochs 1 \
        --deepspeed_config "/projappl/project_462000319/villekom/instruction-tuning/Instruction-tuning-experiments/ds-configs/ds_config_stage2.json" \