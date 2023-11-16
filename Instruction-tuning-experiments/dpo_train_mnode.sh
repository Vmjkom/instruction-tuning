#!/bin/bash
#SBATCH --job-name=mnode_dpo
#SBATCH --account=project_462000319
#SBATCH --partition=standard-g
#SBATCH --cpus-per-task=4
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=0
#SBATCH -t 12:00:00
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err


# rm -f logs/latest_mnode.out logs/latest_mnode.err
ln -s $SLURM_JOB_NAME-$SLURM_JOB_ID.out logs/latest_mnode.out
ln -s $SLURM_JOB_NAME-$SLURM_JOB_ID.err logs/latest_mnode.err

echo "JOB NAME" $SLURM_JOB_NAME
echo "PARTITION" $SLURM_JOB_PARTITION
echo "NNODES" $SLURM_NNODES
echo "NCPUS" $SLURM_CPUS_PER_TASK
echo "NODES" $SLURM_NODELIST

module purge
module load LUMI/22.08 partition/G rocm/5.2.3
module use /appl/local/csc/modulefiles
module load pytorch



source /scratch/project_462000319/zosaelai2/.venv2/bin/activate

export NCCL_SOCKET_IFNAME=hsn0

#export TORCH_EXTENSIONS_DIR=/tmp/$USER/torch_extensions
export CACHE=$TRANSFORMERS_CACHE
# export MODEL_PATH=/scratch/project_462000319/$USER/oa_models
export LOGS=/scratch/project_462000319/$USER/logs
export PYTORCH_ROCM_ARCH=gfx90a
export TOKENIZERS_PARALLELISM=true

#Distributed variables
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
export LOCAL_RANK=$SLURM_LOCALID
#export RANK=$SLURM_PROCID
export WORLD_SIZE=$((SLURM_GPUS_ON_NODE*SLURM_NNODES))
export RDZV_HOST=$(hostname)
export RDZV_PORT=29400
#export CUDA_LAUNCH_BLOCKING=1
echo master_addr


#LOGGING
#export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TRANSFORMERS_VERBOSITY=error
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

export OMP_NUM_THREADS=1

srun -l python3 -m torch.distributed.run --nnodes=$SLURM_NNODES \
        --nproc-per-node=$SLURM_GPUS_ON_NODE \
        --node_rank $LOCAL_RANK \
        --master_addr $MASTER_ADDR \
        --master_port $MASTER_PORT \
        --rdzv-id=$SLURM_JOB_ID \
        --rdzv-backend=c10d \
        --rdzv-endpoint="$RDZV_HOST:$RDZV_PORT" \
        train_dpo.py \
        --model "/scratch/project_462000319/zosaelai2/models/sft_finetuned/merged-33B_torch_step70128_bfloat16-oasst-dolly-both-6epochs" \
        --tokenizer "/scratch/project_462000319/tokenizers/tokenizer_v6_fixed_fin" \
        --training_data "oasst" \
        --lang "en" \
        --num_train_epochs 1 \

# deepspeed --num_gpus=8 train_dpo.py \
#       --model "/scratch/project_462000319/zosaelai2/models/sft_finetuned/merged-33B_torch_step70128_bfloat16-dolly-fi-2epochs-oasst-fi-2epochs" \
#       --tokenizer "/scratch/project_462000319/tokenizers/tokenizer_v6_fixed_fin" \
#       --training_data "oasst" \
#       --lang "en" \
#       --num_train_epochs 1 \