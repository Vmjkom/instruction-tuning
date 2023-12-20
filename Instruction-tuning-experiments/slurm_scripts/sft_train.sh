#!/bin/bash
#SBATCH --job-name=trl_sft_poro_lora_dolly
#SBATCH --account=project_462000241
#SBATCH --partition=small-g
#SBATCH --cpus-per-task=56
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --mem=480G
#SBATCH --exclusive
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err


rm -f logs/latest.out logs/latest.err
ln -s $SLURM_JOB_NAME-$SLURM_JOB_ID.out logs/latest.out
ln -s $SLURM_JOB_NAME-$SLURM_JOB_ID.err logs/latest.err

echo "JOB NAME" $SLURM_JOB_NAME
echo "PARTITION" $SLURM_JOB_PARTITION
echo "NNODES" $SLURM_NNODES
echo "NCPUS" $SLURM_CPUS_PER_TASK
echo "NODES" $SLURM_NODELIST
echo "NGPUS" $SLURM_GPUS_ON_NODE

module purge
module use /appl/local/csc/modulefiles
module load pytorch
source .venv/bin/activate

export TORCH_EXTENSIONS_DIR=/tmp/$USER/torch_extensions
rm -rf $TORCH_EXTENSIONS_DIR
export CACHE=$TRANSFORMERS_CACHE
export LOGS=/scratch/project_462000319/$USER/logs
export PYTORCH_ROCM_ARCH=gfx90a
export PYTHONPATH=.venv/lib/python3.10/site-packages/

#Distributed variables
#export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
#master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
#export MASTER_ADDR=$master_addr
export LOCAL_RANK=$SLURM_LOCALID
#export RANK=$SLURM_PROCID
#export WORLD_SIZE=$((SLURM_GPUS_ON_NODE*SLURM_NNODES))
#export RDZV_HOST=$(hostname)
#export RDZV_PORT=29400
#echo master_addr


#LOGGING/DEBUGGING
export TRANSFORMERS_VERBOSITY=error
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1

export OMP_NUM_THREADS=1

srun --jobid $SLURM_JOBID bash -c "torchrun --role $(hostname -s): --tee 3 --nnodes  $SLURM_NNODES \
        --nproc-per-node=$SLURM_GPUS_ON_NODE \
        --node_rank \$SLURM_PROCID \
        train_sft.py \
        --model LumiOpen/Poro-34B \
        --num_train_epochs 1 \
        --save_directory /scratch/project_462000241/villekom/models/trl/sft \
        --local_rank $LOCAL_RANK \
        --deepspeed_config ./ds-configs/ds_config.json \
        --use_peft \
        --debug "