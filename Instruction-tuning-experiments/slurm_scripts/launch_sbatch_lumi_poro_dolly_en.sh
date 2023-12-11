#!/bin/bash
#SBATCH --job-name=generate_poro_dolly_sft_en  # Job name
#SBATCH --output=logs/%j.out # Name of stdout output file
#SBATCH --error=logs/%j.err  # Name of stderr error file
#SBATCH --partition=standard-g  # Partition (queue) name
#SBATCH --nodes=1              # Total number of nodes
#SBATCH --ntasks-per-node=1     # 8 MPI ranks per node, 128 total (16x8)
#SBATCH --gpus-per-node=8
#SBATCH --time=4:00:00       # Run time (d-hh:mm:ss)
#SBATCH --account=project_462000319  # Project for billing
#SBATCH -o logs/%j.out
#SBATCH -e logs/%j.err

# Hoping to resolve "Cassini Event Queue overflow detected." errors
export FI_CXI_DEFAULT_CQ_SIZE=262144    # default 131072

export NCCL_SOCKET_IFNAME=hsn

export TRANSFORMERS_CACHE=/scratch/project_462000319/.cache
# export TRANSFORMERS_CACHE=/scratch/project_462000319/zosaelai2/.cache
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="/scratch/project_462000319/zosaelai2"

module load cray-python
module load LUMI/22.08 partition/G rocm/5.2.3

module use /appl/local/csc/modulefiles/
module load pytorch

source /scratch/project_462000319/zosaelai2/.venv/bin/activate

echo "$(python -c 'import torch; print(torch.cuda.is_available())')"

# deepspeed --num_gpus=8 huggingface-finetune.py \
#       --model "/scratch/project_462000319/zosaelai2/models/33B_torch_step70128_bfloat16" \
#       --tokenizer "/scratch/project_462000319/tokenizers/tokenizer_v6_fixed_fin" \
#       --training_data "dolly" \
#       --lang "en" \
#       --task "sft" \
#       --num_train_epochs 3 \

# python merge_peft_model.py --lora_adapter "/scratch/project_462000319/zosaelai2/models/sft_checkpoints/33B_torch_step70128_bfloat16-dolly-en-3epochs/checkpoint-1263" \
#       --output_dir "/scratch/project_462000319/zosaelai2/models/sft_finetuned/merged-33B_torch_step70128_bfloat16-dolly-en-3epochs"	

python generate.py \
        --model /scratch/project_462000319/zosaelai2/models/sft_finetuned/merged-33B_torch_step70128_bfloat16-dolly-en-2epochs \
        --file data/dolly-fi/dolly-fi-eval.jsonl \
        --tokenizer /scratch/project_462000319/tokenizers/tokenizer_v6_fixed_fin/ \
        --max_prompts 0 \
        --lang fi \
        --eval_only False

echo "END $SLURM_JOBID: $(date)"
