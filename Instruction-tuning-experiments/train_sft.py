import sys
import os
import torch
import numpy as np
import logging
from datasets import DatasetDict, Dataset, load_dataset
from argparse import ArgumentParser

from transformers import (
    AutoTokenizer,
    TrainingArguments,
    AutoModelForCausalLM
)
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    LoraConfig,
    TaskType
)
from trl import (
    SFTTrainer,
    DataCollatorForCompletionOnlyLM
)

# custom classes
from utils import load_model, logits_argmax, print_rank_0, is_last_rank
from instruction_finetuning_datasets import read_data_sft

user_token = "<|user|>"
assistant_token = "<|assistant|>"

def argparser():
    ap = ArgumentParser()
    ap.add_argument('--deepspeed_config', type=str, default="./ds-configs/oa_zero3_config_sft.json")
    ap.add_argument('--model', type=str)
    ap.add_argument('--num_train_epochs', type=int, default=1)
    ap.add_argument('--per_device_batch_size', type=int, default=1)
    ap.add_argument('--training_data', type=str, default="oasst")
    ap.add_argument('--lang', type=str, default="fi")
    ap.add_argument('--local_rank', type=int)
    ap.add_argument('--use_peft', action='store_true')
    ap.add_argument('--transformers_cache',type=str, default="/scratch/project_462000319/transformers_cache")
    ap.add_argument('--prompt_structure', default=False, type=lambda x: (str(x).lower() == 'true'))
    ap.add_argument('--save_directory',type=str)
    ap.add_argument('--debug',action='store_true')
    return ap

#FOR DOLLY
def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['INSTRUCTION'])):
        text = f"{user_token}: {example['INSTRUCTION'][i]}\n {assistant_token}: {example['RESPONSE'][i]}"
        output_texts.append(text)
        if i % 50 == 0:
            logging.debug(f"Formatted prompt {text}")
    return output_texts


def train_sft(args):
   # if args.debug:
   #     logging.basicConfig(level=logging.DEBUG)
    log_dir = './logs/'
    base_model_name = os.path.join(args.model)
    output_dir = os.path.join(args.save_directory)
    # This needs to be defined before model loading for deepspeed stage 3 to work correctly
    training_args = TrainingArguments(
        deepspeed=args.deepspeed_config,
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=100,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=1,
        eval_accumulation_steps=4,
        log_on_each_node=False,
        logging_strategy="steps",
        logging_steps=10,
        num_train_epochs=args.num_train_epochs,
        save_strategy="epoch",
        save_steps=100,
        save_total_limit=1,
        weight_decay=0.01,
        logging_dir=log_dir,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        gradient_checkpointing=False,
        report_to='tensorboard',
        local_rank=args.local_rank,
        bf16=True,
        bf16_full_eval=True,
        log_level='debug' if args.debug else 'info'
    )
    # TOKENIZER
    print_rank_0("===== Loading tokenizer =====")
    print_rank_0(f"tokenizer : {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = 'right'
    print_rank_0("===== Loaded tokenizer =====")

    # MODEL
    print_rank_0("===== Loading model =====")
    print_rank_0(f"base model:, {args.model}")    
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        cache_dir=args.transformers_cache,
        )
    print_rank_0("===== Loaded model =====")
    if args.use_peft:
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=6, lora_alpha=32,
                                     lora_dropout=0.1)
    else:
        peft_config = None
    print_rank_0("Loaded peft config")

    dataset = load_dataset("Villekom/oa_dolly_15k_fi",split="train").train_test_split(test_size=0.1)
    print_rank_0("Loaded dataset")
    instruction_template = "<|user|>"
    response_template = "<|assistant|>"
    collator = DataCollatorForCompletionOnlyLM(instruction_template=instruction_template,
                                               response_template=response_template,
                                               tokenizer=tokenizer,
                                               mlm=False)
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        data_collator=collator,
        tokenizer=tokenizer,
        formatting_func=formatting_prompts_func,
        max_seq_length=2048,
        peft_config=peft_config
    )
    print_rank_0("Starting training")
    trainer.train()
    base_model_name = os.path.basename(args.model)
    save_directory = os.path.join(args.save_directory, base_model_name)
    if args.use_peft:
        trainer.model.save_pretrained(save_directory + "-peft")
    else:
        trainer.save_model(save_directory)
    #eval_results = trainer.evaluate(dataset['evaluation'])

    #print_rank_0('Training data', args.training_data)
    print_rank_0(f'Model:, {args.model}')
    print_rank_0(f'Learning rate:, {args.learning_rate}')
    print_rank_0(f'batch size:, {args.per_device_batch_size}')
    print_rank_0(f'Gradient accumulation steps:, {args.gradient_accumulation_steps}')
    print_rank_0(f"Evaluation results:, {eval_results['eval_loss']}")

def main(argv):
    args = argparser().parse_args(argv[1:])
    train_sft(args)

if __name__ == '__main__':
    sys.exit(main(sys.argv))