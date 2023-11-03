import sys
import os
import torch
import numpy as np
from logging import warning
from datasets import DatasetDict, Dataset
from argparse import ArgumentParser

from transformers import (
    AutoTokenizer,
    TrainingArguments
)

from trl import (
    SFTTrainer,
    DataCollatorForCompletionOnlyLM
)

# custom classes
from utils import load_model, logits_argmax
from instruction_finetuning_datasets import read_data

model_max_length = 2048
user_token = "<|user|>"
assistant_token = "<|assistant|>"

def argparser():
    ap = ArgumentParser()
    ap.add_argument('--deepspeed_config', type=str, default="./ds-configs/oa_zero3_config_sft.json")
    ap.add_argument('--learning_rate', type=float, default=2e-5)
    ap.add_argument('--model', type=str)
    ap.add_argument('--tokenizer', type=str)
    ap.add_argument('--task', type=str, default="sft")
    ap.add_argument('--num_train_epochs', type=int, default=1)
    ap.add_argument('--per_device_batch_size', type=int, default=1)
    ap.add_argument('--output_dir', type=str, default="output")
    ap.add_argument('--gradient_accumulation_steps', type=int, default=4)
    ap.add_argument('--output_file', type=str)
    ap.add_argument('--training_data', type=str, default="oasst")
    ap.add_argument('--lang', type=str, default="fi")
    ap.add_argument('--local_rank', type=int)
    ap.add_argument('--use_lora', default=True, type=lambda x: (str(x).lower() == 'true'))
    ap.add_argument('--transformers_cache',type=str, default="/scratch/project_2007628/transformers_cache")
    ap.add_argument('--dropout',type=float, default=0.1)
    ap.add_argument('--prompt_structure', default=False, type=lambda x: (str(x).lower() == 'true'))
    return ap

def filter_by_length(datasetdict, max_length):
    for k in datasetdict:
        dataset = datasetdict[k]
        filtered = dataset.filter(lambda e: len(e['input_ids']) <= max_length)
        orig_length = len(dataset['input_ids'])
        filt_length = len(filtered['input_ids'])
        if filt_length < orig_length:
            warning(
                f'filtered {k} from {orig_length} to {filt_length} '
                f'({filt_length/orig_length:.1%}) by max_length {max_length}'
            )
            datasetdict[k] = filtered
    return datasetdict

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['prompt'])):
        if not example['context'] or example['context'].isspace():
            text = example['prompt']
        else:
            text = example['context'] + '\n' + example['prompt']
        output_texts.append(text)
    return output_texts


def train_sft(args):
    log_dir = './logs/'
    base_model_name = os.path.basename(args.model)
    output_dir = os.path.join("../models/sft_checkpoints/", base_model_name +
                              "-" + args.training_data +
                              "-" + args.lang +
                              "-" + str(args.num_train_epochs) + "epochs")
    print("Saving checkpoints to", output_dir)

    # This needs to be defined before model loading for deepspeed stage 3 to work correctly
    training_args = TrainingArguments(
        deepspeed=args.deepspeed_config,
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=100,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=1,
        log_on_each_node=False,
        logging_strategy="steps",
        logging_steps=10,
        num_train_epochs=args.num_train_epochs,
        save_strategy="epoch",
        save_steps=100,
        save_total_limit=1,
        disable_tqdm=False,
        weight_decay=0.01,
        logging_dir=log_dir,
        optim='adamw_hf',
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        report_to='tensorboard',
        bf16=True,
        half_precision_backend="cuda_amp",
        local_rank=args.local_rank,
    )

    # TOKENIZER
    print("===== Loading tokenizer =====")
    print("tokenizer :", args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    print("===== Loaded tokenizer =====")

    # MODEL
    print("===== Loading model =====")
    print("base model:", args.model)
    model = load_model(args.model, args.transformers_cache, args.use_lora)
    print("===== Loaded model =====")

    # # add special tokens if necessary
    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    # if tokenizer.sep_token is None:
    #     tokenizer.add_special_tokens({'sep_token': '<|endofprompt|>'})
    # model.resize_token_embeddings(len(tokenizer))

    if args.lang == "both":
        print("Combine fi and en datasets")
        train_data1 = read_data(args.training_data, split="train", lang="fi", task=args.task)
        train_data2 = read_data(args.training_data, split="train", lang="en", task=args.task)

        val_data1 = read_data(args.training_data, split="valid", lang="fi", task=args.task)
        val_data2 = read_data(args.training_data, split="valid", lang="en", task=args.task)

        eval_data1 = read_data(args.training_data, split="eval", lang="fi", task=args.task)
        eval_data2 = read_data(args.training_data, split="eval", lang="en", task=args.task)

        train_data = {'prompt': train_data1['prompt'] + train_data2['prompt'],
                      'context': train_data1['context'] + train_data2['context'],
                      'response': train_data1['response'] + train_data2['response']}
        train_data = Dataset.from_dict(train_data)

        val_data = {'prompt': val_data1['prompt'] + val_data2['prompt'],
                    'context': val_data1['context'] + val_data2['context'],
                    'response': val_data1['response'] + val_data2['response']}
        val_data = Dataset.from_dict(val_data)

        eval_data = {'prompt': eval_data1['prompt'] + eval_data2['prompt'],
                     'context': eval_data1['context'] + eval_data2['context'],
                     'response': eval_data1['response'] + eval_data2['response']}
        eval_data = Dataset.from_dict(eval_data)

    else:
        train_data = read_data(args.training_data, split="train", lang=args.lang, task=args.task)
        val_data = read_data(args.training_data, split="valid", lang=args.lang, task=args.task)
        eval_data = read_data(args.training_data, split="eval", lang=args.lang, task=args.task)

    print("Size of training data", len(train_data))
    print("Size of validation data", len(val_data))
    print("Size of evaluation data", len(eval_data))

    dataset = DatasetDict({
        'train': train_data,
        'validation': val_data,
        'evaluation': eval_data,
    })

    print("Size of training data", len(dataset['train']))

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
        eval_dataset=dataset['validation'],
        data_collator=collator,
        tokenizer=tokenizer,
        formatting_func=formatting_prompts_func,
        #preprocess_logits_for_metrics=logits_argmax,
    )

    trainer.train()
    base_model_name = os.path.basename(args.model)
    save_directory = os.path.join("../models/sft_finetuned/", base_model_name + "-" + args.training_data + "-" + args.lang)
    if args.use_lora:
        trainer.model.save_pretrained(save_directory + "-lora")
    else:
        trainer.save_model(save_directory)
    eval_results = trainer.evaluate(dataset['evaluation'])

    print('Training data', args.training_data)
    print('Model:', args.model)
    print('Learning rate:', args.learning_rate)
    print('batch size:', args.per_device_batch_size)
    print('Gradient accumulation steps:', args.gradient_accumulation_steps)
    print('Evaluation results:', eval_results['eval_loss'])

def main(argv):
    args = argparser().parse_args(argv[1:])
    train_sft(args)

if __name__ == '__main__':
    sys.exit(main(sys.argv))