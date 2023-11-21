import sys
import os
import torch
import numpy as np
from logging import warning
from datasets import DatasetDict
from argparse import ArgumentParser
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    LoraConfig,
    TaskType
)

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)


# custom classes
from instruction_finetuning_datasets import read_data_sft

import logging
torch.cuda.empty_cache()
model_max_length = 2048
user_token = "<|user|>"
assistant_token = "<|assistant|>"
chatml_start_token = "<|im_start|>"
chatml_end_token = "<|im_end|>"

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
    ap.add_argument('--eval_task', type=str, default="arc_challenge")
    ap.add_argument('--data_split', type=int, default=1)
    ap.add_argument('--lang', type=str, default="fi")
    ap.add_argument('--local_rank', type=int)
    ap.add_argument('--use_lora', default=True, type=lambda x: (str(x).lower() == 'true'))
    ap.add_argument('--chatml_format', default=False, type=lambda x: (str(x).lower() == 'true'))
    ap.add_argument('--transformers_cache',type=str, default="/scratch/project_462000319/transformers_cache")
    ap.add_argument('--dropout',type=float, default=0.1)
    ap.add_argument('--prompt_structure', default=False, type=lambda x: (str(x).lower() == 'true'))
    return ap

def load_model(model_name, transformers_cache, use_lora=False, ignore_bias_buffers=False):
    print("load_model")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=transformers_cache,
        num_labels=1,
        torch_dtype=torch.bfloat16
    )
    # print(model)
    if ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]
    if use_lora is True:
        print("Using lora")
        model.enable_input_require_grads()
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32,
                                     lora_dropout=0.1,
                                     target_modules=['query_key_value'])  # Parameter values from Sampo's script
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        print("Loaded lora model")
    return model

def logits_argmax(logits):
    # https://github.com/huggingface/transformers/issues/15466
    return logits.argmax(axis=-1)

class PromptMaskingDataCollator(DataCollatorForLanguageModeling):    # Sampo's script
    def __call__(self, features, return_tensors=None):
        data = super().__call__(features, return_tensors)
        # if tokenizer has assistant_token, use it to signal prompt boundary, else use <|endofprompt|>
        if assistant_token in self.tokenizer.additional_special_tokens:
            assistant_id = self.tokenizer(assistant_token)['input_ids'][0]
        elif chatml_start_token in self.tokenizer.additional_special_tokens:
            assistant_id = self.tokenizer(chatml_start_token)['input_ids'][0]
        else:
            assistant_id = -100
        # print("assistant_id:", assistant_id)
        for i in range(len(data['labels'])):
            assistant_indices = np.where(data['labels'][i] == assistant_id)[0]
            # print("labels:", data['labels'][i])
            # print("decoded:", self.tokenizer.decode(data['input_ids'][i]))
            # print("assistant_indices:", assistant_indices)
            # print("last assistant index:", assistant_indices[-1])
            if len(assistant_indices) > 0:
                data['labels'][i, :assistant_indices[-1]] = -100
            else:
                warning('missing assistant_token in labels')
            # print("labels:", data['labels'][i])
            # print("decoded labels:", self.tokenizer.decode(data['labels'][i][assistant_indices[-1]:]))
            # print("-"*100)
        return data

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

def preprocess_sft(data, tokenizer):   # Sampo's script -- modified with prompt structure
    prompts = data['prompt']
    contexts = data['context']
    responses = data['response']
    end_of_prompt = tokenizer.pad_token # tokenizer.sep_token
    end_of_text = tokenizer.eos_token
    combined = []
    for prompt, context, response in zip(prompts, contexts, responses):
        if not context or context.isspace():
            input_i = prompt
        else:
            input_i = context + '\n' + prompt
        # FinGPT needs end_of_prompt to signal prompt boundary, Poro uses assistant_token or chatml_start_token
        if (assistant_token in tokenizer.additional_special_tokens) or (chatml_start_token in tokenizer.additional_special_tokens):
            combined_line = input_i + '\n' + response + end_of_text
        else:
            combined_line = input_i + end_of_prompt + '\n' + response + end_of_text
        # print("combined_line:", combined_line)
        combined.append(combined_line)
    # Truncation would be problematic for this task
    tokenized = tokenizer(combined, truncation=True)
    return tokenized


def train_sft(args):
    log_dir = './logs/'
    base_model_name = os.path.basename(args.model)
    if args.chatml_format:
        if args.training_data == "eval_tasks":
            output_dir = os.path.join("../../models/sft_checkpoints/", base_model_name + 
                                "-chatml" +
                                "-" + args.eval_task + 
                                "-" + str(args.data_split) +
                                "-" + str(args.num_train_epochs) + "epochs")
        else:
            output_dir = os.path.join("../../models/sft_checkpoints/", base_model_name + 
                                    "-chatml" +
                                    "-" + args.training_data + 
                                    "-" + args.lang +
                                    "-" + str(args.num_train_epochs) + "epochs")
    else:
        if args.training_data == "eval_tasks":
            output_dir = os.path.join("../../models/sft_checkpoints/", base_model_name + 
                                      "-" + args.eval_task + 
                                      "-" + str(args.data_split) +
                                      "-" + str(args.num_train_epochs) + "epochs")
        else:
            output_dir = os.path.join("../../models/sft_checkpoints/", base_model_name + 
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
        #warmup_steps=300,
        warmup_ratio=0.1,
        half_precision_backend="cuda_amp",
        local_rank=args.local_rank,
    )

    # print(training_args)

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

    # add special tokens if necessary
    # special tokens are needed for FinGPT models but not for Poro and beyond
    # if assistant_token not in tokenizer.additional_special_tokens:
    # print("Adding special tokens")
    # if tokenizer.pad_token is None:
    #     tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    # if tokenizer.sep_token is None:
    #      tokenizer.add_special_tokens({'sep_token': '<|endofprompt|>'})
    # model.resize_token_embeddings(len(tokenizer))
    # resize_token_embeddings(model, len(tokenizer))

    print("Loading data for SFT")
    train_data = read_data_sft(args.training_data, split="train", lang=args.lang, chatml_format=args.chatml_format, eval_task=args.eval_task)
    val_data = read_data_sft(args.training_data, split="valid", lang=args.lang, chatml_format=args.chatml_format, eval_task=args.eval_task)
    eval_data = read_data_sft(args.training_data, split="eval", lang=args.lang, chatml_format=args.chatml_format, eval_task=args.eval_task)

    print("Size of training data", len(train_data))
    print("Size of validation data", len(val_data))
    print("Size of evaluation data", len(eval_data))

    dataset = DatasetDict({
        'train': train_data,
        'validation': val_data,
        'evaluation': eval_data,
    })

    dataset = dataset.map(
        lambda d: preprocess_sft(d, tokenizer),
        batched=True
    )

    print("Filtering by length")
    dataset = filter_by_length(dataset, model_max_length)
    data_collator = PromptMaskingDataCollator(
        tokenizer=tokenizer,
        mlm=False
    )

    print("Size of training data", len(dataset['train']))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=data_collator,
        tokenizer=tokenizer,
        preprocess_logits_for_metrics=logits_argmax,
    )

    trainer.train()
    base_model_name = os.path.basename(args.model)
    save_directory = os.path.join("../../models/sft_finetuned/", base_model_name + "-" + args.training_data + "-" + args.lang)
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
    print('Save directory:', save_directory)


def main(argv):
    args = argparser().parse_args(argv[1:])
    if args.task == "sft":
        train_sft(args)
    else:
        print(args.task, "is not supported! The only supported tasks for sft.")

if __name__ == '__main__':
    sys.exit(main(sys.argv))
