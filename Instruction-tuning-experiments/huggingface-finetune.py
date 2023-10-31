import sys
import os
import torch
import numpy as np
from logging import warning
from datasets import DatasetDict, Dataset
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

from trl import (
    DPOTrainer,
    create_reference_model
)

# custom classes
from instruction_finetuning_datasets import read_data

import logging
torch.cuda.empty_cache()
model_max_length = 2048

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

def load_model(model_name, transformers_cache, use_lora=False, ignore_bias_buffers=False, task="sft"):
    print("load_model")
    if task == "dpo":
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=transformers_cache,
            num_labels=1,
            #torch_dtype=torch.bfloat16
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            cache_dir=transformers_cache,
            num_labels=1,
            torch_dtype=torch.bfloat16
        )

    if ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    if use_lora is True:
        print("Using lora")
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32,
                                     lora_dropout=0.1)  # Parameter values from Sampo's script
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

        end_of_prompt_id = self.tokenizer.sep_token_id
        for i in range(len(data['labels'])):
            eop_indices = np.where(data['labels'][i] == end_of_prompt_id)[0]
            # print("eop_indices:", eop_indices)
            if len(eop_indices) > 0:
                # TODO this should really be eop_indices[0]+1 but that
                # would mask the eop which would mess up the current
                # logic for separating the prompt from the output
                data['labels'][i,:eop_indices[0]] = -100
            else:
                warning('missing eop in labels')
            # print(data['labels'])
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

def preprocess_sft(data, tokenizer, prompt_structure):   # Sampo's script -- modified with prompt structure
    prompts = data['prompt']
    contexts = data['context']
    responses = data['response']
    end_of_prompt = tokenizer.sep_token
    end_of_text = tokenizer.eos_token
    user_token = "<|user|>"
    assistant_token = "<|assistant|>"
    combined = []
    for prompt, context, response in zip(prompts, contexts, responses):
        if prompt_structure:
            if not context or context.isspace():
                # input_i = "Alla on kysymys sairaudesta. " + "Kirjoita vastaus, vastaa kysymykseen.\n\n" + "### Ohje:\n" + prompt + "### Vastaus: \n"
                input_i = "Alla on kysymys sairaudesta. " + "Kirjoita vastaus, vastaa kysymykseen.\n\n" + user_token + "\n" + prompt + "\n" + assistant_token + "\n"
            else:
                # input_i = "Alla on ohje, joka kuvaa tehtävää, ja johon on liitetty kontekstia tarjoava syöte. " + "Kirjoita vastaus, joka täyttää ohjeen mukaisen pyynnön.\n\n" + "### Ohje:\n" + prompt + "### Konteksti: \n" + context + "\n\n### Vastaus: \n"
                input_i = "Alla on ohje, joka kuvaa tehtävää, ja johon on liitetty kontekstia tarjoava syöte. " + "Kirjoita vastaus, joka täyttää ohjeen mukaisen pyynnön.\n\n" + user_token + "\n" + prompt + "### Konteksti: \n" + context + "\n\n" + assistant_token + "\n"
        else:
            if not context or context.isspace():
                input_i = prompt
            else:
                input_i = context + '\n' + prompt
        # combined.append(input_i + end_of_prompt + response + end_of_text)
        # combined_line = user_token + '\n' + input_i + end_of_prompt + assistant_token + '\n' + response + end_of_text
        combined_line = input_i + end_of_prompt + '\n' + response
        # print(combined_line)
        # print("="*50)
        combined.append(combined_line)
    # Truncation would be problematic for this task
    tokenized = tokenizer(combined, truncation=True)
    return tokenized

def preprocess_dpo(data):   # Sampo's script -- modified with prompt structure
    prompts = data['prompt']
    contexts = data['context']
    accepted = data['accepted_response']
    rejected = data['rejected_response']
    # user_token = "<|user|>"
    # assistant_token = "<|assistant|>"
    dpo_dataset = {
        "prompt": [],
        "chosen": [],
        "rejected": []
    }
    for prompt, context, accepted, rejected in zip(prompts, contexts, accepted, rejected):
        # if not context or context.isspace():
        #     input_prompt = prompt
        # else:
        #     input_prompt = context + prompt
        # dpo_dataset["prompt"].append(input_prompt)
        dpo_dataset["prompt"].append(prompt)
        dpo_dataset["chosen"].append(accepted)
        dpo_dataset["rejected"].append(rejected)
    return dpo_dataset


def train_sft(args):
    log_dir = './logs/'
    base_model_name = os.path.basename(args.model)
    output_dir = os.path.join("../models/sft_checkpoints/", base_model_name + "-" + args.training_data + "-" + args.lang)
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
        num_train_epochs=1,
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
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({'sep_token': '<|endofprompt|>'})
    model.resize_token_embeddings(len(tokenizer))

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

    dataset = dataset.map(
        lambda d: preprocess_sft(d, tokenizer, args.prompt_structure),
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

    # print PeftModel param dimensions
    # for name, param in trainer.model.named_parameters():
    #     if "lora" in name:
    #         print(name, "---", param.shape)

    trainer.train()
    base_model_name = os.path.basename(args.model)
    save_directory = os.path.join("../models/sft_finetuned/", base_model_name + "-" + args.training_data + "-" + args.lang)
    if args.use_lora:
        trainer.model.save_pretrained(save_directory + "-lora")
        # print PeftModel param dimensions
        # for name, param in trainer.model.named_parameters():
        #     if "lora" in name:
        #         print(name, "---", param.shape)
    else:
        trainer.save_model(save_directory)
    eval_results = trainer.evaluate(dataset['evaluation'])

    print('Training data', args.training_data)
    print('Model:', args.model)
    print('Learning rate:', args.learning_rate)
    print('batch size:', args.per_device_batch_size)
    print('Gradient accumulation steps:', args.gradient_accumulation_steps)
    print('Evaluation results:', eval_results['eval_loss'])


def train_dpo(args):
    # https://github.com/huggingface/trl/blob/main/examples/scripts/dpo.py
    log_dir = './logs/'
    base_model_name = os.path.basename(args.model)
    output_dir = os.path.join("../models/dpo_checkpoints/", base_model_name + "-" + args.training_data + "-" + args.lang)
    print("Saving checkpoints to", output_dir)

    # This needs to be defined before model loading for deepspeed stage 3 to work correctly
    # 4. initialize training arguments:
    training_args = TrainingArguments(
        deepspeed="./ds-configs/oa_deepspeed_rl_zero3.json",
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=100,
        learning_rate=1e-3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        log_on_each_node=False,
        logging_strategy="steps",
        logging_steps=10,
        max_steps=1000,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=1,
        disable_tqdm=False,
        logging_dir=log_dir,
        optim='rmsprop',
        warmup_steps=150,
        gradient_checkpointing=True,
        report_to='tensorboard',
        # fp16=True,
        bf16=True,
        # half_precision_backend="cuda_amp",
        local_rank=args.local_rank,
    )

    # TOKENIZER
    print("tokenizer :", args.tokenizer)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    # MODEL
    # 1. load a pretrained model
    print("base model:", args.model)
    print("=== Loading model ===")
    model = load_model(args.model, args.transformers_cache, args.use_lora, task=args.task)

    print("=== Loading model_ref ===")
    model_ref = load_model(args.model, args.transformers_cache, args.use_lora, task=args.task)
    # Error msg: DeepSpeed ZeRO-3 is enabled and is not compatible with `create_reference_model()
    # model_ref = create_reference_model(model, num_shared_layers=6)

    # add special tokens if necessary
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({'sep_token': '<|endofprompt|>'})
    model.resize_token_embeddings(len(tokenizer))

    # 2. Load training dataset and 3. Load eval data
    print("load train_data")
    train_data = read_data(args.training_data, split="train", lang=args.lang, task=args.task)
    print("load val_data")
    val_data = read_data(args.training_data, split="valid", lang=args.lang, task=args.task)
    print("load eval_data")
    eval_data = read_data(args.training_data, split="eval", lang=args.lang, task=args.task)

    print("Size of training data", len(train_data))
    print("Size of validation data", len(val_data))
    print("Size of evaluation data", len(eval_data))

    dataset = DatasetDict({
        'train': train_data,
        'validation': val_data,
        'evaluation': eval_data,
    })

    dataset = dataset.map(
        lambda d: preprocess_dpo(d),
        batched=True
    )

    print("Size of training data", len(dataset['train']))

    # 5. initialize the DPO trainer
    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=0.2,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer,
        max_length=512,
        max_target_length=128,
        max_prompt_length=128,
        generate_during_eval=True,
    )

    # 6. train
    dpo_trainer.train()

    base_model_name = os.path.basename(args.model)
    save_directory = os.path.join("../models/dpo_finetuned/", base_model_name + "-" + args.training_data + "-" + args.lang)
    if args.use_lora:
        dpo_trainer.model.save_pretrained(save_directory + "-lora")
        # print PeftModel param dimensions
        for name, param in dpo_trainer.model.named_parameters():
            if "lora" in name:
                print(name, "---", param.shape)
    else:
        dpo_trainer.save_model(save_directory)
    eval_results = dpo_trainer.evaluate(dataset['evaluation'])

    print('Training data', args.training_data)
    print('Model:', args.model)
    print('Learning rate:', args.learning_rate)
    print('batch size:', args.per_device_batch_size)
    print('Gradient accumulation steps:', args.gradient_accumulation_steps)
    print('Evaluation results:', eval_results['eval_loss'])


def main(argv):
    args = argparser().parse_args(argv[1:])
    if args.task == "sft":
        train_sft(args)
    elif args.task == "dpo":
        train_dpo(args)
    else:
        print(args.task, "is not supported! The only supported tasks for sft or dpo.")

if __name__ == '__main__':
    sys.exit(main(sys.argv))