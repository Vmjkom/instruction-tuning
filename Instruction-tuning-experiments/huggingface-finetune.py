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

from instruction_finetuning_datasets import read_data

import logging
torch.cuda.empty_cache()
model_max_length = 2048

def argparser():
    ap = ArgumentParser()
    ap.add_argument('--deepspeed_config', type=str, default="./ds-configs/oa_zero3_config_sft.json")
    ap.add_argument('--learning_rate', type=float, default=1e-5)
    ap.add_argument('--model', type=str)
    ap.add_argument('--tokenizer', type=str)
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

def load_model(model_name, transformers_cache):
    print("load_model")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=transformers_cache,
        num_labels=1,
        torch_dtype=torch.bfloat16
    )
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
            if len(eop_indices) > 0:
                # TODO this should really be eop_indices[0]+1 but that
                # would mask the eop which would mess up the current
                # logic for separating the prompt from the output
                data['labels'][i,:eop_indices[0]] = -100
            else:
                warning('missing eop in labels')

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

def preprocess(data, tokenizer, prompt_structure):   # Sampo's script -- modified with prompt structure
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
        combined_line = user_token + '\n' + input_i + end_of_prompt + assistant_token + '\n' + response + end_of_text
        combined.append(combined_line)

    # Truncation would be problematic for this task
    tokenized = tokenizer(combined, truncation=True)

    return tokenized

def main(argv):
    args = argparser().parse_args(argv[1:])
    log_dir = './logs/'
    base_model_name = os.path.basename(args.model)
    output_dir = os.path.join("../models/checkpoints/", base_model_name + "-" + args.training_data + "-" + args.lang)
    print("Saving checkpoints to", output_dir)

    # This needs to be defined before model loading for deepspeed stage 3 to work correctly
    training_args = TrainingArguments(
        deepspeed=args.deepspeed_config,
        output_dir=output_dir,
        evaluation_strategy="steps",
        eval_steps=100,
        learning_rate=1e-5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        per_device_eval_batch_size=1,
        log_on_each_node=False,
        logging_strategy="steps",
        logging_steps=10,
        num_train_epochs=1,
        save_strategy="steps",
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
    model = load_model(args.model, args.transformers_cache)
    print("===== Loaded model =====")

    if args.use_lora:
        print("Using lora")
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1) # Parameter values from Sampo's script
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        print("===== Loaded lora model =====")

    # add special tokens if necessary
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    if tokenizer.sep_token is None:
        tokenizer.add_special_tokens({'sep_token': '<|endofprompt|>'})
    model.resize_token_embeddings(len(tokenizer))
    
    train_data = read_data(args.training_data, split="train", lang=args.lang)
    val_data = read_data(args.training_data, split="valid", lang=args.lang)
    eval_data = read_data(args.training_data, split="eval", lang=args.lang)

    print("Size of training data", len(train_data))
    print("Size of validation data", len(val_data))
    print("Size of evaluation data", len(eval_data))

    dataset = DatasetDict({
            'train': train_data,
            'validation': val_data,
            'evaluation': eval_data,
    })

    dataset = dataset.map(
         lambda d: preprocess(d, tokenizer, args.prompt_structure),
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
    for name, param in trainer.model.named_parameters():
        if "lora" in name:
            print(name, "---", param.shape)

    trainer.train()
    base_model_name = os.path.basename(args.model)
    save_directory = os.path.join("../models/finetuned/", base_model_name + "-" + args.training_data + "-" + args.lang)
    if args.use_lora:
        trainer.model.save_pretrained(save_directory + "-lora")
        # print PeftModel param dimensions
        for name, param in trainer.model.named_parameters():
            if "lora" in name:
                print(name, "---", param.shape)
    else:
        trainer.save_model(save_directory)
    eval_results = trainer.evaluate(dataset['evaluation'])

    print('Training data', args.training_data)
    print('Model:', args.model)
    print('Learning rate:', args.learning_rate)
    print('batch size:', args.per_device_batch_size)
    print('Gradient accumulation steps:', args.gradient_accumulation_steps)
    print('Evaluation results:', eval_results['eval_loss'])

if __name__ == '__main__':
    sys.exit(main(sys.argv))