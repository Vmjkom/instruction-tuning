import sys
import os
import torch
import numpy as np
from instruction_finetuning_datasets import read_data
from logging import warning
from datasets import DatasetDict
from argparse import ArgumentParser
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    # BloomForCausalLM,
    # BloomTokenizerFast,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import logging
torch.cuda.empty_cache()
model_max_length = 2048

def argparser():
    ap = ArgumentParser()
    ap.add_argument('--deepspeed_config', type=str, default="./ds-configs/oa_zero3_config_sft.json")
    # ap.add_argument('--learning_rate', type=float, default=6e-6)
    # ap.add_argument('--model', type=str)
    # ap.add_argument('--num_train_epochs', type=int, default=1)
    # ap.add_argument('--per_device_batch_size', type=int, default=1)
    # ap.add_argument('--output_dir', type=str, default="output")
    # ap.add_argument('--gradient_accumulation_steps', type=int, default=1)
    # ap.add_argument('--output_file', type=str)
    ap.add_argument('--training_data', type=str, default="dolly")
    ap.add_argument('--local_rank', type=int)
    ap.add_argument('--use_lora', default=True, type=lambda x: (str(x).lower() == 'true'))
    ap.add_argument('--transformers_cache',type=str, default="../../transformers_cache")
    ap.add_argument('--dropout',type=float, default=0.1)
    ap.add_argument('--prompt_structure', default=False, type=lambda x: (str(x).lower() == 'true'))
    return ap

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

    # This needs to be defined before model loading for deepspeed stage 3 to work correctly
    training_args = TrainingArguments(
        deepspeed=args.deepspeed_config,
        output_dir="../sft_tuning",
        evaluation_strategy="steps",
        eval_steps=100,
        learning_rate=1e-5,
        per_device_train_batch_size=2, # trying to prevent GPU OOM
        gradient_accumulation_steps=8, # trying to prevent GPU OOM
        per_device_eval_batch_size=2, # trying to prevent GPU OOM
        log_on_each_node=False,
        logging_strategy="steps",
        logging_steps=1,
        num_train_epochs=2,
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
    # tokenizer_path = "TurkuNLP/gpt3-finnish-xl"
    tokenizer_path = "/scratch/project_2007628/tokenizers/tokenizer_v6_fixed_fin"
    print("tokenizer_path:", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print("===== Loaded tokenizer =====")

    # MODEL
    print("===== Loading model =====")
    # model_path = "TurkuNLP/gpt3-finnish-xl"
    model_path = "../../33B_torch_step67824_bfloat16"
    print("model_path:", model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, cache_dir=args.transformers_cache, num_labels=1, torch_dtype=torch.bfloat16
    )
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
    
    train_data = read_data(args.training_data, split="train")
    val_data = read_data(args.training_data, split="valid")
    eval_data = read_data(args.training_data, split="eval")

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
    )

    trainer.train()
    
    if args.use_lora:
        trainer.model.save_pretrained("../sft_tuning/poro_sft_lora")
    else:
        trainer.save_model(os.path.join("../sft_tuning/poro_sft_lora"))

    eval_results = trainer.evaluate(dataset['evaluation'])

    print('Training data', args.training_data)
    print('Model:', args.model)
    print('Learning rate:', args.learning_rate)
    print('batch size:', args.per_device_batch_size)
    print('Gradient accumulation steps:', args.gradient_accumulation_steps)
    print('Evaluation results:', eval_results['eval_loss'])

if __name__ == '__main__':
    sys.exit(main(sys.argv))