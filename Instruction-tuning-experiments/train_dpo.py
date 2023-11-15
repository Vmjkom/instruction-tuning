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


from trl import (
    DPOTrainer,
    create_reference_model
)

# custom classes
from instruction_finetuning_datasets import read_data_dpo


def train_dpo(args):
    # https://github.com/huggingface/trl/blob/main/examples/scripts/dpo.py
    log_dir = './logs/'
    base_model_name = os.path.basename(args.model)
    output_dir = os.path.join("../../models/dpo_checkpoints/", base_model_name + "-" + args.training_data + "-" + args.lang)
    print("Saving checkpoints to", output_dir)

    # This needs to be defined before model loading for deepspeed stage 3 to work correctly
    # 4. initialize training arguments:
    training_args = TrainingArguments(
        deepspeed="./ds-configs/oa_deepspeed_rl_zero3.json",
        remove_unused_columns=False,
        output_dir=output_dir,
        logging_dir=log_dir,
        evaluation_strategy="steps",
        eval_steps=50,
        num_train_epochs=args.num_train_epochs,
        save_strategy="epoch",
        save_total_limit=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        log_on_each_node=False,
        logging_strategy="steps",
        logging_steps=10,
        logging_first_step=True,
        report_to='tensorboard',
        learning_rate=args.learning_rate,
        optim="rmsprop",
        warmup_steps=100,
        bf16=True,
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
    # create_reference_model error: DeepSpeed ZeRO-3 is enabled and is not compatible with `create_reference_model()
    # model_ref = create_reference_model(model, num_shared_layers=6)

    # add special tokens if necessary
    # if assistant_token not in tokenizer.additional_special_tokens:
    #     print("Adding special tokens")
    #     if tokenizer.pad_token is None:
    #         tokenizer.add_special_tokens({'pad_token': '<|pad|>'})
    #     if tokenizer.sep_token is None:
    #         tokenizer.add_special_tokens({'sep_token': '<|endofprompt|>'})
    #     model.resize_token_embeddings(len(tokenizer))

    # 2-3. Load training/valid/eval datasets
    print("load train_data")
    train_data = read_data_dpo(args.training_data, split="train", lang=args.lang)
    print("load val_data")
    val_data = read_data_dpo(args.training_data, split="valid", lang=args.lang)
    print("load eval_data")
    eval_data = read_data_dpo(args.training_data, split="eval", lang=args.lang)

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
        model=model,
        ref_model=model_ref,
        args=training_args,
        beta=0.1,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        tokenizer=tokenizer,
        max_length=model_max_length,
        max_target_length=256,
        max_prompt_length=256,
        padding_value=tokenizer.pad_token_id
        # generate_during_eval=True,
    )

    # 6. train
    dpo_trainer.train()

    base_model_name = os.path.basename(args.model)
    save_directory = os.path.join("../../models/dpo_finetuned/", base_model_name + "-" + args.training_data + "-" + args.lang)
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
    print('Save directory:', save_directory)