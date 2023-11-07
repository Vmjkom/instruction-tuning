#!/usr/bin/env python3
import os.path
import sys
import json
import torch
import numpy as np

from argparse import ArgumentParser
from logging import warning

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from evaluate import load
from utils import timed


DTYPE_MAP = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}

DMAP_CHOICES = ['auto', 'sequential']


def argparser():
    ap = ArgumentParser()
    ap.add_argument('--tokenizer', default=None)
    ap.add_argument('--lang', default="en", type=str)
    ap.add_argument('--max_prompts', default=10, type=int)
    ap.add_argument('--min_new_tokens', default=10, type=int)
    ap.add_argument('--max_new_tokens', default=200, type=int)
    ap.add_argument('--temperature', default=1.0, type=float)
    ap.add_argument('--num_return_sequences', default=1, type=int)
    ap.add_argument('--memory-usage', action='store_true')
    ap.add_argument('--show-devices', action='store_true')    
    ap.add_argument('--dtype', choices=DTYPE_MAP.keys(), default='bf16')
    ap.add_argument('--device-map', choices=DMAP_CHOICES, default='auto')
    ap.add_argument('--trust-remote-code', default=None, action='store_true')
    ap.add_argument('--transformers_cache',type=str, default="/scratch/project_2007628/transformers_cache")
    ap.add_argument('model')
    ap.add_argument('file', nargs='?')
    return ap


def report_memory_usage(message, out=sys.stderr):
    print(f'max memory allocation {message}:', file=out)
    total = 0
    for i in range(torch.cuda.device_count()):
        mem = torch.cuda.max_memory_allocated(i)
        print(f'  cuda:{i}: {mem/2**30:.1f}G', file=out)
        total += mem
    print(f'  TOTAL: {total/2**30:.1f}G', file=out)


@timed
def generate(prompts, tokenizer, model, args):
    generated_responses = []
    pipe = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=args.temperature,
        min_new_tokens=args.min_new_tokens,
        max_new_tokens=args.max_new_tokens,
        num_return_sequences=args.num_return_sequences,
        # repetition_penalty=1.2,
    )
    for prompt in prompts:
        prompt = prompt.rstrip('\n')
        generated = pipe(prompt)
        for g in generated:
            text = g['generated_text']
            print("-"*10, "Prompt:", prompt, "-"*10)
            text = text.replace(prompt, '', 1)
            print(text)
            print('-'*78)
            generated_responses.append(text)
    return generated_responses


@timed
def load_model(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map=args.device_map,
        torch_dtype=DTYPE_MAP[args.dtype],
        trust_remote_code=args.trust_remote_code,
        cache_dir=args.transformers_cache
    )
    return model


def check_devices(model, args):
    if args.show_devices:
        print(f'devices:', file=sys.stderr)
    for name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if args.show_devices:
                print(f'  {name}.{param_name}:{param.device}', file=sys.stderr)
            elif param.device.type != 'cuda':
                warning(f'{name}.{param_name} on device {param.device}')


def load_prompts(filepath, max_prompts=10, lang="en"):
    prompts = []
    responses = []
    print("filepath:", filepath)
    if os.path.splitext(filepath)[-1] == ".txt":
        prompts = open(filepath)
    elif os.path.splitext(filepath)[-1] == ".jsonl":
        test_data = [json.loads(line) for line in open(filepath)][:max_prompts]
        if "oasst" in filepath:
            text_col = "text"
            if lang == "en":
                text_col = "orig_text"
            for line in test_data:
                if line['role'] == "prompter":
                    prompt = "<|user|> " + line[text_col]
                    prompts.append(prompt)
        if "dolly" in filepath:
            prompt_col = "instruction"
            context_col = "context"
            response_col = "response"
            if lang == "en":
                prompt_col = "orig_instruction"
                context_col = "orig_context"
                response_col = "orig_response"
            for line in test_data:
                if not line[context_col] or line[context_col].isspace():
                    prompt = "<|user|> " + line[prompt_col]
                else:
                    prompt = line[context_col] + "\n<|user|> " + line[prompt_col]
                prompts.append(prompt.rstrip())
                responses.append(line[response_col])
    return prompts, responses

def compute_bertscore(references, predictions):
    bertscore = load("bertscore")
    results = bertscore.compute(predictions=predictions, references=references, lang="en")
    precision = np.mean(np.array(results['precision']))
    recall = np.mean(np.array(results['recall']))
    f1 = np.mean(np.array(results['f1']))
    print("F1:", f1)
    print("Precision:", precision)
    print("Recall:", recall)
    return results

def main(argv):
    args = argparser().parse_args(argv[1:])

    if args.tokenizer is None:
        args.tokenizer = args.model

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = load_model(args)

    if args.memory_usage:
        report_memory_usage('after model load')

    check_devices(model, args)
    
    if not args.file:
        generate(sys.stdin, tokenizer, model, args)
    else:
        prompts, responses = load_prompts(args.file, args.max_prompts, args.lang)
        generated = generate(prompts, tokenizer, model, args)
        print("prompts:", len(prompts))
        print("generated:", len(generated))
        results = compute_bertscore(references=prompts, predictions=generated)

    if args.memory_usage:
        report_memory_usage('after generation')


if __name__ == '__main__':
    sys.exit(main(sys.argv))
