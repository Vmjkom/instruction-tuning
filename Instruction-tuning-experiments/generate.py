#!/usr/bin/env python3
import os.path
import sys
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from argparse import ArgumentParser
from logging import warning

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from evaluate import load
from utils import timed
from collections import Counter

# for language detection
import fasttext

# for toxicity detection
import transformers

DTYPE_MAP = {
    'fp32': torch.float32,
    'fp16': torch.float16,
    'bf16': torch.bfloat16,
}

DMAP_CHOICES = ['auto', 'sequential']

user_token = "<|user|>"
assistant_token = "<|assistant|>"
chatml_start_token = "<|im_start|>"
chatml_end_token = "<|im_end|>"

def argparser():
    ap = ArgumentParser()
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
    ap.add_argument('--transformers_cache',type=str, default="/scratch/project_462000319/transformers_cache")
    ap.add_argument('--model', type=str)
    ap.add_argument('--file', type=str)
    ap.add_argument('--tokenizer', type=str)
    ap.add_argument('--eval_only', default=False, type=lambda x: (str(x).lower() == 'true'))
    ap.add_argument('--base_model', default=False, type=lambda x: (str(x).lower() == 'true'))
    ap.add_argument('--chatml_format', default=False, type=lambda x: (str(x).lower() == 'true'))
    ap.add_argument('--detect_lang', default=True, type=lambda x: (str(x).lower() == 'true'))
    ap.add_argument('--detect_toxicity', default=True, type=lambda x: (str(x).lower() == 'true'))
    ap.add_argument('--output_file', type=str, default=None)
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
def generate(prompts, tokenizer, model, args, responses):
    bad_words_ids = tokenizer.encode(['<NAME>', ' <NAME>'])
    generated_responses = []

    if args.eval_only:
        pipe = pipeline(
            'text-generation',
            model=model,
            tokenizer=tokenizer,
            do_sample=True,
            max_new_tokens=100,
            min_new_tokens=10,
            temperature=0.5,
            repetition_penalty=1.1,
            bad_words_ids=[[word] for word in bad_words_ids],
            num_return_sequences=args.num_return_sequences,
            batch_size=4
        )

        prompts = [prompt.rstrip('\n') for prompt in prompts]
        generated = pipe(prompts)
        for index in range(len(generated)):
            for g in generated[index]:
                text = g['generated_text']
                text = text.replace(prompt, '', 1)
                generated_responses.append(text)
    else:    
        pipe = pipeline(
            'text-generation',
            model=model,
            tokenizer=tokenizer,
            do_sample=True,
            max_new_tokens=100,
            min_new_tokens=10,
            temperature=0.5,
            repetition_penalty=1.1,
            bad_words_ids=[[word] for word in bad_words_ids],
            num_return_sequences=args.num_return_sequences,
        )

        for i, prompt in enumerate(prompts):
            prompt = prompt.rstrip('\n')
            true_response = responses[i]
            generated = pipe(prompt)
            for g in generated:
                print("-"*10, "PROMPT:", prompt, "-"*10)
                text = g['generated_text']
                text = text.replace(prompt, '', 1)
                print("RESPONSE:", text)
                # print("TRUE RESPONSE:", true_response)
                generated_responses.append(text)
    return generated_responses


@timed
def load_model(args):
    print("Loading model:", args.model)
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


def read_oasst_sft(path, lang='fi', chatml_format=False):
    # end_of_text = tokenizer.eos_token
    if lang == 'fi':
        text_col = "text"
    else:
        text_col = "orig_text"
    path = Path(path)
    with open(path, 'rb') as f:
        oasst_dict = list(f)
    questions_dict = {}
    context_wq_dict = {}
    context_return = []
    answers_return = []
    questions_return = []
    for index, json_str in enumerate(oasst_dict):
        result = json.loads(json_str)
        if result["role"] == "prompter":
            if chatml_format:
                question_combined = chatml_start_token + "user\n" + result[text_col] + chatml_end_token
            else:
                question_combined = user_token + result[text_col]
            questions_dict[result["message_id"]] = question_combined
            context_wq_dict[result["message_id"]] = " "
            if result["parent_id"]:
                try:
                    context_wq_dict[result["message_id"]] = context_wq_dict[result["parent_id"]]
                except:
                    context_wq_dict[result["message_id"]] = " "
        elif result["role"] == "assistant":
            try:
                questions_return.append(questions_dict[result["parent_id"]])
            except:
                continue
            if chatml_format:
                answer_combined = chatml_start_token + "assistant\n" + result[text_col] + chatml_end_token
            else:
                answer_combined = assistant_token + result[text_col]
            answers_return.append(answer_combined)
            if context_wq_dict[result["parent_id"]]:
                context_return.append(context_wq_dict[result["parent_id"]])
                context_wq_dict[result["message_id"]] = context_wq_dict[result["parent_id"]] + questions_dict[
                    result["parent_id"]] + answer_combined
            else:
                context_wq_dict[result["message_id"]] = questions_dict[result["parent_id"]] + "\n" + answer_combined
            context_wq_dict[result["message_id"]] = context_wq_dict[result["parent_id"]] + "\n" + questions_dict[
                result["parent_id"]] + "\n" + answer_combined
    return questions_return, context_return, answers_return

def load_prompts(filepath, max_prompts=10, lang="en", base_model=False, chatml_format=False):
    prompts = []
    responses = []
    instruction = "Vastaa kysymkseen suomeksi."
    if lang == "en":
        instruction = "Answer the question in English."
    print("filepath:", filepath)
    if os.path.splitext(filepath)[-1] == ".txt":
        prompts = open(filepath)
    elif os.path.splitext(filepath)[-1] == ".jsonl":
        if "oasst" in filepath:
            questions, contexts, answers = read_oasst_sft(filepath, lang=lang, chatml_format=chatml_format)
            if max_prompts <= 0:
                max_prompts = len(questions)
            for index in range(max_prompts):
                prompt = contexts[index] + "\n" + questions[index] + "\n" + assistant_token
                prompts.append(prompt)
                responses.append(answers[index])
        if "dolly" in filepath:
            if max_prompts > 0:
                test_data = [json.loads(line) for line in open(filepath)][:max_prompts]
            else:
                test_data = [json.loads(line) for line in open(filepath)]
            prompt_col = "instruction"
            context_col = "context"
            response_col = "response"
            if lang == "en":
                prompt_col = "orig_instruction"
                context_col = "orig_context"
                response_col = "orig_response"
            for line in test_data:
                if not line[context_col] or line[context_col].isspace():
                    if base_model:
                        prompt = line[prompt_col]
                    elif chatml_format:
                        prompt = chatml_start_token + "user\n" + line[prompt_col] + chatml_end_token + "\n" + chatml_start_token + "assistant" + "\n"
                    else:
                        prompt = user_token + " " + line[prompt_col] + "\n" + assistant_token
                else:
                    if base_model:
                        prompt = line[context_col] + "\n" + line[prompt_col] 
                    elif chatml_format:
                        prompt = chatml_start_token + "user\n" + line[context_col] + "\n" + line[prompt_col] + chatml_end_token + "\n" + chatml_start_token + "assistant" + "\n"
                    else:
                        prompt = line[context_col] + "\n" + user_token + line[prompt_col] + "\n" + assistant_token
                prompts.append(prompt.rstrip())
                if chatml_format:
                    response = line[response_col] + chatml_end_token
                else:
                    response = line[response_col]
                responses.append(response)
    return prompts, responses

def compute_bertscore(references, predictions, lang):
    bertscore = load("bertscore")
    if lang == "fi":
        # load Finnish BERT to evaluate Finnish text
        results = bertscore.compute(predictions=predictions, references=references, 
                                    model_type="TurkuNLP/bert-base-finnish-cased-v1", num_layers=9)
    else:
        results = bertscore.compute(predictions=predictions, references=references, lang=lang)
    precision = np.mean(np.array(results['precision']))
    recall = np.mean(np.array(results['recall']))
    f1 = np.mean(np.array(results['f1']))
    print("F1:", f1)
    print("Precision:", precision)
    print("Recall:", recall)
    return results

def detect_language(predictions, top_k=1):
    # get the fasttext lid tool binary
    lid_bin = "/scratch/project_462000319/zosaelai2/lid.176.bin"
    lid_model = fasttext.load_model(lid_bin)
    # remove \n from predictions because fasttext processes each line
    predictions = [pred.replace("\n", " ") for pred in predictions]
    langs = lid_model.predict(predictions)
    langs = [lang[0].split("__")[-1] for lang in langs[0]]
    return langs

def predict_toxicity_score(predictions, cache_dir, score_thresh=0.1):
    model = transformers.AutoModelForSequenceClassification.from_pretrained("TurkuNLP/bert-large-finnish-cased-toxicity",
                                                                            cache_dir=cache_dir)
    tokenizer = transformers.AutoTokenizer.from_pretrained("TurkuNLP/bert-large-finnish-cased-v1")
    pipe = transformers.pipeline(task="text-classification", model=model, tokenizer=tokenizer, function_to_apply="sigmoid", top_k=None)
    scores = pipe(predictions)
    # scores include (in order): toxicity, obscene, insult, threat, identity_attack, severe_toxicity
    # for simplicity, get only scores for toxicity 
    toxicity_score = np.array([score[0]['score'] for score in scores])
    mean_toxicity = np.mean(toxicity_score)
    below_thresh = (toxicity_score < score_thresh).sum()
    above_thresh = (toxicity_score > score_thresh).sum()
    return below_thresh, above_thresh, mean_toxicity


def main(argv):
    args = argparser().parse_args(argv[1:])
    if args.tokenizer is None:
        args.tokenizer = args.model
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    model = load_model(args)
    if args.memory_usage:
        report_memory_usage('after model load')
    # check_devices(model, args)
    prompts, responses = load_prompts(args.file, args.max_prompts, lang=args.lang, base_model=args.base_model, chatml_format=args.chatml_format)
    print("prompts:", len(prompts))
    print("responses:", len(responses))
    generated = generate(prompts, tokenizer, model, args, responses)
    print("generated:", len(generated))
    results = compute_bertscore(references=responses, predictions=generated, lang=args.lang)
    print("Model:", args.model)
    print("Dataset:", args.file)
    print("Lang:", args.lang)
    if args.detect_lang:
        print("Languages in responses:")
        lang_preds = detect_language(generated)
        lang_counts = Counter(lang_preds)
        print(lang_counts)
    if args.detect_toxicity:
        score_thresh = 0.1
        non_toxic, toxic, mean_score = predict_toxicity_score(generated, 
                                                              cache_dir=args.transformers_cache, 
                                                              score_thresh=score_thresh)
        print("Toxicity score thresh:", score_thresh)
        print("Non-toxic:", non_toxic)
        print("Toxic:", toxic)
        print("Mean score:", mean_score)
    if args.output_file is not None:
        assert len(prompts) == len(responses) == len(generated)
        output = {"prompt": prompts, "generated_response": generated, "true_response": responses}
        output = pd.DataFrame.from_dict(output)
        output.to_json(args.output_file)
        print("Output file:", args.output_file)
    if args.memory_usage:
        report_memory_usage('after generation')


if __name__ == '__main__':
    sys.exit(main(sys.argv))
