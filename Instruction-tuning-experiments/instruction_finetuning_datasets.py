from datasets import Dataset
from pathlib import Path
import os
import re
import json
import numpy as np

user_token = "<|user|>"
assistant_token = "<|assistant|>"
chatml_start_token = "<|im_start|>"
chatml_end_token = "<|im_end|>"


def read_oasst(path, lang='fi', chatml_format=False):
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
        # print("-"*20, "Index", index, "-"*20)
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
                # print("Question:", questions_dict[result["parent_id"]])
            except:
                continue
            if chatml_format:
                answer_combined = chatml_start_token + "assistant\n" + result[text_col] + chatml_end_token
            else:
                answer_combined = assistant_token + result[text_col]
            answers_return.append(answer_combined)
            # print("Answer:", answer_combined)
            # answers_return.append(result[text_col])
            if context_wq_dict[result["parent_id"]]:
                context_return.append(context_wq_dict[result["parent_id"]])
                # print("Context:", context_wq_dict[result["parent_id"]])
                # context_wq_dict[result["message_id"]] = context_wq_dict[result["parent_id"]] + questions_dict[result["parent_id"]] + result[text_col]
                context_wq_dict[result["message_id"]] = context_wq_dict[result["parent_id"]] + questions_dict[
                    result["parent_id"]] + answer_combined
            else:
                # context_wq_dict[result["message_id"]] = questions_dict[result["parent_id"]] + "\n" + result[text_col]
                context_wq_dict[result["message_id"]] = questions_dict[result["parent_id"]] + "\n" + answer_combined
            # context_wq_dict[result["message_id"]] = context_wq_dict[result["parent_id"]] + "\n\n" + questions_dict[result["parent_id"]] + "\n\n" + result[text_col]
            context_wq_dict[result["message_id"]] = context_wq_dict[result["parent_id"]] + "\n" + questions_dict[
                result["parent_id"]] + "\n" + answer_combined
    return questions_return, context_return, answers_return


def read_dolly(path, lang="fi", chatml_format=False):
    if lang == "fi":
        instruction_col = "instruction"
        context_col = "context"
        response_col = "response"
    else:
        instruction_col = "orig_instruction"
        context_col = "orig_context"
        response_col = "orig_response"
    path = Path(path)
    with open(path, 'rb') as f:
        dolly_dict = list(f)
    questions = []
    answers = []
    context = []
    for json_str in dolly_dict:
        result = json.loads(json_str)
        # prompt = result['instruction'] + '\n\n'
        if chatml_format:
            prompt = chatml_start_token + "user\n" + result[instruction_col] + chatml_end_token
        else:
            prompt = user_token + result[instruction_col]
        if result[context_col] and not result[context_col].isspace():
            context.append(result[context_col])
        else:
            context.append(' ')
        questions.append(prompt)
        # answers.append(result["response"])
        if chatml_format:
            answer = chatml_start_token + "assistant\n" + result[response_col] + chatml_end_token
        else:
            answer = assistant_token + result[response_col]
        answers.append(answer)
    return questions, context, answers


def read_eval_tasks(task="arc_challenge", split="train", chatml_format=False):
    parent_path = "/scratch/project_462000319/jburdge/data/eval_datasets"
    eval_task_datasets = {
        "arc_challenge": {
            "train": "arc/arc_challenge-train-split.jsonl",
            "valid": "arc/arc_challenge-valid-split.jsonl"
            },
        "drop": {
            "train": "drop/drop-train-split.jsonl",
            "valid": "drop/drop-valid-split.jsonl"
            },
        "gsm8k": {
            "train": "gsm8k/gsm8k-train-split.jsonl",
            "valid": "gsm8k/gsm8k-valid-split.jsonl"
            },
        "hellaswag": {
            "train": "hellaswag/hellaswag-train-split.jsonl",
            "valid": "hellaswag/hellaswag-valid-split.jsonl"
            }
    }
    questions = []
    answers = []
    contexts = []
    data_path = Path(os.path.join(parent_path, eval_task_datasets[task][split]))
    results = [json.loads(line) for line in open(data_path)]
    for result in results:
        if task != "hellaswag":
            result = re.split("Question:|Answer:", result['text'])
            answer = assistant_token + result[-1].strip()
            question = user_token + result[-2].strip()
            # print("Question:", question)
            # print("Answer:", answer)
            if answer and question:
                questions.append(question)
                answers.append(answer)
                # dummy context, don't mind it
                contexts.append('')
        else:
            result = result['text'].split(".")
            question = user_token + result[0]+"."
            answer = assistant_token + result[1]+"."
            if len(question) > 1 and len(answer) > 1:
                questions.append(question)
                answers.append(answer)
                # dummy context, don't mind it
                contexts.append('')
    return questions, contexts, answers


def read_hh(path, chatml_format=False):
    questions = []
    contexts = []
    answers = []
    hh_data = [json.loads(line) for line in open(path)]
    for entry in hh_data:
        chosen = entry['chosen']
        chosen = chosen.replace("\n\nHuman:", "\n"+user_token).strip()
        chosen = chosen.replace("\n\nAssistant:", "\n"+assistant_token).strip()
        question = chosen[:chosen.rindex(assistant_token)].strip()
        answer = chosen[chosen.rindex(assistant_token):].strip()
        questions.append(question)
        answers.append(answer)
        contexts.append('')
    return questions, contexts, answers
                

def read_data_sft(data="dolly", split="train", lang="fi", chatml_format=False, eval_task="arc_challenge"):
    questions = []
    context = []
    answers = []
    if "train" in split:
        if "dolly" in data:
            if "lang" == "both":
                languages = ["fi", "en"]
            else:
                languages = [lang]
            for la in languages:
                dolly_questions, dolly_context, dolly_answers = read_dolly("data/dolly-fi/dolly-fi-train.jsonl", 
                                                                               lang=la,
                                                                               chatml_format=chatml_format)
                questions = questions + dolly_questions
                context = context + dolly_context
                answers = answers + dolly_answers
            print("Size of dolly training data", len(questions))

        if "instruct_qa" in data:
            if "lang" == "both":
                languages = ["fi", "en"]
            else:
                languages = [lang]
            for la in languages:
                instruct_questions, instruct_context, instruct_answers = read_dolly("data/instruct_qa/instruct_qa_fi_train.jsonl",
                                                                                        lang=la,
                                                                                        chatml_format=chatml_format)
                questions = questions + instruct_questions
                context = context + instruct_context
                answers = answers + instruct_answers
            print("Size of instruct_qa training data", len(questions))
        if "oasst" in data:
            if "lang" == "both":
                languages = ["fi", "en"]
            else:
                languages = [lang]
            for la in languages:
                oasst_questions, oasst_context, oasst_answers = read_oasst("data/oasst-fi/oasst1-fi-train-filter.jsonl", 
                                                                               lang=la, 
                                                                               chatml_format=chatml_format)
                questions = questions + oasst_questions
                context = context + oasst_context
                answers = answers + oasst_answers
            print("Size of oasst training data", len(questions))
        if "eval_tasks" in data:
            eval_questions, eval_context, eval_answers = read_eval_tasks(task=eval_task, 
                                                                             split=split, 
                                                                             chatml_format=chatml_format)
            questions = questions + eval_questions
            context = context + eval_context
            answers = answers + eval_answers
            print("Size of eval_tasks training data", len(questions))
        if "hh" in data:
            parent_path = "/scratch/project_462000319/finetuning_data/hh_rlhf"
            hh_questions, hh_context, hh_answers = read_hh(os.path.join(parent_path, "hh_rlhf-train.jsonl"),
                                                               chatml_format=chatml_format)
            questions = questions + hh_questions
            context = context + hh_context
            answers = answers + hh_answers
            print("Size of HH training data", len(questions))
    elif "valid" in split:
        if "dolly" in data:
            if "lang" == "both":
                languages = ["fi", "en"]
            else:
                languages = [lang]
            for la in languages:
                dolly_questions, dolly_context, dolly_answers = read_dolly("data/dolly-fi/dolly-fi-valid.jsonl", 
                                                                               lang=la,
                                                                               chatml_format=chatml_format)
                questions = questions + dolly_questions
                context = context + dolly_context
                answers = answers + dolly_answers
        if "instruct_qa" in data:
            if "lang" == "both":
                languages = ["fi", "en"]
            else:
                languages = [lang]
            for la in languages:
                instruct_questions, instruct_context, instruct_answers = read_dolly("data/instruct_qa/instruct_qa_fi_valid.jsonl",
                                                                                        lang=la,
                                                                                        chatml_format=chatml_format)
                questions = questions + instruct_questions
                context = context + instruct_context
                answers = answers + instruct_answers
        if "oasst" in data:
            if "lang" == "both":
                languages = ["fi", "en"]
            else:
                languages = [lang]
            for la in languages:
                oasst_questions, oasst_context, oasst_answers = read_oasst("data/oasst-fi/oasst1-fi-valid-filter.jsonl",
                                                                       lang=la,
                                                                       chatml_format=chatml_format)
                questions = questions + oasst_questions
                context = context + oasst_context
                answers = answers + oasst_answers
        if "eval_tasks" in data:
            eval_questions, eval_context, eval_answers = read_eval_tasks(task=eval_task,
                                                                             split=split, 
                                                                             chatml_format=chatml_format)
            questions = questions + eval_questions
            context = context + eval_context
            answers = answers + eval_answers
        if "hh" in data:
            parent_path = "/scratch/project_462000319/finetuning_data/hh_rlhf"
            hh_questions, hh_context, hh_answers = read_hh(os.path.join(parent_path, "hh_rlhf-valid.jsonl"),
                                                               chatml_format=chatml_format)
            questions = questions + hh_questions
            context = context + hh_context
            answers = answers + hh_answers
            print("Size of HH training data", len(questions))
    elif "eval" in split:
        if "dolly" in data:
            if "lang" == "both":
                languages = ["fi", "en"]
            else:
                languages = [lang]
            for la in languages:
                dolly_questions, dolly_context, dolly_answers = read_dolly("data/dolly-fi/dolly-fi-eval.jsonl", 
                                                                               lang=la,
                                                                               chatml_format=chatml_format)
                questions = questions + dolly_questions
                context = context + dolly_context
                answers = answers + dolly_answers
        if "instruct_qa" in data:
            if "lang" == "both":
                languages = ["fi", "en"]
            else:
                languages = [lang]
            for la in languages:
                instruct_questions, instruct_context, instruct_answers = read_dolly(
                "data/instruct_qa/instruct_qa_fi_eval.jsonl",
                lang=la,
                chatml_format=chatml_format)
                questions = questions + instruct_questions
                context = context + instruct_context
                answers = answers + instruct_answers
        if "oasst" in data:
            if "lang" == "both":
                languages = ["fi", "en"]
            else:
                languages = [lang]
            for la in languages:
                oasst_questions, oasst_context, oasst_answers = read_oasst("data/oasst-fi/oasst1-fi-eval-filter.jsonl",
                                                                       lang=la,
                                                                       chatml_format=chatml_format)
                questions = questions + oasst_questions
                context = context + oasst_context
                answers = answers + oasst_answers
        if "eval_tasks" in data:
            eval_questions, eval_context, eval_answers = read_eval_tasks(task=eval_task,
                                                                             split="valid", 
                                                                             chatml_format=chatml_format)
            questions = questions + eval_questions
            context = context + eval_context
            answers = answers + eval_answers
        if "hh" in data:
            parent_path = "/scratch/project_462000319/finetuning_data/hh_rlhf"
            hh_questions, hh_context, hh_answers = read_hh(os.path.join(parent_path, "hh_rlhf-test.jsonl"),
                                                               chatml_format=chatml_format)
            questions = questions + hh_questions
            context = context + hh_context
            answers = answers + hh_answers
            print("Size of HH training data", len(questions))
    data = {
        'prompt': questions,
        'context': context,
        'response': answers,
    }
    data = Dataset.from_dict(data)
    # data = data.shuffle(seed=42)
    return data


