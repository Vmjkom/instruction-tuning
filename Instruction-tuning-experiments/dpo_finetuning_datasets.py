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

def read_oasst(path, lang='fi', score_type='toxicity', max_examples=0):
    if lang == 'fi':
        text_col = "text"
    else:
        text_col = "orig_text"
    path = Path(path)
    with open(path, 'rb') as f:
        oasst_dict = list(f)
    questions_dict = {}
    context_wq_dict = {}
    answers_dict = {}
    for index, json_str in enumerate(oasst_dict):
        result = json.loads(json_str)
        if result["role"] == "prompter":
            question_combined = user_token + result[text_col]
            questions_dict[result["message_id"]] = question_combined
            context_wq_dict[result["message_id"]] = " "
            if result["parent_id"]:
                try:
                    context_wq_dict[result["message_id"]] = context_wq_dict[result["parent_id"]]
                except:
                    context_wq_dict[result["message_id"]] = " "
        elif result["role"] == "assistant":
            if result['labels'] is not None and score_type in result['labels']['name']:
                response_score_index = int(result['labels']['name'].index(score_type))
                response_score = result['labels']['value'][response_score_index]
                if result["parent_id"] in questions_dict:
                    if result["parent_id"] not in answers_dict:
                        answers_dict[result["parent_id"]] = {'question': questions_dict[result["parent_id"]],
                                                             'context': '',
                                                             'answers': []
                                                             }
                    answer_combined = assistant_token + result[text_col]
                    answers_dict[result["parent_id"]]['answers'].append((answer_combined, response_score))
                    if context_wq_dict[result["parent_id"]]:
                        answers_dict[result["parent_id"]]["context"] = context_wq_dict[result["parent_id"]]
                        context_wq_dict[result["message_id"]] = context_wq_dict[result["parent_id"]] + questions_dict[
                            result["parent_id"]] + answer_combined
                    else:
                        context_wq_dict[result["message_id"]] = questions_dict[result["parent_id"]] + "\n" + answer_combined
                    context_wq_dict[result["message_id"]] = context_wq_dict[result["parent_id"]] + "\n" + questions_dict[
                        result["parent_id"]] + "\n" + answer_combined
    questions_list = []
    contexts_list = []
    answers_best_list = []
    answers_worst_list = []
    for key in answers_dict:
        # sort answers by response score
        sorted_answers = sorted(answers_dict[key]["answers"], key=lambda x: float(x[1]), reverse=True)
        # only return prompts that have more than one answer
        if len(sorted_answers) > 1:
            questions_list.append(answers_dict[key]["question"])
            contexts_list.append(answers_dict[key]["context"])
            answers_best_list.append(sorted_answers[-1][0])
            answers_worst_list.append(sorted_answers[0][0])
            # answers_best_list.append(sorted_answers[0][0])
            # answers_worst_list.append(sorted_answers[-1][0])
    if max_examples > 0:
        questions_list = questions_list[:max_examples]
        contexts_list = contexts_list[:max_examples]
        answers_best_list = answers_best_list[:max_examples]
        answers_worst_list = answers_worst_list[:max_examples]
    # for i in range(20):
    #     print("CONTEXT:", contexts_list[i])
    #     print("\nQUESTION:", questions_list[i])
    #     print("\nCHOSEN:", answers_best_list[i])
    #     print("\nREJECTED:", answers_worst_list[i])
    #     print("-"*100)
    return questions_list, contexts_list, answers_best_list, answers_worst_list

def read_ultrafeedback(path, max_examples=0):
    data = [json.loads(line) for line in open(path)]
    prompts_list = []
    contexts_list = []
    answers_best_list = []
    answers_worst_list = []
    criteria = ['instruction_following', 'honesty', 'truthfulness', 'helpfulness']
    for index, entry in enumerate(data):
        instruction = entry['instruction']
        completions = entry['completions']
        scores = []
        for comp in completions:
            score = np.mean([float(comp['annotations'][crit]['Rating'])
                             if (comp['annotations'][crit]['Rating']).isnumeric() else 0
                             for crit in criteria])
            scores.append(score)
        # check if best and worst scores are not equal
        if len(scores) > 1 and np.max(scores) > np.min(scores):
            best_answer = completions[np.argmax(scores)]['response']
            worst_answer = completions[np.argmin(scores)]['response']
            prompts_list.append(instruction)
            # we actually don't have context, this is just to harmonise formatting with oasst
            contexts_list.append('')
            answers_best_list.append(best_answer)
            answers_worst_list.append(worst_answer)
    if max_examples > 0:
        prompts_list = prompts_list[:max_examples]
        contexts_list = contexts_list[:max_examples]
        answers_best_list = answers_best_list[:max_examples]
        answers_worst_list = answers_worst_list[:max_examples]
    return prompts_list, contexts_list, answers_best_list, answers_worst_list


def read_hh(path, single_turn_only=False, max_examples=0):
    questions_list = []
    contexts_list = []
    chosen_list = []
    rejected_list = []
    hh_data = [json.loads(line) for line in open(path)]
    for entry in hh_data:
        chosen = entry['chosen']
        rejected = entry['rejected']
        turn_indices = [m.start() for m in re.finditer("Human:", chosen)]
        if (single_turn_only is False) or (single_turn_only is True and len(turn_indices) == 1):
            chosen = chosen.replace("\n\nHuman:", "\n"+user_token).strip()
            chosen = chosen.replace("\n\nAssistant:", "\n"+assistant_token).strip()
            # user_token_index = 0
            # while user_token_index != -1:
            #     asst_token_index = chosen.find(assistant_token)
            #     user_token_index = chosen.find()
            #     prompt = chosen[user_token_index:asst_token_index]
            #     chosen_answer = chosen[asst_token_index:]
            #     next_user_token = chosen_answer.find(user_token)
            #     chosen_answer = chosen_answer[:next_user_token]
            question = chosen[:chosen.rindex(assistant_token)].strip()
            answer_chosen = chosen[chosen.rindex(assistant_token):].strip()
            rejected = entry['rejected']
            rejected = rejected.replace("\n\nHuman:", "\n"+user_token).strip()
            rejected = rejected.replace("\n\nAssistant:", "\n"+assistant_token).strip()
            answer_rejected = rejected[rejected.rindex(assistant_token):].strip()
            questions_list.append(question)
            chosen_list.append(answer_chosen)
            rejected_list.append(answer_rejected)
            contexts_list.append('')
    if max_examples > 0:
        questions_list = questions_list[:max_examples]
        contexts_list = contexts_list[:max_examples]
        chosen_list = chosen_list[:max_examples]
        rejected_list = rejected_list[:max_examples]
    # for i in range(30):
    #     print("QUESTION:", questions_list[i])
    #     print("\nCHOSEN:", chosen_list[i])
    #     print("\nREJECTED:", rejected_list[i])
    #     print("-"*100)
    return questions_list, contexts_list, chosen_list, rejected_list


def read_oasst_lang_alignment(path):
    languages = ["fi", "en"]
    col_names = {
        "fi": {"text": "text"},
        "en": {"text": "orig_text"}
    }
    path = Path(path)
    with open(path, 'rb') as f:
        oasst_dict = list(f)
    questions_dict = {}
    context_wq_dict = {}
    context_return = []
    # answers_return = []
    chosen_answers_return = []
    rejected_answers_return = []
    questions_return = []
    for index, json_str in enumerate(oasst_dict):
        # print("-"*20, "Index", index, "-"*20)
        result = json.loads(json_str)
        if result["role"] == "prompter":
            questions_dict[result["message_id"]] = {lang: user_token + " " + result[col_names[lang]["text"]] for lang in languages}
            context_wq_dict[result["message_id"]] = {lang: " " for lang in languages }
            if result["parent_id"]:
                try:
                    context_wq_dict[result["message_id"]] = {lang: context_wq_dict[result["parent_id"]][lang] for lang in languages}
                except:
                    context_wq_dict[result["message_id"]] = {lang: " " for lang in languages}
        elif result["role"] == "assistant":
            try:
                questions_return.extend([questions_dict[result["parent_id"]][lang] for lang in languages])
                # print("Question:", questions_dict[result["parent_id"]])
            except:
                continue
            answer_combined = {lang: assistant_token + " " + result[col_names[lang]["text"]] for lang in languages}
            chosen_answers_return.extend([answer_combined[lang] for lang in languages])
            rejected_answers_return.extend([answer_combined[lang] for lang in reversed(languages)])
            # answers_return.append(answer_combined)
            # print("Answer:", answer_combined)
            # answers_return.append(result[text_col])
            if context_wq_dict[result["parent_id"]]:
                context_return.extend([context_wq_dict[result["parent_id"]][lang] for lang in languages])
                # context_return.append(context_wq_dict[result["parent_id"]])
                # print("Context:", context_wq_dict[result["parent_id"]])
                # context_wq_dict[result["message_id"]] = context_wq_dict[result["parent_id"]] + questions_dict[result["parent_id"]] + result[text_col]
                context_wq_dict[result["message_id"]] ={lang: context_wq_dict[result["parent_id"]][lang] + questions_dict[
                    result["parent_id"]][lang] + answer_combined[lang] for lang in languages}
            else:
                # context_wq_dict[result["message_id"]] = questions_dict[result["parent_id"]] + "\n" + result[text_col]
                context_wq_dict[result["message_id"]] = {lang: questions_dict[result["parent_id"]][lang] + "\n" + answer_combined[lang] 
                                                         for lang in languages}
            # context_wq_dict[result["message_id"]] = context_wq_dict[result["parent_id"]] + "\n\n" + questions_dict[result["parent_id"]] + "\n\n" + result[text_col]
            context_wq_dict[result["message_id"]] = {lang: context_wq_dict[result["parent_id"]][lang] + "\n" + questions_dict[
                result["parent_id"]][lang] + "\n" + answer_combined[lang] for lang in languages}
    return questions_return, context_return, chosen_answers_return, rejected_answers_return


def read_dolly_lang_alignment(path, max_examples=0):
    languages = ["fi", "en"]
    col_names = {
        "fi": {
            "instruction": "instruction",
            "context": "context",
            "response": "response"
            },
        "en": {
            "instruction": "orig_instruction",
            "context": "orig_context",
            "response": "orig_response"
        }
    }
    path = Path(path)
    with open(path, 'rb') as f:
        dolly_dict = list(f)
    prompts_list = []
    contexts_list = []
    answers_chosen_list = []
    answers_rejected_list = []
    for json_str in dolly_dict:
        entry = json.loads(json_str)
        prompts = [entry[col_names[lang]["instruction"]] for lang in languages]
        contexts = [entry[col_names[lang]["context"]] for lang in languages]
        answers_chosen = [entry[col_names[lang]["response"]] for lang in languages]
        answers_rejected = [entry[col_names[lang]["response"]] for lang in reversed(languages)]
        if answers_rejected[0] != answers_rejected[1]:
            prompts_list.extend(prompts)
            contexts_list.extend(contexts)
            answers_chosen_list.extend(answers_chosen)
            answers_rejected_list.extend(answers_rejected)
    if max_examples > 0:
        prompts_list = prompts_list[:max_examples]
        contexts_list = contexts_list[:max_examples]
        answers_chosen_list = answers_chosen_list[:max_examples]
        answers_rejected_list = answers_rejected_list[:max_examples]
    return prompts_list, contexts_list, answers_chosen_list, answers_rejected_list


def read_data_dpo(data="oasst", split="train", lang="fi", shuffle_data=True, max_examples=10000):
    questions = []
    context = []
    answers_best = []
    answers_worst = []
    if "train" in split:
        if "oasst" in data:
            if "lang" in data:
                oasst_questions, oasst_context, oasst_answers_best, oasst_answers_worst = read_oasst_lang_alignment("data/oasst-fi/oasst1-fi-train-filter.jsonl")
                questions = questions + oasst_questions
                context = context + oasst_context
                answers_best = answers_best + oasst_answers_best
                answers_worst = answers_worst + oasst_answers_worst
                print("Size of oasst lang alignment training data", len(questions))
            else:
                if "lang" == "both":
                    languages = ["fi", "en"]
                else:
                    languages = [lang]
                for la in languages:
                    oasst_questions, oasst_context, oasst_answers_best, oasst_answers_worst = read_oasst("data/oasst-fi/oasst1-fi-train-filter.jsonl",
                                                                                                     lang=la,
                                                                                                     max_examples=max_examples)
                    questions = questions + oasst_questions
                    context = context + oasst_context
                    answers_best = answers_best + oasst_answers_best
                    answers_worst = answers_worst + oasst_answers_worst
                print("Size of oasst preference training data", len(questions))
        if "ultrafeedback" in data:
            ultra_questions, ultra_context, ultra_answers_best, ultra_answers_worst = read_ultrafeedback("data/UltraFeedback/ultrafeedback-train.jsonl",
                                                                                                         max_examples=max_examples)
            questions = questions + ultra_questions
            context = context + ultra_context
            answers_best = answers_best + ultra_answers_best
            answers_worst = answers_worst + ultra_answers_worst
            print("Size of ultrafeedback training data", len(questions))
        if "dolly" in data:
            dolly_questions, dolly_context, dolly_answers_best, dolly_answers_worst = read_dolly_lang_alignment("data/dolly-fi/dolly-fi-train.jsonl",
                                                                                                                max_examples=max_examples)
            questions = questions + dolly_questions
            context = context + dolly_context
            answers_best = answers_best + dolly_answers_best
            answers_worst = answers_worst + dolly_answers_worst
            print("Size of dolly training data", len(questions))
        if "hh" in data:
            parent_path = "/scratch/project_462000319/finetuning_data/hh_rlhf"
            if "helpful" in data:
                hh_questions, hh_context, hh_answers_best, hh_answers_worst = read_hh(os.path.join(parent_path, "helpful-base-train.jsonl"),
                                                                                  single_turn_only=True,
                                                                                  max_examples=max_examples)
            
                questions = questions + hh_questions
                context = context + hh_context
                answers_best = answers_best + hh_answers_best
                answers_worst = answers_worst + hh_answers_worst
                print("Size of hh helpful training data", len(questions))
            if "harmless" in data:
                hh_questions, hh_context, hh_answers_best, hh_answers_worst = read_hh(os.path.join(parent_path, "harmless-base-train.jsonl"),
                                                                                  single_turn_only=True,
                                                                                  max_examples=max_examples)
                questions = questions + hh_questions
                context = context + hh_context
                answers_best = answers_best + hh_answers_best
                answers_worst = answers_worst + hh_answers_worst
                print("Size of hh harmless training data", len(questions))
            else:
                hh_questions, hh_context, hh_answers_best, hh_answers_worst = read_hh(os.path.join(parent_path, "hh_rlhf-train.jsonl"),
                                                                                      single_turn_only=True,
                                                                                      max_examples=max_examples)
                questions = questions + hh_questions
                context = context + hh_context
                answers_best = answers_best + hh_answers_best
                answers_worst = answers_worst + hh_answers_worst
                print("Size of hh training data", len(questions))
    elif "valid" in split:
        if "oasst" in data:
            if "lang" in data:
                oasst_questions, oasst_context, oasst_answers_best, oasst_answers_worst = read_oasst_lang_alignment("data/oasst-fi/oasst1-fi-valid-filter.jsonl")
                questions = questions + oasst_questions
                context = context + oasst_context
                answers_best = answers_best + oasst_answers_best
                answers_worst = answers_worst + oasst_answers_worst
            else:
                if "lang" == "both":
                    languages = ["fi", "en"]
                else:
                    languages = [lang]
                for la in languages:
                    oasst_questions, oasst_context, oasst_answers_best, oasst_answers_worst = read_oasst("data/oasst-fi/oasst1-fi-valid-filter.jsonl",
                                                                                                     lang=la,
                                                                                                     max_examples=max_examples)
                    questions = questions + oasst_questions
                    context = context + oasst_context
                    answers_best = answers_best + oasst_answers_best
                    answers_worst = answers_worst + oasst_answers_worst
        if "ultrafeedback" in data:
            ultra_questions, ultra_context, ultra_answers_best, ultra_answers_worst = read_ultrafeedback("data/UltraFeedback/ultrafeedback-valid.jsonl",
                                                                                                         max_examples=max_examples)
            questions = questions + ultra_questions
            context = context + ultra_context
            answers_best = answers_best + ultra_answers_best
            answers_worst = answers_worst + ultra_answers_worst
        if "dolly" in data:
            dolly_questions, dolly_context, dolly_answers_best, dolly_answers_worst = read_dolly_lang_alignment("data/dolly-fi/dolly-fi-valid.jsonl",
                                                                                                                max_examples=max_examples)
            questions = questions + dolly_questions
            context = context + dolly_context
            answers_best = answers_best + dolly_answers_best
            answers_worst = answers_worst + dolly_answers_worst
        if "hh" in data:
            parent_path = "/scratch/project_462000319/finetuning_data/hh_rlhf"
            if "helpful" in data:
                hh_questions, hh_context, hh_answers_best, hh_answers_worst = read_hh(os.path.join(parent_path, "helpful-base-test.jsonl"),
                                                                                  single_turn_only=True,
                                                                                  max_examples=max_examples)
                questions = questions + hh_questions
                context = context + hh_context
                answers_best = answers_best + hh_answers_best
                answers_worst = answers_worst + hh_answers_worst
            if "harmless" in data:
                hh_questions, hh_context, hh_answers_best, hh_answers_worst = read_hh(os.path.join(parent_path, "harmless-base-test.jsonl"),
                                                                                  single_turn_only=True,
                                                                                  max_examples=max_examples)
                questions = questions + hh_questions
                context = context + hh_context
                answers_best = answers_best + hh_answers_best
                answers_worst = answers_worst + hh_answers_worst
            else:
                hh_questions, hh_context, hh_answers_best, hh_answers_worst = read_hh(os.path.join(parent_path, "hh_rlhf-valid.jsonl"),
                                                                                      single_turn_only=True,
                                                                                      max_examples=max_examples)
                questions = questions + hh_questions
                context = context + hh_context
                answers_best = answers_best + hh_answers_best
                answers_worst = answers_worst + hh_answers_worst
    elif "eval" in split:
        if "oasst" in data:
            if "lang" in data:
                oasst_questions, oasst_context, oasst_answers_best, oasst_answers_worst = read_oasst_lang_alignment("data/oasst-fi/oasst1-fi-eval-filter.jsonl")
                questions = questions + oasst_questions
                context = context + oasst_context
                answers_best = answers_best + oasst_answers_best
                answers_worst = answers_worst + oasst_answers_worst
            else:
                if "lang" == "both":
                    languages = ["fi", "en"]
                else:
                    languages = [lang]
                for la in languages:
                    oasst_questions, oasst_context, oasst_answers_best, oasst_answers_worst = read_oasst("data/oasst-fi/oasst1-fi-eval-filter.jsonl",
                                                                                                     lang=la,
                                                                                                     max_examples=max_examples)
                    questions = questions + oasst_questions
                    context = context + oasst_context
                    answers_best = answers_best + oasst_answers_best
                    answers_worst = answers_worst + oasst_answers_worst
        if "ultrafeedback" in data:
            ultra_questions, ultra_context, ultra_answers_best, ultra_answers_worst = read_ultrafeedback("data/UltraFeedback/ultrafeedback-eval.jsonl",
                                                                                                         max_examples=max_examples)
            questions = questions + ultra_questions
            context = context + ultra_context
            answers_best = answers_best + ultra_answers_best
            answers_worst = answers_worst + ultra_answers_worst
        if "dolly" in data:
            dolly_questions, dolly_context, dolly_answers_best, dolly_answers_worst = read_dolly_lang_alignment("data/dolly-fi/dolly-fi-eval.jsonl",
                                                                                                                max_examples=max_examples)
            questions = questions + dolly_questions
            context = context + dolly_context
            answers_best = answers_best + dolly_answers_best
            answers_worst = answers_worst + dolly_answers_worst
        if "hh" in data:
            parent_path = "/scratch/project_462000319/finetuning_data/hh_rlhf"
            if "helpful" in data:
                hh_questions, hh_context, hh_answers_best, hh_answers_worst = read_hh(os.path.join(parent_path, "helpful-base-test.jsonl"),
                                                                                  single_turn_only=True,
                                                                                  max_examples=max_examples)
                questions = questions + hh_questions
                context = context + hh_context
                answers_best = answers_best + hh_answers_best
                answers_worst = answers_worst + hh_answers_worst
            if "harmless" in data:
                hh_questions, hh_context, hh_answers_best, hh_answers_worst = read_hh(os.path.join(parent_path, "harmless-base-test.jsonl"),
                                                                                  single_turn_only=True,
                                                                                  max_examples=max_examples)
                questions = questions + hh_questions
                context = context + hh_context
                answers_best = answers_best + hh_answers_best
                answers_worst = answers_worst + hh_answers_worst
            else:
                hh_questions, hh_context, hh_answers_best, hh_answers_worst = read_hh(os.path.join(parent_path, "hh_rlhf-test.jsonl"),
                                                                                      single_turn_only=True,
                                                                                      max_examples=max_examples)
                questions = questions + hh_questions
                context = context + hh_context
                answers_best = answers_best + hh_answers_best
                answers_worst = answers_worst + hh_answers_worst

    data = {
        'prompt': questions,
        'context': context,
        'accepted_response': answers_best,
        'rejected_response': answers_worst
    }
    data = Dataset.from_dict(data)
    if shuffle_data:
        data = data.shuffle(seed=42)
    return data