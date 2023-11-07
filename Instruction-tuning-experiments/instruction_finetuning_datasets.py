from datasets import Dataset
from pathlib import Path
import json
import numpy as np

# Preprocessing datasets for SFT
def read_oasst_sft(path, lang='fi'):
    # end_of_text = tokenizer.eos_token
    user_token = "<|user|>"
    assistant_token = "<|assistant|>"
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
            question_combined = user_token + " " + result[text_col]
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
            answer_combined = assistant_token + " " + result[text_col]
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


def read_dolly_sft(path, lang="fi"):
    if lang == "fi":
        instruction_col = "instruction"
        context_col = "context"
        response_col = "response"
    else:
        instruction_col = "orig_instruction"
        context_col = "orig_context"
        response_col = "orig_response"
    user_token = "<|user|>"
    assistant_token = "<|assistant|>"
    path = Path(path)
    with open(path, 'rb') as f:
        dolly_dict = list(f)
    questions = []
    answers = []
    context = []
    for json_str in dolly_dict:
        result = json.loads(json_str)
        # prompt = result['instruction'] + '\n\n'
        prompt = user_token + " " + result[instruction_col]
        if result[context_col] and not result[context_col].isspace():
            context.append(result[context_col])
        else:
            context.append(' ')
        questions.append(prompt)
        # answers.append(result["response"])
        answer = assistant_token + " " + result[response_col]
        answers.append(answer)
    return questions, context, answers


# Preprocessing datasets for DPO
def read_oasst_dpo(path, lang='fi', score_type='quality'):
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
        # print("="*20, index, "="*20)
        result = json.loads(json_str)
        if result["role"] == "prompter":
            questions_dict[result["message_id"]] = result[text_col]
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
                    answers_dict[result["parent_id"]]['answers'].append((result[text_col], response_score))
                    if context_wq_dict[result["parent_id"]]:
                        answers_dict[result["parent_id"]]["context"] = context_wq_dict[result["parent_id"]]
                        context_wq_dict[result["message_id"]] = context_wq_dict[result["parent_id"]] + questions_dict[
                            result["parent_id"]] + result[text_col]
                    else:
                        context_wq_dict[result["message_id"]] = questions_dict[result["parent_id"]] + "\n\n" + result[text_col]
                    context_wq_dict[result["message_id"]] = context_wq_dict[result["parent_id"]] + "\n\n" + questions_dict[
                        result["parent_id"]] + "\n\n" + result[text_col]
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
            answers_best_list.append(sorted_answers[0][0])
            answers_worst_list.append(sorted_answers[-1][0])
    return questions_list, contexts_list, answers_best_list, answers_worst_list
def read_ultrafeedback_dpo(path):
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
    return prompts_list, contexts_list, answers_best_list, answers_worst_list


def read_data_dpo(data="dolly", split="train", lang="fi"):
    questions = []
    context = []
    answers = []
    answers_best = []
    answers_worst = []
    if "train" in split:
        if "oasst" in data:
            if "lang" == "both":
                languages = ["fi", "en"]
            else:
                languages = [lang]
            for la in languages:
                oasst_questions, oasst_context, oasst_answers_best, oasst_answers_worst = read_oasst_dpo(
                "data/oasst-fi/oasst1-fi-train.jsonl",
                lang=la)
                questions = questions + oasst_questions
                context = context + oasst_context
                answers_best = answers_best + oasst_answers_best
                answers_worst = answers_worst + oasst_answers_worst
            print("Size of oasst training data", len(questions))

        if "ultrafeedback" in data:
            ultra_questions, ultra_context, ultra_answers_best, ultra_answers_worst = read_ultrafeedback_dpo(
                "data/UltraFeedback/ultrafeedback-train.jsonl")
            questions = questions + ultra_questions
            context = context + ultra_context
            answers_best = answers_best + ultra_answers_best
            answers_worst = answers_worst + ultra_answers_worst

    elif "valid" in split:
        if "oasst" in data:
            if "lang" == "both":
                languages = ["fi", "en"]
            else:
                languages = [lang]
            for la in languages:
                oasst_questions, oasst_context, oasst_answers_best, oasst_answers_worst = read_oasst_dpo(
                "data/oasst-fi/oasst1-fi-valid.jsonl",
                lang=la)
                questions = questions + oasst_questions
                context = context + oasst_context
                answers_best = answers_best + oasst_answers_best
                answers_worst = answers_worst + oasst_answers_worst

        if "ultrafeedback" in data:
            ultra_questions, ultra_context, ultra_answers_best, ultra_answers_worst = read_ultrafeedback_dpo(
                "data/UltraFeedback/ultrafeedback-valid.jsonl")
            questions = questions + ultra_questions
            context = context + ultra_context
            answers_best = answers_best + ultra_answers_best
            answers_worst = answers_worst + ultra_answers_worst

    elif "eval" in split:
        if "oasst" in data:
            if "lang" == "both":
                languages = ["fi", "en"]
            else:
                languages = [lang]
            for la in languages:
                oasst_questions, oasst_context, oasst_answers_best, oasst_answers_worst = read_oasst_dpo(
                "data/oasst-fi/oasst1-fi-eval.jsonl",
                lang=la)
                questions = questions + oasst_questions
                context = context + oasst_context
                answers_best = answers_best + oasst_answers_best
                answers_worst = answers_worst + oasst_answers_worst

        if "ultrafeedback" in data:
            ultra_questions, ultra_context, ultra_answers_best, ultra_answers_worst = read_ultrafeedback_dpo(
                "data/UltraFeedback/ultrafeedback-eval.jsonl")
            questions = questions + ultra_questions
            context = context + ultra_context
            answers_best = answers_best + ultra_answers_best
            answers_worst = answers_worst + ultra_answers_worst
    data = {
        'prompt': questions,
        'context': context,
        'accepted_response': answers_best,
        'rejected_response': answers_worst
    }
    return Dataset.from_dict(data)

def read_data_sft(data="dolly", split="train", lang="fi"):
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
                dolly_questions, dolly_context, dolly_answers = read_dolly_sft("data/dolly-fi/dolly-fi-train.jsonl", lang=la)
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
                instruct_questions, instruct_context, instruct_answers = read_dolly_sft(
                "data/instruct_qa/instruct_qa_fi_train.jsonl",
                lang=la)
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
                oasst_questions, oasst_context, oasst_answers = read_oasst_sft("data/oasst-fi/oasst1-fi-train.jsonl", lang=la)
                questions = questions + oasst_questions
                context = context + oasst_context
                answers = answers + oasst_answers
            print("Size of oasst training data", len(questions))

    elif "valid" in split:
        if "dolly" in data:
            if "lang" == "both":
                languages = ["fi", "en"]
            else:
                languages = [lang]
            for la in languages:
                dolly_questions, dolly_context, dolly_answers = read_dolly_sft("data/dolly-fi/dolly-fi-valid.jsonl", lang=la)
                questions = questions + dolly_questions
                context = context + dolly_context
                answers = answers + dolly_answers

        if "instruct_qa" in data:
            if "lang" == "both":
                languages = ["fi", "en"]
            else:
                languages = [lang]
            for la in languages:
                instruct_questions, instruct_context, instruct_answers = read_dolly_sft(
                "data/instruct_qa/instruct_qa_fi_valid.jsonl",
                lang=la)
                questions = questions + instruct_questions
                context = context + instruct_context
                answers = answers + instruct_answers

        if "oasst" in data:
            if "lang" == "both":
                languages = ["fi", "en"]
            else:
                languages = [lang]
            for la in languages:
                oasst_questions, oasst_context, oasst_answers = read_oasst_sft("data/oasst-fi/oasst1-fi-valid.jsonl",
                                                                       lang=la)
                questions = questions + oasst_questions
                context = context + oasst_context
                answers = answers + oasst_answers

    elif "eval" in split:
        if "dolly" in data:
            if "lang" == "both":
                languages = ["fi", "en"]
            else:
                languages = [lang]
            for la in languages:
                dolly_questions, dolly_context, dolly_answers = read_dolly_sft("data/dolly-fi/dolly-fi-eval.jsonl", lang=la)
                questions = questions + dolly_questions
                context = context + dolly_context
                answers = answers + dolly_answers

        if "instruct_qa" in data:
            if "lang" == "both":
                languages = ["fi", "en"]
            else:
                languages = [lang]
            for la in languages:
                instruct_questions, instruct_context, instruct_answers = read_dolly_sft(
                "data/instruct_qa/instruct_qa_fi_eval.jsonl",
                lang=la)
                questions = questions + instruct_questions
                context = context + instruct_context
                answers = answers + instruct_answers

        if "oasst" in data:
            if "lang" == "both":
                languages = ["fi", "en"]
            else:
                languages = [lang]
            for la in languages:
                oasst_questions, oasst_context, oasst_answers = read_oasst_sft("data/oasst-fi/oasst1-fi-eval.jsonl",
                                                                       lang=la)
                questions = questions + oasst_questions
                context = context + oasst_context
                answers = answers + oasst_answers

    data = {
        'prompt': questions,
        'context': context,
        'response': answers,
    }
    return Dataset.from_dict(data)


