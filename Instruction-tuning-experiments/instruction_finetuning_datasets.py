from datasets import Dataset
from pathlib import Path
import json


# Preprocessing datasets for SFT
def read_oasst(path, lang='fi'):
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


def read_dolly(path, lang="fi"):
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
def read_oasst_dpo(path, lang='fi'):
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
            if result['labels'] is not None and 'quality' in result['labels']['name']:
                quality_score_index = int(result['labels']['name'].index('quality'))
                quality_score = result['labels']['value'][quality_score_index]
                if result["parent_id"] in questions_dict:
                    if result["parent_id"] not in answers_dict:
                        answers_dict[result["parent_id"]] = {'question': questions_dict[result["parent_id"]],
                                                             'context': '',
                                                             'answers': []
                                                             }
                    answers_dict[result["parent_id"]]['answers'].append((result[text_col], quality_score))
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
        # sort answers by quality score
        sorted_answers = sorted(answers_dict[key]["answers"], key=lambda x: float(x[1]), reverse=True)
        # only return questions that have more than one answer
        if len(sorted_answers) > 1:
            questions_list.append(answers_dict[key]["question"])
            contexts_list.append(answers_dict[key]["context"])
            # best answer is answer with highest quality score
            answers_best_list.append(sorted_answers[0][0])
            # best answer is answer with lowers quality score
            answers_worst_list.append(sorted_answers[-1][0])
    return questions_list, contexts_list, answers_best_list, answers_worst_list


def read_data(data="dolly", split="train", lang="fi", task="sft"):
    questions = []
    context = []
    answers = []
    answers_best = []
    answers_worst = []
    if "train" in split:
        if "dolly" in data:
            dolly_questions, dolly_context, dolly_answers = read_dolly("data/dolly-fi/dolly-fi-train.jsonl", lang=lang)
            questions = questions + dolly_questions
            context = context + dolly_context
            answers = answers + dolly_answers
            print("Size of dolly training data", len(dolly_questions))

        if "instruct_qa" in data:
            instruct_questions, instruct_context, instruct_answers = read_dolly(
                "data/instruct_qa/instruct_qa_fi_train.jsonl",
                lang=lang)
            questions = questions + instruct_questions
            context = context + instruct_context
            answers = answers + instruct_answers
            print("Size of instruct_qa training data", len(instruct_questions))

        if "oasst" in data and "sft" in task:
            oasst_questions, oasst_context, oasst_answers = read_oasst("data/oasst-fi/oasst1-fi-train.jsonl", lang=lang)
            questions = questions + oasst_questions
            context = context + oasst_context
            answers = answers + oasst_answers
            print("Size of oasst SFT training data", len(questions))

        if "oasst" in data and "dpo" in task:
            oasst_questions, oasst_context, oasst_answers_best, oasst_answers_worst = read_oasst_dpo(
                "data/oasst-fi/oasst1-fi-train.jsonl",
                lang=lang)
            questions = questions + oasst_questions
            context = context + oasst_context
            answers_best = answers_best + oasst_answers_best
            answers_worst = answers_worst + oasst_answers_worst
            print("Size of oasst DPO training data", len(questions))

    elif "valid" in split:
        if "dolly" in data:
            dolly_questions, dolly_context, dolly_answers = read_dolly("data/dolly-fi/dolly-fi-valid.jsonl", lang=lang)
            questions = questions + dolly_questions
            context = context + dolly_context
            answers = answers + dolly_answers

        if "instruct_qa" in data:
            instruct_questions, instruct_context, instruct_answers = read_dolly(
                "data/instruct_qa/instruct_qa_fi_valid.jsonl",
                lang=lang)
            questions = questions + instruct_questions
            context = context + instruct_context
            answers = answers + instruct_answers

        if "oasst" in data and "sft" in task:
            oasst_questions, oasst_context, oasst_answers = read_oasst("data/oasst-fi/oasst1-fi-valid.jsonl",
                                                                       lang=lang)
            questions = questions + oasst_questions
            context = context + oasst_context
            answers = answers + oasst_answers

        if "oasst" in data and "dpo" in task:
            oasst_questions, oasst_context, oasst_answers_best, oasst_answers_worst = read_oasst_dpo(
                "data/oasst-fi/oasst1-fi-valid.jsonl",
                lang=lang)
            questions = questions + oasst_questions
            context = context + oasst_context
            answers_best = answers_best + oasst_answers_best
            answers_worst = answers_worst + oasst_answers_worst

    elif "eval" in split:
        if "dolly" in data:
            dolly_questions, dolly_context, dolly_answers = read_dolly("data/dolly-fi/dolly-fi-eval.jsonl", lang=lang)
            questions = questions + dolly_questions
            context = context + dolly_context
            answers = answers + dolly_answers

        if "instruct_qa" in data:
            instruct_questions, instruct_context, instruct_answers = read_dolly(
                "data/instruct_qa/instruct_qa_fi_eval.jsonl",
                lang=lang)
            questions = questions + instruct_questions
            context = context + instruct_context
            answers = answers + instruct_answers

        if "oasst" in data and "sft" in task:
            oasst_questions, oasst_context, oasst_answers = read_oasst("data/oasst-fi/oasst1-fi-eval.jsonl",
                                                                       lang=lang)
            questions = questions + oasst_questions
            context = context + oasst_context
            answers = answers + oasst_answers

        if "oasst" in data and "dpo" in task:
            oasst_questions, oasst_context, oasst_answers_best, oasst_answers_worst = read_oasst_dpo(
                "data/oasst-fi/oasst1-fi-eval.jsonl",
                lang=lang)
            questions = questions + oasst_questions
            context = context + oasst_context
            answers_best = answers_best + oasst_answers_best
            answers_worst = answers_worst + oasst_answers_worst

    if "dpo" in task:
        data = {
            'prompt': questions,
            'context': context,
            'accepted_response': answers_best,
            'rejected_response': answers_worst
        }
    else:
        data = {
            'prompt': questions,
            'context': context,
            'response': answers,
        }

    return Dataset.from_dict(data)
