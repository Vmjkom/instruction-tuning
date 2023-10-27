from datasets import Dataset
from pathlib import Path
import json


def read_oasst(path, lang='fi'):
  if lang == 'fi':
    text_col = "text"
  else:
    text_col = "orig_text"
  path=Path(path)
  with open(path, 'rb') as f:
    oasst_dict=list(f)
  questions_dict = {}
  context_wq_dict = {}
  context_return = []
  answers_return = []
  questions_return = []
  for json_str in oasst_dict:
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
      try:
        questions_return.append(questions_dict[result["parent_id"]])
      except:
        continue
      answers_return.append(result[text_col])
      if context_wq_dict[result["parent_id"]]:
        context_return.append(context_wq_dict[result["parent_id"]])
        context_wq_dict[result["message_id"]] = context_wq_dict[result["parent_id"]] + questions_dict[result["parent_id"]] + result[text_col]
      else:
        context_wq_dict[result["message_id"]] = questions_dict[result["parent_id"]] + "\n\n" + result[text_col]
      context_wq_dict[result["message_id"]] = context_wq_dict[result["parent_id"]] + "\n\n" + questions_dict[result["parent_id"]] + "\n\n" + result[text_col]
  return questions_return, context_return, answers_return

def read_dolly(path):
  path=Path(path)
  with open(path, 'rb') as f:
    dolly_dict=list(f)
  
  questions = []
  answers = []
  context = []

  for json_str in dolly_dict:
    result = json.loads(json_str)
    prompt = result['instruction'] + '\n\n'
    if result['context'] and not result['context'].isspace():
      context.append(result['context'])
    else:
      context.append(' ')
    questions.append(prompt)
    answers.append(result["response"])
  return questions, context, answers

def read_data(data="dolly", split="train", lang="fi"):
  questions = []
  answers = []
  context = []

  if "train" in split:
    if "dolly" in data:
      dolly_questions, dolly_context, dolly_answers = read_dolly("data/dolly-fi/dolly-fi-train.jsonl")
      questions = questions + dolly_questions
      context = context + dolly_context
      answers = answers + dolly_answers
      print("Size of dolly training data", len(dolly_questions))

    if "instruct_qa" in data:
      instruct_questions, instruct_context, instruct_answers = read_dolly("data/instruct_qa/instruct_qa_fi_train.jsonl")
      questions = questions + instruct_questions
      context = context + instruct_context
      answers = answers + instruct_answers
      print("Size of instruct_qa training data", len(instruct_questions))

    if "oasst" in data:
      oasst_questions, oasst_context, oasst_answers = read_oasst("data/oasst-fi/oasst1-fi-train.jsonl", lang=lang)
      questions = questions + oasst_questions
      context = context + oasst_context
      answers = answers + oasst_answers
      print("Size of oasst training data", len(questions))

  elif "valid" in split:
    if "dolly" in data:
      dolly_questions, dolly_context, dolly_answers = read_dolly("data/dolly-fi/dolly-fi-valid.jsonl")
      questions = questions + dolly_questions
      context = context + dolly_context
      answers = answers + dolly_answers

    if "instruct_qa" in data:
      instruct_questions, instruct_context, instruct_answers = read_dolly("data/instruct_qa/instruct_qa_fi_valid.jsonl")
      questions = questions + instruct_questions
      context = context + instruct_context
      answers = answers + instruct_answers

    if "oasst" in data:
      oasst_questions, oasst_context, oasst_answers = read_oasst("data/oasst-fi/oasst1-fi-valid.jsonl", lang=lang)
      questions = questions + oasst_questions
      context = context + oasst_context
      answers = answers + oasst_answers

  elif "eval" in split:
    if "dolly" in data:
      dolly_questions, dolly_context, dolly_answers = read_dolly("data/dolly-fi/dolly-fi-eval.jsonl")
      questions = questions + dolly_questions
      context = context + dolly_context
      answers = answers + dolly_answers

    if "instruct_qa" in data:
      instruct_questions, instruct_context, instruct_answers = read_dolly("data/instruct_qa/instruct_qa_fi_eval.jsonl")
      questions = questions + instruct_questions
      context = context + instruct_context
      answers = answers + instruct_answers

    if "oasst" in data:
      oasst_questions, oasst_context, oasst_answers = read_oasst("data/oasst-fi/oasst1-fi-eval.jsonl", lang=lang)
      questions = questions + oasst_questions
      context = context + oasst_context
      answers = answers + oasst_answers

  data = {
    'prompt': questions,
    'context': context,
    'response': answers,
    }

  return Dataset.from_dict(data)