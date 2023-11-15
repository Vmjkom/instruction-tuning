import sys
from functools import wraps
from time import time
from logging import warning
import torch
from transformers import AutoModelForCausalLM
from peft import (
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    LoraConfig,
    TaskType
)

def timed(f):
    @wraps(f)
    def timed_f(*args, **kwargs):
        start = time()
        result = f(*args, **kwargs)
        end = time()
        print(f'{f.__name__}: {end-start:.1f} seconds', file=sys.stderr)
        return result
    return timed_f

def logits_argmax(logits):
    # https://github.com/huggingface/transformers/issues/15466
    return logits.argmax(axis=-1)


def load_model(model_name, transformers_cache, use_lora=False, ignore_bias_buffers=False):
    print("load_model")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=transformers_cache,
        num_labels=1,
        torch_dtype=torch.bfloat16
    )
    if ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]
    if use_lora is True:
        print("Using lora")
        model.enable_input_require_grads()
        peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32,
                                     lora_dropout=0.1)
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        print("Loaded lora model")
    return model

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
