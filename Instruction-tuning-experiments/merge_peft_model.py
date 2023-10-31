import sys
import torch
from peft import (
    PeftModel,
    PeftConfig
)

from transformers import AutoModelForCausalLM

from argparse import ArgumentParser
def argparser():
    ap = ArgumentParser()
    ap.add_argument('--lora_adapter', type=str)
    ap.add_argument('--output_dir', type=str)
    return ap


def main(argv):
    args = argparser().parse_args(argv[1:])

    # Load PeFT LoRA model
    print("LoRA adatper:", args.lora_adapter)
    config = PeftConfig.from_pretrained(args.lora_adapter)
    print("Base model:", config.base_model_name_or_path)
    base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path,
                                                 torch_dtype=torch.bfloat16,
                                                 cache_dir="/scratch/project_2007628/transformers_cache")
    print("Loaded base model")
    model_to_merge = PeftModel.from_pretrained(base_model, args.lora_adapter, torch_dtype=torch.bfloat16)
    print("Loaded PeftModel")

    merged_model = model_to_merge.merge_and_unload()
    merged_model.save_pretrained(args.output_dir)
    print("Done merging model! Saved merged model to", args.output_dir)

if __name__ == '__main__':
    sys.exit(main(sys.argv))