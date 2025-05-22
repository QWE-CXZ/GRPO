import argparse

import torch
from peft import PeftModel, PeftConfig,TaskType
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default=None, required=True, type=str,
                        help="Base model name or path")
    parser.add_argument('--tokenizer_path', default=None, type=str,
                        help="Please specify tokenization path.")
    parser.add_argument('--lora_model', default=None, required=True, type=str,
                        help="Please specify LoRA model to be merged.")
    parser.add_argument('--resize_emb', action='store_true', help='Whether to resize model token embeddings')
    parser.add_argument('--output_dir', default='./merged', type=str)
    parser.add_argument('--cache_dir',default='Qwen_model_file', type=str,)
    args = parser.parse_args()
    print(args)

    base_model_path = args.base_model
    lora_model_path = args.lora_model
    output_dir = args.output_dir
    print(f"Base model: {base_model_path}")
    print(f"LoRA model: {lora_model_path}")
    peft_config = PeftConfig.from_pretrained(lora_model_path)

    assert peft_config.task_type == TaskType.CAUSAL_LM

    print("Loading LoRA for causal language model")
    print(base_model_path)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        cache_dir=args.cache_dir,
        torch_dtype='auto',
        trust_remote_code=True,
        device_map="auto",
    )
    if args.tokenizer_path:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    if args.resize_emb:
        base_model_token_size = base_model.get_input_embeddings().weight.size(0)
        if base_model_token_size != len(tokenizer):
            base_model.resize_token_embeddings(len(tokenizer))
            print(f"Resize vocabulary size {base_model_token_size} to {len(tokenizer)}")
    print("Loading LoRA model")
    new_model = PeftModel.from_pretrained(
        base_model,
        lora_model_path,
        device_map="auto",
        torch_dtype='auto',
    )
    new_model.eval()
    print(f"Merging with merge_and_unload...")
    base_model = new_model.merge_and_unload()

    print("Saving to Hugging Face format...")
    tokenizer.save_pretrained(output_dir)
    base_model.save_pretrained(output_dir, max_shard_size='10GB')
    print(f"Done! model saved to {output_dir}")



if __name__ == '__main__':
    main()
