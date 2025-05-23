import argparse
import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    BitsAndBytesConfig,
)
import datasets
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
from typing import Optional, List, Dict,Literal,Type

SYSTEM_PROMPT="""
    使用中文按以下格式回答问题:
    <reasoning>
    ...
    </reasoning>
    <answer>
    ...
    </answer>
    """
def get_model(args):
    load_type = 'auto'
    config_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": load_type,
        "low_cpu_mem_usage": True,
        "device_map": 'auto',
        "cache_dir": args.cache_dir,
    }
    if args.load_in_8bit:
        config_kwargs['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
    elif args.load_in_4bit:
        config_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=load_type,
        )
    base_model = AutoModelForCausalLM.from_pretrained(args.base_model, **config_kwargs)
    try:
        base_model.generation_config = GenerationConfig.from_pretrained(args.base_model, trust_remote_code=True)
    except OSError:
        print("Failed to load generation config, use default.")
    if args.lora_model:
        model = PeftModel.from_pretrained(base_model, args.lora_model, torch_dtype=load_type, device_map='auto')
        print("Loaded lora model")
    else:
        model = base_model
        print("Loaded raw model")
    model.eval()
    return model

def extract_reasoning(text:str) ->str:
    reasoning = text.split("<reasoning>")[-1]
    reasoning = reasoning.split("</reasoning>")[0]
    return reasoning.strip()
def extract_answer(text:str) ->List[str]:
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    answer = answer.split("[")[-1]
    answer = answer.split("]")[0]
    if ',' not in answer:
        return [answer.strip()]
    answer = answer.split(",")
    answer = [a.strip() for a in answer]
    return answer
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', default=None, type=str, required=True)
    parser.add_argument('--dataset_path', default=None, type=str, required=True)
    parser.add_argument('--cache_dir', default=None, type=str, )
    parser.add_argument('--lora_model', default="", type=str, help="If None, perform inference on the base model")
    parser.add_argument('--tokenizer_path', default=None, type=str)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument('--load_in_8bit', action='store_true', help='Whether to load model in 8bit')
    parser.add_argument('--load_in_4bit', action='store_true', help='Whether to load model in 4bit')
    args = parser.parse_args()
    print(args)
    if args.tokenizer_path is None:
        args.tokenizer_path = args.base_model

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True, padding_side='left')
    model = get_model(args)
    dataset = datasets.load_from_disk(args.dataset_path)
    right_count=0
    similarity_list=[]
    for i in tqdm(range(len(dataset))):
        instruction=dataset[i]['instruction']
        true_response=dataset[i]['response']
        true_reasoning=extract_reasoning(true_response)
        true_answer=extract_answer(true_response)

        messages = [
            {
                "role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user", "content": instruction}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to('cuda')

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        reasoning = extract_reasoning(response)
        answer = extract_answer(response)

        if set(true_answer) == set(answer):
            right_count += 1

        similarity = util.cos_sim(semantic_model.encode(reasoning), semantic_model.encode(true_reasoning))
        similarity_list.append(similarity.item())
    print(f"Accuracy: {right_count/len(dataset)}")
    mean=sum(similarity_list)/len(similarity_list)
    print(f"Average similarity: {mean}")
    print(f"Max similarity: {max(similarity_list)}")
    print(f"Min similarity: {min(similarity_list)}")
    squared_diff_sum = sum((x - mean) ** 2 for x in similarity_list)
    variance = squared_diff_sum / len(similarity_list)
    print(f"Variance: {variance}")


if __name__ == '__main__':
    main()
