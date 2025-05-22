import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    PeftModel,
)
from trl import GRPOConfig, GRPOTrainer, TrlParser
import torch
import datasets
from datasets import Dataset
from dataclasses import dataclass
from accelerate import PartialState

from typing import Optional, List, Dict,Literal,Type
from transformers.trainer_pt_utils import LabelSmoother
import re
from tqdm import tqdm

from sentence_transformers import SentenceTransformer, util
semantic_model = SentenceTransformer('all-MiniLM-L6-v2')

is_flash_attn_2_available = False
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input

    is_flash_attn_2_available = True
except ImportError:
    is_flash_attn_2_available = False

try:
    local_rank = int(os.environ["LOCAL_RANK"])
except:
    local_rank = 0

@dataclass
class ModelArguments:
    model_name_or_path: str = "Qwen-sft-merged"
    tokenizer_name_or_path: Optional[str] = "Qwen-sft-merged"
    model_max_length: Optional[int] = 1024
    use_fast_tokenizer: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = True
    rope_scaling: Optional[Literal["linear", "dynamic"]]=None
    flash_attn: bool = False
    attn_implementation:str="flash_attention_2"

@dataclass
class DataArguments:
    dataset_path_name: str = "data/grpo_data"
    num_proc:int = 4
    max_train_samples:int = 1500

@dataclass
class LoraArguments:
    task_type = TaskType.CAUSAL_LM
    target_model:str="all"
    rank:int=32
    lora_alpha:int=64
    lora_dropout:float=0


def load_model(model_args:ModelArguments):
    if is_flash_attn_2_available:
        model=AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            load_in_8bit=model_args.load_in_8bit,
            load_in_4bit=model_args.load_in_4bit,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation=model_args.attn_implementation,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            load_in_8bit=model_args.load_in_8bit,
            load_in_4bit=model_args.load_in_4bit,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    model.config.use_cache = False
    model.train()
    return model

def find_all_linear_names(peft_model, int4=False, int8=False):
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)

SYSTEM_PROMPT="""
使用中文按以下格式回答问题:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""
def get_dataset(dataset_path_name:str,num_proc:int,):
        dataset = datasets.load_from_disk(dataset_path_name)
        dataset = dataset.shuffle(seed=42)
        dataset=dataset.map(
            lambda x: {
                'prompt': [
                    {'role': 'system', 'content': SYSTEM_PROMPT},
                    {'role': 'user', 'content': x['instruction']}
                ],
                'answer': x['response']
            },
            num_proc=num_proc,
            remove_columns=['response','instruction'],
        )
        print(dataset[0])
        return dataset
def save_model(model, tokenizer, args):
    """Save the model and the tokenizer."""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

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
    
def reasoning_reward_func(completions, answer, **kwargs):
    """
    计算推理奖励
    """
    responses = [completion[0]['content'] for completion in completions]
    output_reasoning = [extract_reasoning(response) for response in responses]
    ref_reasoning = [extract_reasoning(ans) for ans in answer]
    similarities = util.cos_sim(semantic_model.encode(output_reasoning), semantic_model.encode(ref_reasoning))
    rewards = []
    for sim in similarities.diagonal().tolist():  # 取对角线上的值（单个样本的相似度）
        if sim > 0.9:
            rewards.append(2.0)  # 非常接近
        elif sim > 0.7:
            rewards.append(1.5)  # 相关性较高
        elif sim > 0.5:
            rewards.append(1.0)  # 可能部分正确
        else:
            rewards.append(0.0)  # 相关性低
    return rewards

def answer_reward_func(completions, answer, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    output_answer = [extract_answer(response) for response in responses]
    ref_answer = [extract_answer(ans) for ans in answer]
    rewards = []
    for output_a,ref_a in zip(output_answer,ref_answer):
        if len(output_a) == 1 and len(ref_a) == 1:
            if output_a[0] == ref_a[0]:
                rewards.append(2.0) #全对得2分
            else:
                rewards.append(0.0)
        elif len(output_a) > 1 and len(ref_a) == 1:
            reward=0.0
            for a in output_a:
                if a == ref_a[0]:
                    reward=1.0 # 部分正确得1分
                    break
            rewards.append(reward)
        elif len(output_a) == 1 and len(ref_a) > 1:
            reward=0.0
            for a in ref_a:
                if a == output_a[0]:
                    reward=1.0 # 部分正确得1分
                    break
            rewards.append(reward)

        elif len(output_a) > 1 and len(ref_a) > 1:
            if set(output_a) == set(ref_a):
                rewards.append(2.5)# 全对得2.5分
                continue
            reward=0.0
            for a in output_a:
                if a in ref_a:
                    reward+=0.5 #答对一个得0.5分
            rewards.append(reward)
        else:
            rewards.append(0.0)
    return rewards
   
# 严格格式奖励：必须完全匹配 <reasoning>...</reasoning><answer>...</answer>
def strict_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if re.match(pattern, r) else 0.0 for r in responses]

# 软格式奖励：只需包含 <reasoning> 和 <answer> 部分
def soft_format_reward_func(completions, **kwargs) -> list[float]:
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if re.search(pattern, r) else 0.0 for r in responses]

def count_xml(text) -> float:
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1])*0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1)*0.001
    return count

def xmlcount_reward_func(completions, **kwargs) -> list[float]:
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

def main():
    parser=TrlParser((ModelArguments,GRPOConfig,DataArguments,LoraArguments))
    model_args,training_args,data_args,lora_args=parser.parse_args_and_config()

    tokenizer=AutoTokenizer.from_pretrained(model_args.tokenizer_name_or_path,trust_remote_code=model_args.trust_remote_code)

    with PartialState().local_main_process_first():
        train_dataset=get_dataset(data_args.dataset_path_name,num_proc=data_args.num_proc)
    
    model = load_model(model_args)
    if training_args.gradient_checkpointing:
        print("gradient_checkpointing")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    
    lora_args.target_model = find_all_linear_names(peft_model=model,int4=model_args.load_in_4bit,int8=model_args.load_in_8bit)
    peft_config=LoraConfig(
        task_type=lora_args.task_type,
        target_modules=lora_args.target_model,
        r=lora_args.rank,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        inference_mode=False
    )
    model = get_peft_model(model, peft_config)

    trainer=GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            reasoning_reward_func,
            answer_reward_func,
            strict_format_reward_func,
            soft_format_reward_func,
            xmlcount_reward_func
            ],
        args=training_args,
        train_dataset=train_dataset,
    )
    if trainer.is_world_process_zero():
        logger.info("training")
    train_results = trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint if training_args.resume_from_checkpoint else None
    )
    metrics = train_results.metrics
    metrics["train_samples"] = data_args.max_train_samples
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    if trainer.is_world_process_zero():
        logger.info(f"Training metrics: {metrics}")
        logger.info(f"Saving model checkpoint to {training_args.output_dir}")
        save_model(model, tokenizer, training_args)

if __name__=="__main__":
    main()
