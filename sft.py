import os
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import datasets
from dataclasses import dataclass

from typing import Optional, List, Dict,Literal,Type
from transformers.trainer_pt_utils import LabelSmoother
import os
from tqdm import tqdm
from transformers import (
    Trainer,
    TrainingArguments,
    HfArgumentParser,
    DataCollatorForSeq2Seq,
)
from peft import (
    TaskType,
    LoraConfig,
    get_peft_model,
    PeftModel,
)
from accelerate import PartialState
from datasets import Dataset,concatenate_datasets

is_flash_attn_2_available = False
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input

    is_flash_attn_2_available = True
except ImportError:
    is_flash_attn_2_available = False

IGNORE_INDEX = LabelSmoother.ignore_index
try:
    local_rank = int(os.environ["LOCAL_RANK"])
except:
    local_rank = 0
@dataclass
class ModelArguments:
    model_name_or_path: str = "Qwen/Qwen2.5-3B-Instruct"
    tokenizer_name_or_path: Optional[str] = "Qwen/Qwen2.5-3B-Instruct"
    cache_dir: Optional[str] = "Qwen_model_file"
    model_max_length: Optional[int] = 1024
    use_fast_tokenizer: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    trust_remote_code: bool = True
    rope_scaling: Optional[Literal["linear", "dynamic"]]=None
    flash_attn: bool = False
    shift_attn: bool = False
    attn_implementation:str="flash_attention_2"

@dataclass
class DataArguments:
    dataset_path_name: str = "sft_data"
    num_proc:int = 4
    max_train_samples:int = 500

@dataclass
class TrainingArguments(TrainingArguments):
    output_dir = "./output/Qwen2.5-3B-Instruct"
    per_device_train_batch_size:int = 4
    gradient_accumulation_steps:int = 1
    logging_steps:int = 10
    num_train_epochs:int = 3
    save_steps:int = 100
    learning_rate:float = 1e-4
    gradient_checkpointing:bool = False
@dataclass
class LoraArguments:
    task_type = TaskType.CAUSAL_LM
    target_model:str="all"
    rank:int=16
    lora_alpha:int=32
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
            cache_dir=model_args.cache_dir,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            load_in_8bit=model_args.load_in_8bit,
            load_in_4bit=model_args.load_in_4bit,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            cache_dir=model_args.cache_dir,
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
def build_dataset(
        dataset_path_name:str,
        tokenizer:AutoTokenizer,
        num_proc:int,
        model_max_length: int,
        seed:int=42,
):
    assert os.path.exists(dataset_path_name)
    dataset = datasets.load_from_disk(dataset_path_name)
    def process_function(examples):
        instruction=(f"<|im_start|>system\n{examples['prompt']}<|im_end|>\n<|im_start|>user"
                     f"\n{examples['instruction']}<|im_end|>\n<|im_start|>assistant\n")
        instruction=tokenizer(instruction,add_special_tokens=False)
        response=tokenizer(f"{examples['response']}",add_special_tokens=False)
        input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.eos_token_id]
        attention_mask = instruction["attention_mask"] + response["attention_mask"] + [tokenizer.eos_token_id]
        labels = [IGNORE_INDEX] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eos_token_id]
        if len(input_ids) > model_max_length:
            input_ids = input_ids[:model_max_length]
            attention_mask = attention_mask[:model_max_length]
            labels = labels[:model_max_length]
        result={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return result

    train_dataset=dataset.map(
        process_function,
        num_proc=num_proc,
        remove_columns=dataset.column_names,
    )
    train_dataset=train_dataset.shuffle(seed=seed)
    print(train_dataset[0])
    return train_dataset
def save_model(model, tokenizer, args):
    """Save the model and the tokenizer."""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments, LoraArguments))
    model_args, data_args, training_args,lora_args = parser.parse_args_into_dataclasses()
    #print(training_args)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name_or_path,
    )

    model = load_model(model_args)
    if training_args.gradient_checkpointing:
        print("gradient_checkpointing")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        
    # output_layer = getattr(model, "lm_head")
    # if isinstance(output_layer, torch.nn.Linear) and output_layer.weight.dtype != torch.float32:
    #     def fp32_forward_post_hook(module: torch.nn.Module, args: Tuple[torch.Tensor], output: torch.Tensor):
    #         return output.to(torch.float32)

    #     output_layer.register_forward_hook(fp32_forward_post_hook)
    
    lora_args.target_model = find_all_linear_names(peft_model=model,int4=model_args.load_in_4bit,int8=model_args.load_in_8bit)
    peft_config=LoraConfig(
        task_type=lora_args.task_type,
        target_modules=lora_args.target_model,
        r=lora_args.rank,
        lora_alpha=lora_args.lora_alpha,
        lora_dropout=lora_args.lora_dropout,
        inference_mode=False
    )
    logger.info(f"lora config:{lora_args}")
    with PartialState().local_main_process_first():
        train_dataset=build_dataset(data_args.dataset_path_name,tokenizer=tokenizer,
                      num_proc=data_args.num_proc,model_max_length=model_args.model_max_length,
                      )
        max_train_samples=min(len(train_dataset),data_args.max_train_samples)
        train_dataset=train_dataset.select(range(max_train_samples))

    model = get_peft_model(model, peft_config)
    for param in filter(lambda p: p.requires_grad, model.parameters()):
        param.data = param.data.to(torch.float32)

    # # Initialize our Trainer
    # if training_args.gradient_checkpointing and getattr(model, "supports_gradient_checkpointing", False):
    #     model.gradient_checkpointing_enable()
    #     model.config.use_cache = False
    #     logger.info("Gradient checkpointing enabled.")
    # else:
    #     model.config.use_cache = True
    #     logger.info("Gradient checkpointing disabled.")
    model.enable_input_require_grads()

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        label_pad_token_id=IGNORE_INDEX,
    )
    trainer=Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        train_dataset=train_dataset,
        args=training_args
    )
    if trainer.is_world_process_zero():
        logger.info("training")
    train_results = trainer.train(
        resume_from_checkpoint=training_args.resume_from_checkpoint if training_args.resume_from_checkpoint else None
    )
    metrics = train_results.metrics
    metrics["train_samples"] = max_train_samples
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    if trainer.is_world_process_zero():
        logger.info(f"Training metrics: {metrics}")
        logger.info(f"Saving model checkpoint to {training_args.output_dir}")
        save_model(model, tokenizer, training_args)
if __name__ == '__main__':
    main()
