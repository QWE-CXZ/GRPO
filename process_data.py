import json
import datasets
from datasets import Dataset
##USER_INSTRUCTION
USER_INSTRUCTION="""
分析这段文本的安全性，并按照[安全,偏见歧视,道德法律,身心伤害,财产与隐私,攻击冒犯]为文本打上标签：
"""
SYSTEM_PROMPT="""
使用中文按以下格式回答问题:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""
file_path="data/raw_data.json"
with open(file_path, "r", encoding="utf-8") as f:
    datas = json.load(f)
processed_data=[]
for data in datas:
    instruction=USER_INSTRUCTION+data['context']
    label_num=len(data['answer'])
    if label_num>1 and label_num==2:
        answer='['+data['answer'][0]+','+data['answer'][1]+']'
    elif label_num==1:
        answer = '[' + data['answer'][0] + ']'
    elif label_num>2:
        continue
    output=f"<reasoning>\n{data['reasoning']}\n</reasoning>\n<answer>\n{answer}\n</answer>"
    processed_data.append({"prompt": SYSTEM_PROMPT,"instruction":instruction, "response": output})
dataset = Dataset.from_list(processed_data)
sft_dataset=dataset.select(range(500))
print(sft_dataset.shape)
grpo_dataset=dataset.select(range(500,2000))
print(grpo_dataset.shape)
testset=dataset.select(range(2000,dataset.shape[0]))
print(testset.shape)
print(testset[0]['response'])
sft_save_path="data/sft_data"
grpo_save_path="data/grpo_data"
testset_save_path="data/test_data"
sft_dataset.save_to_disk(sft_save_path)
grpo_dataset.save_to_disk(grpo_save_path)
testset.save_to_disk(testset_save_path)
