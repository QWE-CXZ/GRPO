import json
import datasets
from datasets import Dataset
##USER_INSTRUCTION
USER_INSTRUCTION="""
分析这段文本的安全性，并按照[安全,偏见歧视,道德法律,身心伤害,财产与隐私,攻击冒犯]为文本打上标签：
"""
SYSTEM_PROMPT="""
并使用中文按以下格式回答问题:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""
file_path="gen_data.json"
with open(file_path, "r", encoding="utf-8") as f:
    datas = json.load(f)
processed_data=[]
for data in datas:
    instruction=USER_INSTRUCTION+data['context']
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": instruction}
    ]

    label_num=len(data['answer'])
    if label_num>1 and label_num==2:
        answer='['+data['answer'][0]+','+data['answer'][1]+']'
    elif label_num==1:
        answer = '[' + data['answer'][0] + ']'
    elif label_num>2:
        continue
    output=f"<reasoning>\n{data['reasoning']}\n</reasoning>\n<answer>\n{answer}\n</answer>"
    processed_data.append({"prompt": prompt, "response": output})
dataset = Dataset.from_list(processed_data)
# for i in range(10):
#     print("------------------")
#     print(dataset[i]['response'])
save_path='raw_data'
dataset.save_to_disk(save_path)