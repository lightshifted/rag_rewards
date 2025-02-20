# data_prep.py
from datasets import load_dataset, Dataset


SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

def get_medical_procedures(split = "train") -> Dataset:
    dataset = load_dataset("lightshifted/mimic-icd10-cm",
            token="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",
            split="train")
    data = dataset.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['train_prompt']}
        ],
        'answer': x['icd_cm_target']
    })
    return data 

if __name__ == "__main__":
    dataset = get_medical_procedures()
