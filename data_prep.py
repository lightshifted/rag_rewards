# data_prep.py
from datasets import load_from_disk, Dataset
import pandas as pd
import textwrap

df = pd.read_csv("data/rl_train_data_02.13.2025.csv")
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.075, seed=72)

dataset.save_to_disk("train_data")

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
    data = load_from_disk('train_data')[split] # type: ignore
    data = data.map(lambda x: { # type: ignore
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['train_prompt']}
        ],
        'answer': x['cpt_mods']
    }) # type: ignore
    return data # type: ignore

if __name__ == "__main__":
    dataset = get_medical_procedures()
