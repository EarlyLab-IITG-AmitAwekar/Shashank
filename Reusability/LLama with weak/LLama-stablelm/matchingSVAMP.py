import pandas as pd
import re

def extract_answer(text):
    if pd.isna(text) or not text:
        return None
    m = re.search(r'####(.*)', str(text))
    if m:
        num = re.search(r'[-+]?\d*\.?\d+', m.group(1))
        if num:
            return float(num.group(0))
    nums = re.findall(r'[-+]?\d*\.?\d+', str(text))
    return float(nums[-1]) if nums else None

# Load CSV and calculate match rate
df = pd.read_csv('./llama_stablelm2_svamp_final_results.csv')
df['llama_answer'] = df['llama_cot_raw'].apply(extract_answer)
df['stablelm2_answer'] = df['stablelm2_with_cot_raw'].apply(extract_answer)

# Calculate match percentage
valid_df = df.dropna(subset=['llama_answer', 'stablelm2_answer'])
match_rate = (valid_df['llama_answer'] == valid_df['stablelm2_answer']).mean()

print(f"Average match rate: {match_rate:.2%}")
