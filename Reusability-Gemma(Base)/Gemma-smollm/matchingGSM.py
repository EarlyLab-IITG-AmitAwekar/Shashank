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
df = pd.read_csv('./gemma3_smollm2_gsm8k_final_results.csv')
df['gemma3_answer'] = df['gemma3_cot_raw'].apply(extract_answer)
df['smollm2_answer'] = df['smollm2_with_cot_raw'].apply(extract_answer)

# Calculate match percentage
valid_df = df.dropna(subset=['gemma3_answer', 'smollm2_answer'])
match_rate = (valid_df['gemma3_answer'] == valid_df['smollm2_answer']).mean()

print(f"Average match rate: {match_rate:.2%}")
