import pandas as pd
import re

def extract_boolean_answer(text):
    """Extract boolean answer from StrategyQA responses"""
    if pd.isna(text) or not text:
        return None
    
    text = str(text).lower().strip()
    
    # Handle empty strings or error messages
    if not text or "error:" in text:
        return None
    
    # Look for explicit answer patterns
    answer_patterns = [
        r'answer:\s*(true|false|yes|no)',
        r'final answer:\s*(true|false|yes|no)',
        r'therefore.*?answer.*?is.*?(true|false|yes|no)',
        r'the answer is.*?(true|false|yes|no)',
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, text)
        if match:
            answer = match.group(1)
            return answer in ['true', 'yes']
    
    # Find last boolean occurrence
    boolean_matches = list(re.finditer(r'\b(true|false|yes|no)\b', text))
    if boolean_matches:
        last_answer = boolean_matches[-1].group(1)
        return last_answer in ['true', 'yes']
    
    return None

# Load CSV and calculate match rate
df = pd.read_csv('llama_smollm2_strategyqa_final_results.csv')
df['llama_answer'] = df['llama_cot_raw'].apply(extract_boolean_answer)
df['smollm2_answer'] = df['smollm2_with_cot_raw'].apply(extract_boolean_answer)

# Calculate match percentage
valid_df = df.dropna(subset=['llama_answer', 'smollm2_answer'])
match_rate = (valid_df['llama_answer'] == valid_df['smollm2_answer']).mean()

print(f"Average match rate: {match_rate:.2%}")
