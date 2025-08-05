import pandas as pd
import re
import json
import regex

def find_boxed_content(text):
    if not text:
        return []
    try:
        pattern = regex.compile(r'\\boxed\s*(?P<brace>\{(?:[^{}]|(?&brace))*\})')
        matches = pattern.findall(text)
        return [m[1:-1].strip() for m in matches]
    except:
        return []

def parse_json_answer(text):
    if pd.isna(text) or not text:
        return ""
    text = str(text).strip()
    
    # Try JSON parsing
    json_pattern = r'\{\s*"answer"\s*:\s*"([^"]+)"\s*\}'
    match = re.search(json_pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Fallback to boxed content
    boxed = find_boxed_content(text)
    return boxed[-1] if boxed else ""

def extract_clean_text(text):
    if not text:
        return ""
    text = str(text)
    text = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'\1/\2', text)
    text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)
    text = re.sub(r'[{}]', '', text)
    return text.strip()

# Load CSV and calculate match rate
df = pd.read_csv('gemma3_stablelm2_math_final_results.csv')

# Extract answers
gemma3_boxed = df['gemma3_cot_raw'].apply(find_boxed_content)
df['gemma3_answer'] = gemma3_boxed.apply(lambda x: extract_clean_text(x[-1]) if x else "")
df['stablelm2_answer'] = df['stablelm2_with_cot_raw'].apply(lambda x: extract_clean_text(parse_json_answer(x)))

# Calculate match percentage
valid_df = df[(df['gemma3_answer'] != "") & (df['stablelm2_answer'] != "")]
match_rate = (valid_df['gemma3_answer'].str.lower() == valid_df['stablelm2_answer'].str.lower()).mean()

print(f"Average match rate: {match_rate:.2%}")
