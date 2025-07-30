import requests
import json
import re
import pandas as pd
from tqdm import tqdm
import time
import os
import numpy as np

# --- Configuration ---
API_URL = 'http://127.0.0.1:11434/api/generate'
PHI_MODEL = 'phi:2.7b'  # Changed from SMOLLM2_MODEL = 'smollm2:1.7b'
LLAMA_RESULTS_CSV_PATH = '../../llama_strategyqa_dataset_results.csv'
FINAL_RESULTS_CSV_PATH = 'llama_phi_strategyqa_final_results.csv'  # Updated filename
SUMMARY_CSV_PATH = 'llama_phi_strategyqa_accuracy_summary.csv'  # Updated filename
CHUNK_SAVE_SIZE = 20

# --- Prompt Templates ---

PROMPT_DIRECT = """
You are an expert at answering strategy and reasoning questions.
Your task is to analyze the given question and provide a clear boolean answer.
Think through the problem systematically to ensure the correct solution.

IMPORTANT: You must provide your answer in this EXACT format:
Answer: True
OR
Answer: False

Do not include any other text, explanations, or symbols. Only provide the boolean answer.

Question: {question}

Answer:"""

PROMPT_PHI_WITH_COT = """
You are not allowed to use your own reasoning or knowledge. You must ONLY use the provided reasoning (Chain of Thought) below to answer the question. Do not add, change, or infer anything. Just follow the steps exactly as given, even if they seem wrong or incomplete.

Continue from where the provided reasoning stops and complete the solution.

IMPORTANT: You must provide your final answer in this EXACT format:
Answer: True
OR
Answer: False

Do not provide any other text, reasoning, or symbols. Only the final boolean answer.

Question: {question}

Reasoning (Chain of Thought):
{cot}

Answer:"""

# --- Helper Functions ---

def generate_llm_response(llm_name: str, prompt: str) -> str:
    """Generate response from LLM with enhanced error handling"""
    if not prompt or not isinstance(prompt, str):
        return "Error: Invalid prompt"
    
    payload = {"model": llm_name, "prompt": prompt, "stream": False}
    try:
        response = requests.post(API_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json().get('response', '').strip()
        return result if result else "Error: Empty response"
    except requests.exceptions.RequestException as e:
        return f"Error: API request failed: {str(e)}"
    except Exception as e:
        return f"Error: {str(e)}"

def extract_llm_boolean_answer(text):
    """Enhanced boolean answer extraction for StrategyQA responses"""
    # Handle NaN, None, or non-string inputs
    if pd.isna(text) or text is None:
        return None
    
    # Convert to string if it's not already
    if not isinstance(text, str):
        text = str(text)
    
    # Handle empty strings or error messages
    if not text.strip() or "Error:" in text:
        return None
    
    # Convert to lowercase for easier matching
    text_lower = text.lower().strip()
    
    # Method 1: Look for explicit answer patterns (most reliable)
    answer_patterns = [
        r'answer:\s*(true|false|yes|no)',
        r'final answer:\s*(true|false|yes|no)',
        r'therefore.*?answer.*?is.*?(true|false|yes|no)',
        r'the answer is.*?(true|false|yes|no)',
        r'therefore.*?(true|false|yes|no)',
        r'conclusion.*?(true|false|yes|no)'
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, text_lower)
        if match:
            answer = match.group(1)
            return answer in ['true', 'yes']
    
    # Method 2: Look for standalone conclusions at the end
    end_patterns = [
        r'\b(true|false|yes|no)\b\s*\.?\s*$',
        r'therefore.*?\b(true|false|yes|no)\b',
        r'conclusion.*?\b(true|false|yes|no)\b'
    ]
    
    for pattern in end_patterns:
        match = re.search(pattern, text_lower)
        if match:
            answer = match.group(1)
            return answer in ['true', 'yes']
    
    # Method 3: Find all true/false/yes/no occurrences and take the last one
    boolean_matches = list(re.finditer(r'\b(true|false|yes|no)\b', text_lower))
    if boolean_matches:
        last_answer = boolean_matches[-1].group(1)
        return last_answer in ['true', 'yes']
    
    # Method 4: Look for affirmative/negative patterns (fallback)
    if re.search(r'\b(correct|right|accurate|valid|positive)\b', text_lower):
        return True
    if re.search(r'\b(incorrect|wrong|inaccurate|invalid|negative|not true)\b', text_lower):
        return False
    
    return None

def extract_strategyqa_answer(text):
    """Extract boolean answer from StrategyQA format with enhanced error handling"""
    # Handle NaN, None, or non-string inputs
    if pd.isna(text) or text is None:
        return None
    
    # Convert to string if it's not already
    if not isinstance(text, str):
        text = str(text)
    
    if not text.strip():
        return None
    
    # Convert to lowercase for easier matching
    text_lower = text.lower().strip()
    
    # Direct boolean conversion with priority order
    if text_lower in ['true', '1', 'yes']:
        return True
    elif text_lower in ['false', '0', 'no']:
        return False
    
    # Look for boolean values in text
    if 'true' in text_lower and 'false' not in text_lower:
        return True
    elif 'false' in text_lower and 'true' not in text_lower:
        return False
    elif 'yes' in text_lower and 'no' not in text_lower:
        return True
    elif 'no' in text_lower and 'yes' not in text_lower:
        return False
    
    return None

def safe_get_column_value(row, column_name, default_value=""):
    """Safely get column value, handling NaN and missing columns"""
    try:
        value = row[column_name]
        if pd.isna(value):
            return default_value
        return str(value)
    except (KeyError, IndexError):
        print(f"Warning: Column '{column_name}' not found in row")
        return default_value

def check_if_processing_needed(csv_path):
    """Check if Phi processing is already complete"""
    if not os.path.exists(csv_path):
        return True
    
    try:
        df = pd.read_csv(csv_path)
        phi_cols = [  # Updated column names
            'phi_direct_raw',
            'phi_with_cot_raw', 
            'phi_with_corrupted_cot_raw',
            'phi_with_partial_cot_raw'
        ]
        
        # Check if all Phi columns exist and have data
        for col in phi_cols:
            if col not in df.columns:
                return True
            if df[col].isna().all():
                return True
        
        print(f"Phi processing already complete. Found {len(df)} rows with Phi data.")
        return False
    except Exception as e:
        print(f"Error checking existing CSV: {e}")
        return True

def main():
    """Load LLaMA data, collect Phi responses, combine and evaluate"""
    print("=== Phi Processing and Evaluation Script (StrategyQA) ===")  # Updated title

    # Check if processing is needed
    if not check_if_processing_needed(FINAL_RESULTS_CSV_PATH):
        print("Skipping Phi processing, moving directly to evaluation...")
    else:
        # Verify LLaMA results exist
        if not os.path.exists(LLAMA_RESULTS_CSV_PATH):
            print(f"Error: LLaMA results file '{LLAMA_RESULTS_CSV_PATH}' not found.")
            print("Please run the LLaMA data collection script first.")
            return

        # Overwrite check
        if os.path.exists(FINAL_RESULTS_CSV_PATH):
            resp = input(f"File '{FINAL_RESULTS_CSV_PATH}' exists. Overwrite? (y/n): ")
            if resp.lower() != 'y':
                print("Exiting without overwriting.")
                return
            os.remove(FINAL_RESULTS_CSV_PATH)

        # Load and process
        try:
            df_llama = pd.read_csv(LLAMA_RESULTS_CSV_PATH)
            print(f"Loaded {len(df_llama)} rows from LLaMA results")
        except Exception as e:
            print(f"Error loading LLaMA results: {e}")
            return
        
        combined = []
        failed_requests = 0
        
        for idx, row in tqdm(df_llama.iterrows(), total=len(df_llama), desc="Processing Phi"):  # Updated description
            try:
                q = safe_get_column_value(row, 'question')
                cot_full = safe_get_column_value(row, 'llama_cot_steps_only')
                cot_corrupt = safe_get_column_value(row, 'corrupted_cot_steps_only')
                cot_partial = safe_get_column_value(row, 'partial_cot_steps')

                # Skip if question is empty
                if not q.strip():
                    print(f"Warning: Empty question at row {idx}, skipping")
                    continue

                # Phi responses with individual error handling
                responses = {}
                
                try:
                    responses['direct'] = generate_llm_response(PHI_MODEL, PROMPT_DIRECT.format(question=q))
                except Exception as e:
                    print(f"Error generating direct response for row {idx}: {e}")
                    responses['direct'] = f"Error: {str(e)}"
                    failed_requests += 1
                
                try:
                    responses['full'] = generate_llm_response(PHI_MODEL, PROMPT_PHI_WITH_COT.format(question=q, cot=cot_full))
                except Exception as e:
                    print(f"Error generating full CoT response for row {idx}: {e}")
                    responses['full'] = f"Error: {str(e)}"
                    failed_requests += 1
                
                try:
                    responses['corrupt'] = generate_llm_response(PHI_MODEL, PROMPT_PHI_WITH_COT.format(question=q, cot=cot_corrupt))
                except Exception as e:
                    print(f"Error generating corrupted CoT response for row {idx}: {e}")
                    responses['corrupt'] = f"Error: {str(e)}"
                    failed_requests += 1
                
                try:
                    responses['partial'] = generate_llm_response(PHI_MODEL, PROMPT_PHI_WITH_COT.format(question=q, cot=cot_partial))
                except Exception as e:
                    print(f"Error generating partial CoT response for row {idx}: {e}")
                    responses['partial'] = f"Error: {str(e)}"
                    failed_requests += 1

                combined.append({
                    'question': q,
                    'original_answer_raw': safe_get_column_value(row, 'original_answer_raw'),
                    'llama_direct_raw': safe_get_column_value(row, 'llama_direct_raw'),
                    'llama_cot_raw': safe_get_column_value(row, 'llama_cot_raw'),
                    'llama_corrupted_cot_raw': safe_get_column_value(row, 'llama_corrupted_cot_raw'),
                    'llama_cot_steps_only': cot_full,
                    'corrupted_cot_steps_only': cot_corrupt,
                    'partial_cot_steps': cot_partial,
                    'llama_partial_cot_response': safe_get_column_value(row, 'llama_partial_cot_response'),
                    'phi_direct_raw': responses['direct'],  # Updated column names
                    'phi_with_cot_raw': responses['full'],
                    'phi_with_corrupted_cot_raw': responses['corrupt'],
                    'phi_with_partial_cot_raw': responses['partial'],
                })

                # Periodic save
                if (idx + 1) % CHUNK_SAVE_SIZE == 0 or (idx + 1) == len(df_llama):
                    try:
                        df_chunk = pd.DataFrame(combined)
                        header = not os.path.exists(FINAL_RESULTS_CSV_PATH)
                        df_chunk.to_csv(FINAL_RESULTS_CSV_PATH, mode='a', header=header, index=False)
                        combined = []
                        print(f"Saved checkpoint at row {idx + 1}")
                    except Exception as e:
                        print(f"Error saving checkpoint at row {idx + 1}: {e}")
                
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                failed_requests += 1
                continue

        if failed_requests > 0:
            print(f"Warning: {failed_requests} requests failed")

    # Evaluation phase
    print("Starting evaluation...")
    
    try:
        df = pd.read_csv(FINAL_RESULTS_CSV_PATH)
        print(f"Loaded {len(df)} rows for evaluation")
    except Exception as e:
        print(f"Error loading results file for evaluation: {e}")
        return
    
    # Extract boolean answers with improved parsing
    print("Extracting boolean answers...")
    
    try:
        df['original_answer'] = df['original_answer_raw'].apply(extract_strategyqa_answer)
        df['llama_direct_answer'] = df['llama_direct_raw'].apply(extract_llm_boolean_answer)
        df['llama_cot_answer'] = df['llama_cot_raw'].apply(extract_llm_boolean_answer)
        df['llama_corrupted_cot_answer'] = df['llama_corrupted_cot_raw'].apply(extract_llm_boolean_answer)
        df['llama_partial_cot_answer'] = df['llama_partial_cot_response'].apply(extract_llm_boolean_answer)
        df['phi_direct_answer'] = df['phi_direct_raw'].apply(extract_llm_boolean_answer)  # Updated column names
        df['phi_with_cot_answer'] = df['phi_with_cot_raw'].apply(extract_llm_boolean_answer)
        df['phi_with_corrupted_cot_answer'] = df['phi_with_corrupted_cot_raw'].apply(extract_llm_boolean_answer)
        df['phi_with_partial_cot_answer'] = df['phi_with_partial_cot_raw'].apply(extract_llm_boolean_answer)
    except Exception as e:
        print(f"Error extracting answers: {e}")
        return

    # Save updated dataframe
    try:
        df.to_csv(FINAL_RESULTS_CSV_PATH, index=False)
        print(f"Updated CSV saved with {len(df)} rows")
    except Exception as e:
        print(f"Error saving updated CSV: {e}")

    # Debug: Show some examples of answer extraction
    print("\nSample Boolean Answer Extractions:")
    print("-" * 50)
    for i in range(min(3, len(df))):
        try:
            print(f"Row {i}:")
            print(f"  Original: {df.iloc[i]['original_answer']}")
            phi_raw = str(df.iloc[i].get('phi_direct_raw', ''))[:100]  # Updated variable name
            print(f"  Phi Direct Raw: {phi_raw}...")
            print(f"  Phi Direct Answer: {df.iloc[i]['phi_direct_answer']}")
            print()
        except Exception as e:
            print(f"Error displaying sample {i}: {e}")

    # Compute accuracies
    print("Computing accuracies...")
    settings = {
        "LLaMA Direct vs. Original": 'llama_direct_answer',
        "LLaMA CoT vs. Original": 'llama_cot_answer',
        "LLaMA Corrupted CoT vs. Original": 'llama_corrupted_cot_answer',
        "LLaMA Partial CoT vs. Original": 'llama_partial_cot_answer',
        "Phi Direct vs. Original": 'phi_direct_answer',  # Updated labels
        "Phi with CoT vs. Original": 'phi_with_cot_answer',
        "Phi with Corrupted CoT vs. Original": 'phi_with_corrupted_cot_answer',
        "Phi with Partial CoT vs. Original": 'phi_with_partial_cot_answer',
    }
    
    results = []
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    for name, col in settings.items():
        try:
            # Handle None values in comparison for boolean data
            if col in df.columns and 'original_answer' in df.columns:
                valid_mask = df[col].notna() & df['original_answer'].notna()
                if valid_mask.sum() > 0:
                    accuracy = (df.loc[valid_mask, col] == df.loc[valid_mask, 'original_answer']).mean()
                    valid_count = valid_mask.sum()
                    total_count = len(df)
                else:
                    accuracy = 0.0
                    valid_count = 0
                    total_count = len(df)
            else:
                accuracy = 0.0
                valid_count = 0
                total_count = len(df)
            
            results.append({
                'setting': name, 
                'accuracy': accuracy,
                'valid_responses': valid_count,
                'total_questions': total_count
            })
            print(f"{name}: {accuracy:.2%}")
        except Exception as e:
            print(f"Error computing accuracy for {name}: {e}")
            results.append({
                'setting': name, 
                'accuracy': 0.0,
                'valid_responses': 0,
                'total_questions': len(df)
            })

    # Additional statistics for boolean classification
    print("\nDetailed Statistics:")
    print("-" * 50)
    
    for name, col in settings.items():
        try:
            if col in df.columns and 'original_answer' in df.columns:
                valid_mask = df[col].notna() & df['original_answer'].notna()
                if valid_mask.sum() > 0:
                    true_predictions = df.loc[valid_mask, col].sum()
                    false_predictions = valid_mask.sum() - true_predictions
                    true_actual = df.loc[valid_mask, 'original_answer'].sum()
                    false_actual = valid_mask.sum() - true_actual
                    
                    print(f"{name}:")
                    print(f"  Predicted True: {true_predictions}, False: {false_predictions}")
                    print(f"  Actual True: {true_actual}, False: {false_actual}")
                    print(f"  Valid responses: {valid_mask.sum()}/{len(df)}")
                    print()
        except Exception as e:
            print(f"Error computing detailed stats for {name}: {e}")

    # Save summary
    try:
        pd.DataFrame(results).to_csv(SUMMARY_CSV_PATH, index=False)
        print(f"Summary saved to: {SUMMARY_CSV_PATH}")
    except Exception as e:
        print(f"Error saving summary: {e}")
    
    print("Processing completed successfully.")

if __name__ == "__main__":
    main()
