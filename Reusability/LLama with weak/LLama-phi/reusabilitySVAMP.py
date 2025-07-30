import requests
import json
import re
import pandas as pd
from tqdm import tqdm
import time
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from functools import partial

# --- Configuration ---
API_URL = 'http://127.0.0.1:11434/api/generate'
PHI_MODEL = 'phi:2.7b'
LLAMA_RESULTS_CSV_PATH = '../../llama_svamp_results.csv'
FINAL_RESULTS_CSV_PATH = 'llama_phi_svamp_final_results.csv'
SUMMARY_CSV_PATH = 'llama_phi_svamp_accuracy_summary.csv'
CHUNK_SAVE_SIZE = 20  # Save every 20 data points
MAX_WORKERS = min(8, mp.cpu_count())  # Parallel processing workers

# --- Prompt Templates ---

PROMPT_DIRECT = """
You are an expert mathematician and a highly accurate calculator.
Your task is to solve the given math problem.
Think through the problem systematically to ensure the correct solution.

Do not include any reasoning, steps, explanations, or additional text.
Provide only the final numerical answer in the following exact format:

####<number>

Question: {question}

Final Answer:"""

PROMPT_PHI_WITH_COT = """
You are not allowed to use your own reasoning or knowledge. You must ONLY use the provided reasoning (Chain of Thought) below to answer the question. Do not add, change, or infer anything. Just follow the steps exactly as given, even if they seem wrong or incomplete.

Continue from where the provided reasoning stops and complete the solution to provide the final numerical answer in format: ####<number>. Do not provide any other text, reasoning, or symbols.

Question: {question}

Reasoning (Chain of Thought):
{cot}
"""

# --- Helper Functions ---

def generate_llm_response(llm_name: str, prompt: str) -> str:
    """Generate response from LLM with error handling"""
    payload = {"model": llm_name, "prompt": prompt, "stream": False}
    try:
        response = requests.post(API_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json().get('response', '').strip()
        return result if result else "Error: Empty response"
    except requests.exceptions.RequestException as e:
        return f"Error: API request failed: {str(e)}"

def process_single_question(args):
    """Process a single question for parallel execution"""
    idx, row_data = args
    q, cot_full, cot_corrupt, cot_partial = row_data
    
    try:
        # Generate Phi responses
        resp_direct = generate_llm_response(PHI_MODEL, PROMPT_DIRECT.format(question=q))
        resp_full = generate_llm_response(PHI_MODEL, PROMPT_PHI_WITH_COT.format(question=q, cot=cot_full))
        resp_corrupt = generate_llm_response(PHI_MODEL, PROMPT_PHI_WITH_COT.format(question=q, cot=cot_corrupt))
        resp_partial = generate_llm_response(PHI_MODEL, PROMPT_PHI_WITH_COT.format(question=q, cot=cot_partial))
        
        return idx, {
            'phi_direct_raw': resp_direct,
            'phi_with_cot_raw': resp_full,
            'phi_with_corrupted_cot_raw': resp_corrupt,
            'phi_with_partial_cot_raw': resp_partial,
        }
    except Exception as e:
        return idx, {
            'phi_direct_raw': f"Error: {str(e)}",
            'phi_with_cot_raw': f"Error: {str(e)}",
            'phi_with_corrupted_cot_raw': f"Error: {str(e)}",
            'phi_with_partial_cot_raw': f"Error: {str(e)}",
        }

def extract_llm_answer(text):
    """Extract numerical answer from LLM response with proper error handling"""
    # Handle NaN, None, or non-string inputs
    if pd.isna(text) or text is None:
        return None
    
    # Convert to string if it's not already
    if not isinstance(text, str):
        text = str(text)
    
    # Handle empty strings or error messages
    if not text.strip() or "Error:" in text:
        return None
    
    # First, try to find answer in #### format
    m = re.search(r'####\s*([-+]?\d*\.?\d+)', text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    
    # If no #### format found, look for numbers in the text
    # Try to find the last number that appears to be a final answer
    nums = re.findall(r'[-+]?\d*\.?\d+', text)
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            pass
    
    return None

def extract_svamp_answer(text):
    """Extract numerical answer from SVAMP format with proper error handling"""
    # Handle NaN, None, or non-string inputs
    if pd.isna(text) or text is None:
        return None
    
    # Convert to string if it's not already
    if not isinstance(text, str):
        text = str(text)
    
    if not text.strip():
        return None
    
    # Look for #### format first
    m = re.search(r'####\s*([-+]?\d*\.?\d+)', text)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    
    # For SVAMP, the answer might just be a number
    try:
        return float(text.strip())
    except ValueError:
        pass
    
    # Fallback to last number in text
    nums = re.findall(r'[-+]?\d*\.?\d+', text)
    if nums:
        try:
            return float(nums[-1])
        except ValueError:
            pass
    
    return None

def safe_get_column_value(row, column_name, default_value=""):
    """Safely get column value, handling NaN and missing columns"""
    try:
        value = row[column_name]
        if pd.isna(value):
            return default_value
        return str(value)
    except (KeyError, IndexError):
        return default_value

def main():
    """Load LLaMA data, collect Phi responses, combine and evaluate"""
    print("=== Phi Processing and Evaluation Script (SVAMP Dataset) ===")

    # Verify LLaMA results exist
    if not os.path.exists(LLAMA_RESULTS_CSV_PATH):
        print(f"Error: LLaMA results file '{LLAMA_RESULTS_CSV_PATH}' not found.")
        print("Please run the LLaMA SVAMP data collection script first.")
        return

    # Overwrite check
    if os.path.exists(FINAL_RESULTS_CSV_PATH):
        resp = input(f"File '{FINAL_RESULTS_CSV_PATH}' exists. Overwrite? (y/n): ")
        if resp.lower() != 'y':
            print("Exiting without overwriting.")
            return
        os.remove(FINAL_RESULTS_CSV_PATH)

    # Load and process
    df_llama = pd.read_csv(LLAMA_RESULTS_CSV_PATH)
    print(f"Loaded {len(df_llama)} rows from LLaMA SVAMP results")
    
    # Prepare data for parallel processing
    print(f"Preparing data for parallel processing with {MAX_WORKERS} workers...")
    
    processed_data = []
    for idx, row in df_llama.iterrows():
        q = safe_get_column_value(row, 'question')
        if not q.strip():
            continue
        
        cot_full = safe_get_column_value(row, 'llama_cot_steps_only')
        cot_corrupt = safe_get_column_value(row, 'corrupted_cot_steps_only')
        cot_partial = safe_get_column_value(row, 'partial_cot_steps')
        
        processed_data.append({
            'idx': idx,
            'row': row,
            'question_data': (q, cot_full, cot_corrupt, cot_partial)
        })

    # Process in parallel batches
    combined = []
    failed_requests = 0
    batch_size = MAX_WORKERS * 2
    
    for batch_start in range(0, len(processed_data), batch_size):
        batch_end = min(batch_start + batch_size, len(processed_data))
        batch_data = processed_data[batch_start:batch_end]
        
        print(f"Processing batch {batch_start//batch_size + 1}/{(len(processed_data) + batch_size - 1)//batch_size}")
        
        # Prepare arguments for parallel processing
        args_list = [(item['idx'], item['question_data']) for item in batch_data]
        
        # Process batch in parallel
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_idx = {executor.submit(process_single_question, args): args[0] for args in args_list}
            
            for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Processing Phi responses"):
                try:
                    idx, phi_results = future.result()
                    
                    # Find corresponding row data
                    item = next(item for item in batch_data if item['idx'] == idx)
                    row = item['row']
                    
                    combined.append({
                        'question': safe_get_column_value(row, 'question'),
                        'original_answer_raw': safe_get_column_value(row, 'original_answer_raw'),
                        'llama_direct_raw': safe_get_column_value(row, 'llama_direct_raw'),
                        'llama_cot_raw': safe_get_column_value(row, 'llama_cot_raw'),
                        'llama_corrupted_cot_raw': safe_get_column_value(row, 'llama_corrupted_cot_raw'),
                        'llama_cot_steps_only': safe_get_column_value(row, 'llama_cot_steps_only'),
                        'corrupted_cot_steps_only': safe_get_column_value(row, 'corrupted_cot_steps_only'),
                        'partial_cot_steps': safe_get_column_value(row, 'partial_cot_steps'),
                        'llama_partial_cot_response': safe_get_column_value(row, 'llama_partial_cot_response'),
                        **phi_results
                    })
                except Exception as e:
                    print(f"Error processing future: {str(e)}")
                    failed_requests += 1

        # Save batch
        if combined:
            df_chunk = pd.DataFrame(combined)
            header = not os.path.exists(FINAL_RESULTS_CSV_PATH)
            df_chunk.to_csv(FINAL_RESULTS_CSV_PATH, mode='a', header=header, index=False)
            print(f"Saved batch {batch_start//batch_size + 1} with {len(combined)} rows")
            combined = []

    if failed_requests > 0:
        print(f"Warning: {failed_requests} requests failed")

    # Evaluation
    print("Starting evaluation...")
    df = pd.read_csv(FINAL_RESULTS_CSV_PATH)
    
    # Extract answers with proper error handling
    print("Extracting answers...")
    df['original_answer'] = df['original_answer_raw'].apply(extract_svamp_answer)
    df['llama_direct_answer'] = df['llama_direct_raw'].apply(extract_llm_answer)
    df['llama_cot_answer'] = df['llama_cot_raw'].apply(extract_llm_answer)
    df['llama_corrupted_cot_answer'] = df['llama_corrupted_cot_raw'].apply(extract_llm_answer)
    df['llama_partial_cot_answer'] = df['llama_partial_cot_response'].apply(extract_llm_answer)
    df['phi_direct_answer'] = df['phi_direct_raw'].apply(extract_llm_answer)
    df['phi_with_cot_answer'] = df['phi_with_cot_raw'].apply(extract_llm_answer)
    df['phi_with_corrupted_cot_answer'] = df['phi_with_corrupted_cot_raw'].apply(extract_llm_answer)
    df['phi_with_partial_cot_answer'] = df['phi_with_partial_cot_raw'].apply(extract_llm_answer)

    # Save updated dataframe
    df.to_csv(FINAL_RESULTS_CSV_PATH, index=False)
    print(f"Updated CSV saved with {len(df)} rows")

    # Compute accuracies
    print("Computing accuracies...")
    settings = {
        "LLaMA Direct vs. Original": 'llama_direct_answer',
        "LLaMA CoT vs. Original": 'llama_cot_answer',
        "LLaMA Corrupted CoT vs. Original": 'llama_corrupted_cot_answer',
        "LLaMA Partial CoT vs. Original": 'llama_partial_cot_answer',
        "Phi Direct vs. Original": 'phi_direct_answer',
        "Phi with CoT vs. Original": 'phi_with_cot_answer',
        "Phi with Corrupted CoT vs. Original": 'phi_with_corrupted_cot_answer',
        "Phi with Partial CoT vs. Original": 'phi_with_partial_cot_answer',
    }
    
    results = []
    print("\nAccuracy Results:")
    print("-" * 50)
    
    for name, col in settings.items():
        # Handle NaN values in comparison
        valid_mask = df[col].notna() & df['original_answer'].notna()
        if valid_mask.sum() > 0:
            accuracy = (df.loc[valid_mask, col] == df.loc[valid_mask, 'original_answer']).mean()
            valid_count = valid_mask.sum()
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
        print(f"- {name}: {accuracy:.2%} ({valid_count}/{total_count})")

    # Save summary
    pd.DataFrame(results).to_csv(SUMMARY_CSV_PATH, index=False)
    print(f"\nSummary saved to: {SUMMARY_CSV_PATH}")
    print("Processing completed successfully.")

if __name__ == "__main__":
    main()
