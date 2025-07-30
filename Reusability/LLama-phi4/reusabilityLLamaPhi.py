import requests
import json
import re
import pandas as pd
from tqdm import tqdm
import time
import os

# --- Configuration ---
API_URL = 'http://127.0.0.1:11434/api/generate'
PHI4_MODEL = 'phi4:14b'
LLAMA_RESULTS_CSV_PATH = '../llama_gsm8k_results.csv'
FINAL_RESULTS_CSV_PATH = 'llama_phi4_gsm8k_final_results.csv'
SUMMARY_CSV_PATH = 'llama_phi4_gsm8k_accuracy_summary.csv'
CHUNK_SAVE_SIZE = 20  # Save every 20 data points

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

PROMPT_PHI4_WITH_COT = """
You are not allowed to use your own reasoning or knowledge. You must ONLY use the provided reasoning (Chain of Thought) below to answer the question. Do not add, change, or infer anything. Just follow the steps exactly as given, even if they seem wrong or incomplete.

Continue from where the provided reasoning stops and complete the solution to provide the final numerical answer in format: ####<number>. Do not provide any other text, reasoning, or symbols.

Question: {question}

Reasoning (Chain of Thought):
{cot}
"""

# --- Helper Functions ---

def generate_llm_response(llm_name: str, prompt: str) -> str:
    payload = {"model": llm_name, "prompt": prompt, "stream": False}
    try:
        response = requests.post(API_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get('response', '').strip()
    except requests.exceptions.RequestException as e:
        return f"Error: API request failed: {str(e)}"

def extract_llm_answer(text: str):
    if text is None or "Error:" in str(text):
        return None
    m = re.search(r'####(.*)', text)
    if m:
        after_hashes = m.group(1)
        num_match = re.search(r'[-+]?\d*\.?\d+', after_hashes)
        if num_match:
            try:
                return float(num_match.group(0))
            except ValueError:
                pass
    all_nums = re.findall(r'[-+]?\d*\.?\d+', text)
    if all_nums:
        try:
            return float(all_nums[-1])
        except ValueError:
            pass
    return None

def extract_gsm8k_answer(text: str):
    if text is None:
        return None
    m = re.search(r'####(.*)', text)
    if m:
        after_hashes = m.group(1)
        num_match = re.search(r'[-+]?\d*\.?\d+', after_hashes)
        if num_match:
            try:
                return float(num_match.group(0))
            except ValueError:
                pass
    all_nums = re.findall(r'[-+]?\d*\.?\d+', text)
    if all_nums:
        try:
            return float(all_nums[-1])
        except ValueError:
            pass
    return None

def main():
    """Load LLaMA data, collect Phi4 responses, combine and evaluate"""
    print("=== Phi4 Processing and Evaluation Script ===")
    
    # Check if LLaMA results exist
    if not os.path.exists(LLAMA_RESULTS_CSV_PATH):
        print(f"Error: LLaMA results file '{LLAMA_RESULTS_CSV_PATH}' not found.")
        print("Please run 'collect_llama_data.py' first.")
        return
    
    # Check if final results file already exists
    if os.path.exists(FINAL_RESULTS_CSV_PATH):
        response = input(f"File '{FINAL_RESULTS_CSV_PATH}' already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Exiting without overwriting.")
            return
        os.remove(FINAL_RESULTS_CSV_PATH)
    
    # Load LLaMA data
    print(f"Loading LLaMA data from '{LLAMA_RESULTS_CSV_PATH}'...")
    df_llama = pd.read_csv(LLAMA_RESULTS_CSV_PATH)
    print(f"Loaded {len(df_llama)} rows from LLaMA results.")
    
    # Collect Phi4 responses and combine data
    combined_results = []
    
    print("\nStarting Phi4 data collection and combination...")

    for idx, row in tqdm(df_llama.iterrows(), total=len(df_llama), desc="Processing Phi4 Questions"):
        question_text = row['question']
        
        print(f"\nProcessing question {idx + 1}/{len(df_llama)}")
        print(f"Question: {question_text[:100]}...")
        
        # Get processed CoTs from LLaMA data
        llama_cot_steps_only = row['llama_cot_steps_only']
        corrupted_cot_steps_only = row['corrupted_cot_steps_only']
        partial_cot_steps = row['partial_cot_steps']
        
        # Phi4 responses using LLaMA's CoTs
        phi4_direct_raw = generate_llm_response(PHI4_MODEL, PROMPT_DIRECT.format(question=question_text))
        phi4_with_cot = generate_llm_response(PHI4_MODEL, PROMPT_PHI4_WITH_COT.format(question=question_text, cot=llama_cot_steps_only))
        phi4_with_corrupted_cot = generate_llm_response(PHI4_MODEL, PROMPT_PHI4_WITH_COT.format(question=question_text, cot=corrupted_cot_steps_only))
        phi4_with_partial_cot = generate_llm_response(PHI4_MODEL, PROMPT_PHI4_WITH_COT.format(question=question_text, cot=partial_cot_steps))

        # Combine LLaMA and Phi4 data for this question
        combined_row = {
            # Original data
            'question': question_text,
            'original_answer_raw': row['original_answer_raw'],
            
            # LLaMA data
            'llama_direct_raw': row['llama_direct_raw'],
            'llama_cot_raw': row['llama_cot_raw'],
            'llama_corrupted_cot_raw': row['llama_corrupted_cot_raw'],
            'llama_cot_steps_only': row['llama_cot_steps_only'],
            'corrupted_cot_steps_only': row['corrupted_cot_steps_only'],
            'partial_cot_steps': row['partial_cot_steps'],
            'llama_partial_cot_response': row['llama_partial_cot_response'],
            
            # Phi4 data
            'phi4_direct_raw': phi4_direct_raw,
            'phi4_with_cot_raw': phi4_with_cot,
            'phi4_with_corrupted_cot_raw': phi4_with_corrupted_cot,
            'phi4_with_partial_cot_raw': phi4_with_partial_cot,
        }
        
        combined_results.append(combined_row)

        # Save results to CSV every CHUNK_SAVE_SIZE entries
        if (idx + 1) % CHUNK_SAVE_SIZE == 0 or (idx + 1) == len(df_llama):
            df_chunk = pd.DataFrame(combined_results)
            
            # Append to the CSV if file exists, else create new
            if os.path.exists(FINAL_RESULTS_CSV_PATH):
                df_chunk.to_csv(FINAL_RESULTS_CSV_PATH, mode='a', header=False, index=False)
                print(f"Appended chunk {((idx + 1) // CHUNK_SAVE_SIZE)} to final results CSV")
            else:
                df_chunk.to_csv(FINAL_RESULTS_CSV_PATH, index=False)
                print(f"Created new final results CSV with first chunk")
            
            combined_results = []  # Reset list after saving

        time.sleep(0.5)

    print(f"\nPhi4 data collection completed. Loading final results for evaluation...")
    
    # Load the complete combined dataset for evaluation
    df_combined = pd.read_csv(FINAL_RESULTS_CSV_PATH)
    
    # Extract answers from all responses
    print("Extracting answers from all responses...")
    df_combined['original_answer'] = df_combined['original_answer_raw'].apply(extract_gsm8k_answer)
    df_combined['llama_direct_answer'] = df_combined['llama_direct_raw'].apply(extract_llm_answer)
    df_combined['llama_cot_answer'] = df_combined['llama_cot_raw'].apply(extract_llm_answer)
    df_combined['llama_corrupted_cot_answer'] = df_combined['llama_corrupted_cot_raw'].apply(extract_llm_answer)
    df_combined['llama_partial_cot_answer'] = df_combined['llama_partial_cot_response'].apply(extract_llm_answer)
    df_combined['phi4_direct_answer'] = df_combined['phi4_direct_raw'].apply(extract_llm_answer)
    df_combined['phi4_with_cot_answer'] = df_combined['phi4_with_cot_raw'].apply(extract_llm_answer)
    df_combined['phi4_with_corrupted_cot_answer'] = df_combined['phi4_with_corrupted_cot_raw'].apply(extract_llm_answer)
    df_combined['phi4_with_partial_cot_answer'] = df_combined['phi4_with_partial_cot_raw'].apply(extract_llm_answer)

    # Calculate accuracy comparisons
    print("Calculating accuracies...")
    df_combined['llama_vs_original'] = df_combined['llama_direct_answer'] == df_combined['original_answer']
    df_combined['llama_cot_vs_original'] = df_combined['llama_cot_answer'] == df_combined['original_answer']
    df_combined['llama_corruptedCOT_vs_original'] = df_combined['llama_corrupted_cot_answer'] == df_combined['original_answer']
    df_combined['llama_partial_cot_vs_original'] = df_combined['llama_partial_cot_answer'] == df_combined['original_answer']
    df_combined['phi4_direct_vs_original'] = df_combined['phi4_direct_answer'] == df_combined['original_answer']
    df_combined['phi4_w_cot_vs_original'] = df_combined['phi4_with_cot_answer'] == df_combined['original_answer']
    df_combined['phi4_w_corrupt_cot_vs_original'] = df_combined['phi4_with_corrupted_cot_answer'] == df_combined['original_answer']
    df_combined['phi4_w_partial_cot_vs_original'] = df_combined['phi4_with_partial_cot_answer'] == df_combined['original_answer']

    # Calculate model accuracies
    model_accuracies = {
        "LLaMA Direct vs. Original": df_combined['llama_vs_original'].mean(),
        "LLaMA CoT vs. Original": df_combined['llama_cot_vs_original'].mean(),
        "LLaMA Corrupted CoT vs. Original": df_combined['llama_corruptedCOT_vs_original'].mean(),
        "LLaMA with Partial CoT vs. Original": df_combined['llama_partial_cot_vs_original'].mean(),
        "Phi4 Direct vs. Original": df_combined['phi4_direct_vs_original'].mean(),
        "Phi4 with LLaMA's CoT vs. Original": df_combined['phi4_w_cot_vs_original'].mean(),
        "Phi4 with LLaMA's Corrupted CoT vs. Original": df_combined['phi4_w_corrupt_cot_vs_original'].mean(),
        "Phi4 with LLaMA's Partial CoT vs. Original": df_combined['phi4_w_partial_cot_vs_original'].mean(),
    }

    # Display results
    print("\n" + "=" * 60)
    print("FINAL EXPERIMENT RESULTS")
    print("=" * 60)
    for name, acc in model_accuracies.items():
        print(f"  - {name}: {acc:.2%}")

    # Save updated final results with all evaluation columns
    df_combined.to_csv(FINAL_RESULTS_CSV_PATH, index=False)
    print(f"\nSaved final detailed results to '{FINAL_RESULTS_CSV_PATH}'")

    # Save accuracy summary
    summary_df = pd.DataFrame([
        {'setting': name, 'accuracy': acc} for name, acc in model_accuracies.items()
    ])
    summary_df.to_csv(SUMMARY_CSV_PATH, index=False)
    print(f"Saved summary accuracy results to '{SUMMARY_CSV_PATH}'")
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETED SUCCESSFULLY!")
    print(f"- Final combined results: {FINAL_RESULTS_CSV_PATH}")
    print(f"- Accuracy summary: {SUMMARY_CSV_PATH}")
    print(f"- Total questions processed: {len(df_combined)}")

if __name__ == "__main__":
    main()
