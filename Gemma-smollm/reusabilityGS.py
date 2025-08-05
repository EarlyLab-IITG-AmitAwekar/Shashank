import requests
import json
import re
import pandas as pd
from tqdm import tqdm
import time
import os



# --- Configuration ---
API_URL = 'http://127.0.0.1:11434/api/generate'
GEMMA3_MODEL = 'smollm2:1.7b'  # Secondary model remains SmolLM2
GEMMA_RESULTS_CSV_PATH = '../gemma3_gsm8k_results.csv'  # Changed from llama to gemma3
FINAL_RESULTS_CSV_PATH = 'gemma3_smollm2_gsm8k_final_results.csv'  # Updated filename
SUMMARY_CSV_PATH = 'gemma3_smollm2_gsm8k_accuracy_summary.csv'  # Updated filename
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



PROMPT_SMOLLM2_WITH_COT = """  # Renamed for clarity
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
        return response.json().get('response', '').strip()
    except requests.exceptions.RequestException as e:
        return f"Error: API request failed: {str(e)}"



def extract_llm_answer(text: str):
    if not text or "Error:" in text:
        return None
    m = re.search(r'####(.*)', text)
    if m:
        num = re.search(r'[-+]?\d*\.?\d+', m.group(1))
        if num:
            return float(num.group(0))
    nums = re.findall(r'[-+]?\d*\.?\d+', text)
    return float(nums[-1]) if nums else None



def extract_gsm8k_answer(text: str):
    if not text:
        return None
    m = re.search(r'####(.*)', text)
    if m:
        num = re.search(r'[-+]?\d*\.?\d+', m.group(1))
        if num:
            return float(num.group(0))
    nums = re.findall(r'[-+]?\d*\.?\d+', text)
    return float(nums[-1]) if nums else None



def main():
    """Load Gemma 3 data, collect SmolLM2 responses, combine and evaluate"""  # Updated description
    print("=== Gemma 3 + SmolLM2 Processing and Evaluation Script ===")  # Updated print statement



    # Verify Gemma 3 results exist
    if not os.path.exists(GEMMA_RESULTS_CSV_PATH):
        print(f"Error: Gemma 3 results file '{GEMMA_RESULTS_CSV_PATH}' not found.")
        print("Please run 'collect_gemma3_data.py' first.")
        return



    # Overwrite check
    if os.path.exists(FINAL_RESULTS_CSV_PATH):
        resp = input(f"File '{FINAL_RESULTS_CSV_PATH}' exists. Overwrite? (y/n): ")
        if resp.lower() != 'y':
            print("Exiting without overwriting.")
            return
        os.remove(FINAL_RESULTS_CSV_PATH)



    # Load and process
    df_gemma3 = pd.read_csv(GEMMA_RESULTS_CSV_PATH)  # Updated variable name
    combined = []
    for idx, row in tqdm(df_gemma3.iterrows(), total=len(df_gemma3), desc="Processing SmolLM2"):  # Updated description
        q = row['question']
        cot_full = row['gemma3_cot_steps_only']  # Updated column names
        cot_corrupt = row['corrupted_cot_steps_only']
        cot_partial = row['partial_cot_steps']



        # SmolLM2 responses
        resp_direct = generate_llm_response(GEMMA3_MODEL, PROMPT_DIRECT.format(question=q))
        resp_full   = generate_llm_response(GEMMA3_MODEL, PROMPT_SMOLLM2_WITH_COT.format(question=q, cot=cot_full))
        resp_corrupt= generate_llm_response(GEMMA3_MODEL, PROMPT_SMOLLM2_WITH_COT.format(question=q, cot=cot_corrupt))
        resp_partial= generate_llm_response(GEMMA3_MODEL, PROMPT_SMOLLM2_WITH_COT.format(question=q, cot=cot_partial))



        combined.append({
            'question':                 q,
            'original_answer_raw':      row['original_answer_raw'],
            'gemma3_direct_raw':         row['gemma3_direct_raw'],  # Updated column names
            'gemma3_cot_raw':            row['gemma3_cot_raw'],
            'gemma3_corrupted_cot_raw':  row['gemma3_corrupted_cot_raw'],
            'gemma3_cot_steps_only':     cot_full,
            'corrupted_cot_steps_only': cot_corrupt,
            'partial_cot_steps':        cot_partial,
            'gemma3_partial_cot_response':row['gemma3_partial_cot_response'],
            'smollm2_direct_raw':           resp_direct,  # Updated column names
            'smollm2_with_cot_raw':         resp_full,
            'smollm2_with_corrupted_cot_raw':resp_corrupt,
            'smollm2_with_partial_cot_raw': resp_partial,
        })



        # Periodic save
        if (idx+1) % CHUNK_SAVE_SIZE == 0 or (idx+1)==len(df_gemma3):  # Updated variable name
            df_chunk = pd.DataFrame(combined)
            header = not os.path.exists(FINAL_RESULTS_CSV_PATH)
            df_chunk.to_csv(FINAL_RESULTS_CSV_PATH, mode='a', header=header, index=False)
            combined = []
        time.sleep(0.5)



    # Evaluation
    df = pd.read_csv(FINAL_RESULTS_CSV_PATH)
    df['original_answer']        = df['original_answer_raw'].apply(extract_gsm8k_answer)
    df['gemma3_direct_answer']    = df['gemma3_direct_raw'].apply(extract_llm_answer)  # Updated column names
    df['gemma3_cot_answer']       = df['gemma3_cot_raw'].apply(extract_llm_answer)
    df['gemma3_corrupted_cot_answer'] = df['gemma3_corrupted_cot_raw'].apply(extract_llm_answer)
    df['gemma3_partial_cot_answer']   = df['gemma3_partial_cot_response'].apply(extract_llm_answer)
    df['smollm2_direct_answer']      = df['smollm2_direct_raw'].apply(extract_llm_answer)  # Updated column names
    df['smollm2_with_cot_answer']    = df['smollm2_with_cot_raw'].apply(extract_llm_answer)
    df['smollm2_with_corrupted_cot_answer'] = df['smollm2_with_corrupted_cot_raw'].apply(extract_llm_answer)
    df['smollm2_with_partial_cot_answer']   = df['smollm2_with_partial_cot_raw'].apply(extract_llm_answer)



    # Compute accuracies
    settings = {
        "Gemma 3 Direct vs. Original":       'gemma3_direct_answer',  # Updated labels
        "Gemma 3 CoT vs. Original":          'gemma3_cot_answer',
        "Gemma 3 Corrupted CoT vs. Original":'gemma3_corrupted_cot_answer',
        "Gemma 3 Partial CoT vs. Original":  'gemma3_partial_cot_answer',
        "SmolLM2 Direct vs. Original":         'smollm2_direct_answer',  # Updated labels
        "SmolLM2 with CoT vs. Original":       'smollm2_with_cot_answer',
        "SmolLM2 with Corrupted CoT vs. Original":'smollm2_with_corrupted_cot_answer',
        "SmolLM2 with Partial CoT vs. Original":   'smollm2_with_partial_cot_answer',
    }
    results = []
    for name, col in settings.items():
        accuracy = (df[col] == df['original_answer']).mean()
        results.append({'setting': name, 'accuracy': accuracy})
        print(f"- {name}: {accuracy:.2%}")



    # Save summary
    pd.DataFrame(results).to_csv(SUMMARY_CSV_PATH, index=False)
    print("Processing completed successfully.")



if __name__ == "__main__":
    main()
