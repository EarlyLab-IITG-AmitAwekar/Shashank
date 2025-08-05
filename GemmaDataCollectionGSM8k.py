import requests
import json
import re
import pandas as pd
from tqdm import tqdm
import time
import os


# --- Configuration ---
API_URL = 'http://127.0.0.1:11434/api/generate'
GEMMA_MODEL = 'gemma3:27b'  # Corrected to Gemma 3 27B
GEMMA_RESULTS_CSV_PATH = 'gemma3_gsm8k_results.csv'  # Updated file name
CHUNK_SAVE_SIZE = 20  # Save every 20 data points


# --- Prompt Templates ---


PROMPT_GEMMA_COT = """
You are a precise and logical math assistant. Solve the following problem by thinking step-by-step.
Your reasoning should clearly break down the problem into individual steps and calculations.
After presenting your complete reasoning, state the final answer in the specific format 'The final answer is ####<number>'.


Here is an example of the desired format and level of detail:
---
Question: Natalia sold 48 cupcakes on Monday. On Tuesday, she sold half as many as on Monday. On Wednesday, she sold 10 more than on Tuesday. How many cupcakes did she sell in total?
Reasoning:
1. First, calculate the number of cupcakes sold on Tuesday. This is half of Monday's sales: 48 / 2 = 24 cupcakes.
2. Next, calculate the number of cupcakes sold on Wednesday. This is 10 more than Tuesday's sales: 24 + 10 = 34 cupcakes.
3. Finally, sum the cupcakes sold each day to find the total: 48 (Monday) + 24 (Tuesday) + 34 (Wednesday) = 106 cupcakes.
The final answer is ####106
---
Now, solve this problem:
Question: {question}
Reasoning:
"""


PROMPT_DIRECT = """
You are an expert mathematician and a highly accurate calculator.
Your task is to solve the given math problem.
Think through the problem systematically to ensure the correct solution.


Do not include any reasoning, steps, explanations, or additional text.
Provide only the final numerical answer in the following exact format:


####<number>


Question: {question}


Final Answer:"""


PROMPT_GEMMA_CORRUPT_COT = """
Given the math question: "{question}", provide a step-by-step reasoning process (Chain of Thought) to solve it. The steps must:


- Include deliberate mathematical errors (wrong operations, misplaced signs, incorrect assumptions, etc.)
- Present steps in an incorrect or shuffled order (not logically flowing)
- End with a final answer that is always incorrect
- Do not mention that the reasoning is wrong
- Do not correct any step
- Output only the steps and final answer (no intro, explanation, or conclusion)


Final Answer format: ####<number>
"""


PROMPT_GEMMA_WITH_PARTIAL_COT = """
You are a precise and logical math assistant. Below is a partially completed reasoning for a math problem. Continue from where the reasoning stops and complete the solution to provide the final numerical answer.


Question: {question}


Partial Reasoning:
{partial_cot}


Continue the reasoning and provide the final answer in the format: ####<number>
"""


# --- Helper Functions ---


def remove_final_answer(cot_text):
    """Remove the final answer line from CoT text, keeping only the reasoning steps."""
    if cot_text is None:
        return ""
    
    lines = cot_text.strip().split('\n')
    filtered_lines = []
    
    for line in lines:
        # Stop at any line that contains final answer indicators
        if re.search(r'(The final answer is|Final Answer:|####)', line, re.IGNORECASE):
            break
        filtered_lines.append(line)
    
    return '\n'.join(filtered_lines).strip()


def get_partial_cot(cot_text, fraction=0.5):
    """Get partial steps from the CoT text, excluding final answer."""
    cot_text_no_answer = remove_final_answer(cot_text)
    lines = cot_text_no_answer.strip().split('\n')
    # Filter out empty lines
    lines = [line for line in lines if line.strip()]
    # Calculate number of lines to keep
    n = max(1, int(len(lines) * fraction))
    partial_cot = '\n'.join(lines[:n])
    return partial_cot


def download_gsm8k_full() -> list:
    print("Downloading GSM8K dataset...")
    GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
    try:
        response = requests.get(GSM8K_URL)
        response.raise_for_status()
        data = [json.loads(line) for line in response.text.strip().split('\n') if line]
        print(f"Successfully downloaded {len(data)} questions.")
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error downloading dataset: {e}")
        return []


def generate_llm_response(llm_name: str, prompt: str) -> str:
    payload = {"model": llm_name, "prompt": prompt, "stream": False}
    try:
        response = requests.post(API_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return result.get('response', '').strip()
    except requests.exceptions.RequestException as e:
        return f"Error: API request failed: {str(e)}"


def main():
    """Collect all Gemma 3 responses and save to CSV"""
    print("=== Gemma 3 Data Collection Script ===")  # Updated title
    
    # Check if file already exists
    if os.path.exists(GEMMA_RESULTS_CSV_PATH):
        response = input(f"File '{GEMMA_RESULTS_CSV_PATH}' already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Exiting without overwriting.")
            return
        os.remove(GEMMA_RESULTS_CSV_PATH)
    
    questions = download_gsm8k_full()
    if not questions: 
        return

    results = []
    
    print(f"\nStarting Gemma 3 data collection with {len(questions)} questions...")  # Updated description

    for idx, q in enumerate(tqdm(questions, desc="Processing Gemma 3 Questions")):  # Updated description
        question_text = q['question']
        original_answer_raw = q.get('answer', '')
        
        print(f"\nProcessing question {idx + 1}/{len(questions)}")
        print(f"Question: {question_text[:100]}...")
        
        # Gemma 3 generates CoT, direct answer, and corrupted CoT
        gemma_cot = generate_llm_response(GEMMA_MODEL, PROMPT_GEMMA_COT.format(question=question_text))
        gemma_direct_raw = generate_llm_response(GEMMA_MODEL, PROMPT_DIRECT.format(question=question_text))
        corrupted_cot = generate_llm_response(GEMMA_MODEL, PROMPT_GEMMA_CORRUPT_COT.format(question=question_text))

        # Process CoTs to remove final answers
        gemma_cot_steps_only = remove_final_answer(gemma_cot)
        corrupted_cot_steps_only = remove_final_answer(corrupted_cot)
        partial_cot_steps = get_partial_cot(gemma_cot, fraction=0.5)

        # Gemma 3 with partial CoT
        gemma_partial_cot_response = generate_llm_response(GEMMA_MODEL, PROMPT_GEMMA_WITH_PARTIAL_COT.format(question=question_text, partial_cot=partial_cot_steps))

        results.append({
            'question': question_text,
            'original_answer_raw': original_answer_raw,
            'gemma3_direct_raw': gemma_direct_raw,  # Updated column names with "3"
            'gemma3_cot_raw': gemma_cot,
            'gemma3_corrupted_cot_raw': corrupted_cot,
            'gemma3_cot_steps_only': gemma_cot_steps_only,
            'corrupted_cot_steps_only': corrupted_cot_steps_only,
            'partial_cot_steps': partial_cot_steps,
            'gemma3_partial_cot_response': gemma_partial_cot_response,
        })

        # Save results to CSV every CHUNK_SAVE_SIZE entries
        if (idx + 1) % CHUNK_SAVE_SIZE == 0 or (idx + 1) == len(questions):
            df_chunk = pd.DataFrame(results)
            
            # Append to the CSV if file exists, else create new
            if os.path.exists(GEMMA_RESULTS_CSV_PATH):
                df_chunk.to_csv(GEMMA_RESULTS_CSV_PATH, mode='a', header=False, index=False)
                print(f"Appended chunk {((idx + 1) // CHUNK_SAVE_SIZE)} to existing Gemma 3 CSV")  # Updated message
            else:
                df_chunk.to_csv(GEMMA_RESULTS_CSV_PATH, index=False)
                print(f"Created new Gemma 3 CSV with first chunk")  # Updated message
            
            results = []  # Reset list after saving

        time.sleep(0.5)

    print(f"\nGemma 3 data collection completed!")  # Updated message
    print(f"Results saved to: '{GEMMA_RESULTS_CSV_PATH}'")
    print(f"Total questions processed: {len(questions)}")


if __name__ == "__main__":
    main()
