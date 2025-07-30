import requests
import json
import re
import pandas as pd
from tqdm import tqdm
import time
import os


# --- Configuration ---
API_URL = 'http://127.0.0.1:11434/api/generate'
LLAMA_MODEL = 'llama3.1:8b'
LLAMA_RESULTS_CSV_PATH = 'llama_svamp_results.csv'
CHUNK_SAVE_SIZE = 20  # Save every 20 data points


# --- Prompt Templates ---


PROMPT_LLAMA_COT = """
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


PROMPT_LLAMA_CORRUPT_COT = """
Given the math question: "{question}", provide a step-by-step reasoning process (Chain of Thought) to solve it. The steps must:

- Include deliberate mathematical errors (wrong operations, misplaced signs, incorrect assumptions, etc.)
- Present steps in an incorrect or shuffled order (not logically flowing)
- End with a final answer that is always incorrect
- Do not mention that the reasoning is wrong
- Do not correct any step
- Output only the steps and final answer (no intro, explanation, or conclusion)

Final Answer format: ####<number>
"""


PROMPT_LLAMA_WITH_PARTIAL_COT = """
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


def download_svamp_dataset() -> list:
    """Download SVAMP dataset from GitHub"""
    print("Downloading SVAMP dataset...")
    SVAMP_URL = "https://raw.githubusercontent.com/arkilpatel/SVAMP/main/SVAMP.json"
    try:
        response = requests.get(SVAMP_URL)
        response.raise_for_status()
        data = json.loads(response.text)
        
        # Convert SVAMP format to consistent format
        questions = []
        for item in data:
            # SVAMP has 'Body', 'Question', and 'Answer' fields
            question_text = f"{item['Body']} {item['Question']}"
            answer = str(item['Answer'])  # Convert to string for consistency
            
            questions.append({
                'question': question_text,
                'answer': answer
            })
        
        print(f"Successfully downloaded {len(questions)} SVAMP questions.")
        return questions
    except requests.exceptions.RequestException as e:
        print(f"Error downloading SVAMP dataset: {e}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error parsing SVAMP JSON: {e}")
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
    """Collect all LLaMA responses and save to CSV"""
    print("=== LLaMA Data Collection Script (SVAMP Dataset) ===")
    
    # Check if file already exists
    if os.path.exists(LLAMA_RESULTS_CSV_PATH):
        response = input(f"File '{LLAMA_RESULTS_CSV_PATH}' already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Exiting without overwriting.")
            return
        os.remove(LLAMA_RESULTS_CSV_PATH)
    
    questions = download_svamp_dataset()
    if not questions: 
        return

    results = []
    
    print(f"\nStarting LLaMA data collection with {len(questions)} SVAMP questions...")

    for idx, q in enumerate(tqdm(questions, desc="Processing LLaMA Questions")):
        question_text = q['question']
        original_answer_raw = q.get('answer', '')
        
        print(f"\nProcessing question {idx + 1}/{len(questions)}")
        print(f"Question: {question_text[:100]}...")
        
        # LLaMA generates CoT, direct answer, and corrupted CoT
        llama_cot = generate_llm_response(LLAMA_MODEL, PROMPT_LLAMA_COT.format(question=question_text))
        llama_direct_raw = generate_llm_response(LLAMA_MODEL, PROMPT_DIRECT.format(question=question_text))
        corrupted_cot = generate_llm_response(LLAMA_MODEL, PROMPT_LLAMA_CORRUPT_COT.format(question=question_text))

        # Process CoTs to remove final answers
        llama_cot_steps_only = remove_final_answer(llama_cot)
        corrupted_cot_steps_only = remove_final_answer(corrupted_cot)
        partial_cot_steps = get_partial_cot(llama_cot, fraction=0.5)

        # LLaMA with partial CoT
        llama_partial_cot_response = generate_llm_response(LLAMA_MODEL, PROMPT_LLAMA_WITH_PARTIAL_COT.format(question=question_text, partial_cot=partial_cot_steps))

        results.append({
            'question': question_text,
            'original_answer_raw': original_answer_raw,
            'llama_direct_raw': llama_direct_raw,
            'llama_cot_raw': llama_cot,
            'llama_corrupted_cot_raw': corrupted_cot,
            'llama_cot_steps_only': llama_cot_steps_only,
            'corrupted_cot_steps_only': corrupted_cot_steps_only,
            'partial_cot_steps': partial_cot_steps,
            'llama_partial_cot_response': llama_partial_cot_response,
        })

        # Save results to CSV every CHUNK_SAVE_SIZE entries
        if (idx + 1) % CHUNK_SAVE_SIZE == 0 or (idx + 1) == len(questions):
            df_chunk = pd.DataFrame(results)
            
            # Append to the CSV if file exists, else create new
            if os.path.exists(LLAMA_RESULTS_CSV_PATH):
                df_chunk.to_csv(LLAMA_RESULTS_CSV_PATH, mode='a', header=False, index=False)
                print(f"Appended chunk {((idx + 1) // CHUNK_SAVE_SIZE)} to existing LLaMA CSV")
            else:
                df_chunk.to_csv(LLAMA_RESULTS_CSV_PATH, index=False)
                print(f"Created new LLaMA CSV with first chunk")
            
            results = []  # Reset list after saving

        time.sleep(0.5)

    print(f"\nLLaMA data collection completed!")
    print(f"Results saved to: '{LLAMA_RESULTS_CSV_PATH}'")
    print(f"Total questions processed: {len(questions)}")


if __name__ == "__main__":
    main()
