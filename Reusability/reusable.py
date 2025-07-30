import requests
import json
import re
import pandas as pd
from tqdm import tqdm
import time

# --- Configuration ---
API_URL = 'http://127.0.0.1:11434/api/generate'
GEMMA_MODEL = 'gemma3:1b'
LLAMA_MODEL = 'llama3'
MAX_QUESTIONS = 2
RESULTS_CSV_PATH = 'gemma_llama_gsm8k_results_v6.csv'

# --- Prompt Templates ---

PROMPT_GEMMA_COT = """
You are a precise math assistant. Solve the following problem by thinking step-by-step. First, break down your reasoning and calculations. Finally, state the final answer in the specific format 'The final answer is ####<number>'.

Here is an example:
---
Question: Natalia sold 48 cupcakes on Monday. On Tuesday, she sold half as many as on Monday. On Wednesday, she sold 10 more than on Tuesday. How many cupcakes did she sell in total?
Reasoning:
1.  First, find the number of cupcakes sold on Tuesday. This is half of Monday's sales, so: 48 / 2 = 24 cupcakes.
2.  Next, find the number of cupcakes sold on Wednesday. This is 10 more than Tuesday's sales, so: 24 + 10 = 34 cupcakes.
3.  Finally, find the total number of cupcakes sold across all three days by adding the amounts from each day: 48 (Monday) + 24 (Tuesday) + 34 (Wednesday) = 106 cupcakes.
The final answer is ####106
---
Now, solve this problem:
Question: {question}
"""

# New, stricter prompt for direct answers, used by BOTH models.
PROMPT_DIRECT = """
You are a master of mathematics with exceptional problem-solving skills. For each question, think carefully through the problem step-by-step in your mind before computing the solution. Use your deep understanding of math to arrive at the correct result with precision.

Do not output any explanations, steps, symbols, or intermediate reasoning. Only return the final correct numerical answer in the following format:

###<number>

Question: {question}

Final Answer:"""


# New, more reliable prompt to generate a flawed CoT.
PROMPT_GEMMA_CORRUPT_COT = """
Given the math question: "{question}", provide a step-by-step reasoning process (Chain of Thought) to solve it. The steps must:

- Include deliberate mathematical errors (wrong operations, misplaced signs, incorrect assumptions, etc.)
- Present steps in an incorrect or shuffled order (not logically flowing)
- End with a final answer that is always incorrect
- Do not mention that the reasoning is wrong
- Do not correct any step
- Output only the steps and final answer (no intro, explanation, or conclusion)

Final Answer: ####<number>

"""

PROMPT_LLAMA_WITH_COT = """
You can't solve the maths problem by yourself and you have to ONLY use the following reasoning to solve the question.
Provide only the final numerical answer in format: ####<number>. Do not provide any other text, reasoning, or symbols.

Question: {question}

Reasoning (Chain of Thought):
{cot}
"""

# --- Core Functions (No changes needed) ---
def download_gsm8k_full() -> list:
    print("Downloading GSM8K dataset...")
    GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
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

def extract_llm_answer(text: str):
    if text is None or "Error:" in text: return None
    # This regex is now more important to strip away any residual text.
    numbers = re.findall(r'[-+]?\d*\.?\d+', text)
    if not numbers: return None
    try: return float(numbers[-1])
    except ValueError: return None

# --- Main Logic (Updated for new corruption strategy) ---
def main():
    questions = download_gsm8k_full()
    if not questions: return

    questions_to_process = questions[:MAX_QUESTIONS]
    results = []

    print(f"\nStarting experiment with {len(questions_to_process)} questions...")
    for q in tqdm(questions_to_process, desc="Processing Questions"):
        question_text = q['question']
        print(f"Question: {question_text}")

        # Step 1: Gemma generates CoT and a direct answer
        gemma_cot = generate_llm_response(GEMMA_MODEL, PROMPT_GEMMA_COT.format(question=question_text))
        print(f"Gemma CoT: {gemma_cot}")
        gemma_direct_raw = generate_llm_response(GEMMA_MODEL, PROMPT_DIRECT.format(question=question_text))
        

        # Step 2: Create a plausible WRONG answer to feed into the corruption prompt
        # We need the numerical answer from gemma_direct_raw to create the wrong one
        gemma_direct_answer = extract_llm_answer(gemma_direct_raw)
        print(f"Gemma Direct: {gemma_direct_answer}")
        # Step 3: Gemma creates an intentionally flawed CoT to justify the wrong answer
        gemma_corrupted_cot = generate_llm_response(
        GEMMA_MODEL,
        PROMPT_GEMMA_CORRUPT_COT.format(question=question_text)
        )
        print(f"Gemma Corrupted CoT: {gemma_corrupted_cot}")

        # Step 4: Llama generates its answers
        llama_direct_raw = generate_llm_response(LLAMA_MODEL, PROMPT_DIRECT.format(question=question_text))
        print(f"Llama Direct: {llama_direct_raw}")
        llama_with_cot = generate_llm_response(LLAMA_MODEL, PROMPT_LLAMA_WITH_COT.format(question=question_text, cot=gemma_cot))
        print(f"Llama with CoT: {llama_with_cot}")
        llama_with_corrupted_cot = generate_llm_response(LLAMA_MODEL, PROMPT_LLAMA_WITH_COT.format(question=question_text, cot=gemma_corrupted_cot))
        print(f"Llama with Corrupted CoT: {llama_with_corrupted_cot}")

        results.append({
            'question': question_text, 'gemma_direct_raw': gemma_direct_raw,
            'gemma_cot_raw': gemma_cot, 'gemma_corrupted_cot_raw': gemma_corrupted_cot,
            'llama_direct_raw': llama_direct_raw, 'llama_with_cot_raw': llama_with_cot,
            'llama_with_corrupted_cot_raw': llama_with_corrupted_cot,
        })
        time.sleep(0.5) # A small sleep to not overload the server

    # The rest of the evaluation logic remains the same
    df = pd.DataFrame(results)
    print("\nExtracting final answers...")
    df['gemma_direct_answer'] = df['gemma_direct_raw'].apply(extract_llm_answer)
    df['llama_direct_answer'] = df['llama_direct_raw'].apply(extract_llm_answer)
    df['llama_with_cot_answer'] = df['llama_with_cot_raw'].apply(extract_llm_answer)
    df['llama_with_corrupted_cot_answer'] = df['llama_with_corrupted_cot_raw'].apply(extract_llm_answer)

    print("Calculating accuracies...")
    df['llama_direct_vs_gemma_correct'] = df['llama_direct_answer'] == df['gemma_direct_answer']
    df['llama_w_cot_vs_gemma_correct'] = df['llama_with_cot_answer'] == df['gemma_direct_answer']
    df['llama_w_corrupt_cot_vs_gemma_correct'] = df['llama_with_corrupted_cot_answer'] == df['gemma_direct_answer']

    llama_accuracies = {
        "Llama3 Direct vs. Gemma Direct": df['llama_direct_vs_gemma_correct'].mean(),
        "Llama3 with Gemma's Correct CoT vs. Gemma Direct": df['llama_w_cot_vs_gemma_correct'].mean(),
        "Llama3 with Gemma's Flawed CoT vs. Gemma Direct": df['llama_w_corrupt_cot_vs_gemma_correct'].mean(),
    }

    print("\n--- Experiment Finished ---")
    print("\nLlama's Agreement with Gemma's Direct Answer")
    for name, acc in llama_accuracies.items():
        print(f"  - {name}: {acc:.2%}")

    df.to_csv(RESULTS_CSV_PATH, index=False)
    print(f"\nSaved all detailed results to '{RESULTS_CSV_PATH}'")

if __name__ == "__main__":
    main()