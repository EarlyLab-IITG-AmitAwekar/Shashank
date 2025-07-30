import os
import time
import requests
import pandas as pd
import sympy as sp
from latex2sympy2 import latex2sympy
from tqdm import tqdm
import regex


# --- Configuration ---
API_URL = 'http://127.0.0.1:11434/api/generate'
GEMMA3_MODEL = 'gemma3:27b'
LLAMA_RESULTS_CSV_PATH = '../llama_math_dataset_results.csv'
FINAL_RESULTS_CSV_PATH = 'llama_gemma3_math_final_results.csv'
SUMMARY_CSV_PATH = 'llama_gemma3_math_accuracy_summary.csv'
CHUNK_SAVE_SIZE = 20


# --- Prompt Templates ---
PROMPT_DIRECT = """You are an expert mathematician solving competition-level problems.
Your task is to solve the given math problem step by step.


Provide your final answer in the exact format: \\boxed{{answer}}


Question: {question}


Solution:"""


PROMPT_GEMMA3_WITH_COT = """You must ONLY use the provided reasoning below to answer the question. 
Do not add your own reasoning. Continue from where the provided reasoning stops 
and complete the solution.


Provide the final answer in the format: \\boxed{{answer}}


Question: {question}


Reasoning:
{cot}


Continue the solution:"""


def generate_llm_response(model: str, prompt: str) -> str:
    """Generate response from LLM with error handling"""
    if not prompt or not isinstance(prompt, str):
        return "Error: Invalid prompt"
    
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        resp = requests.post(API_URL, json=payload, timeout=120)
        resp.raise_for_status()
        result = resp.json()
        return result.get('response', '').strip()
    except requests.exceptions.RequestException as e:
        return f"Error: API request failed: {e}"
    except Exception as e:
        return f"Error: {e}"


def safe_get_value(row, column, default=""):
    """Safely get value from DataFrame row"""
    try:
        value = row[column]
        if pd.isna(value):
            return default
        return str(value)
    except KeyError:
        print(f"Warning: Column '{column}' not found")
        return default


def find_boxed_content(text: str):
    if not isinstance(text, str) or not text:
        return []
    try:
        pattern = regex.compile(
            r'\\boxed\s*(?P<brace>\{(?:[^{}]|(?&brace))*\})',
            regex.VERBOSE
        )
        matches = pattern.findall(text)
        return [m[1:-1].strip() for m in matches]
    except:
        return []


def split_comma_separated(text: str):
    if not text:
        return []
    parts, buf, depth = [], "", 0
    for ch in text:
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(buf.strip())
            buf = ""
        else:
            buf += ch
    if buf.strip():
        parts.append(buf.strip())
    return parts


def latex_to_sympy_safe(latex_expr):
    try:
        return latex2sympy(latex_expr)
    except:
        return latex_expr.strip() if latex_expr else ""


def extract_math_answer(text):
    if not text or pd.isna(text):
        return None
    text = str(text)
    boxed = find_boxed_content(text)
    if not boxed:
        return None
    last = boxed[-1]
    segs = split_comma_separated(last)
    if len(segs) == 1:
        return latex_to_sympy_safe(segs[0])
    else:
        syms = [latex_to_sympy_safe(s) for s in segs]
        return sp.FiniteSet(*syms)


def math_answers_equal(a, b):
    if a is None or b is None:
        return a == b
    if isinstance(a, str):
        a = latex_to_sympy_safe(a)
    if isinstance(b, str):
        b = latex_to_sympy_safe(b)
    nums = (sp.Integer, sp.Rational, sp.Float, int, float)
    if isinstance(a, nums) and isinstance(b, nums):
        try:
            return abs(float(a) - float(b)) < 1e-10
        except:
            pass
    if isinstance(a, sp.FiniteSet) and isinstance(b, sp.FiniteSet):
        return a == b
    try:
        diff = sp.simplify(a - b)
        return diff == 0
    except:
        try:
            return a.equals(b)
        except:
            return str(a) == str(b)


def main():
    print("=== Gemma3 Processing and Evaluation Script (MATH Dataset) ===")


    # Check CSV existence
    if not os.path.exists(LLAMA_RESULTS_CSV_PATH):
        print(f"Error: LLaMA results file '{LLAMA_RESULTS_CSV_PATH}' not found.")
        return


    # Load CSV with error handling
    try:
        df_llama = pd.read_csv(LLAMA_RESULTS_CSV_PATH)
        print(f"Loaded {len(df_llama)} rows")
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return


    # Validate required columns
    required = [
        'question', 'original_answer_raw', 'llama_direct_raw', 'llama_cot_raw',
        'llama_corrupted_cot_raw', 'llama_cot_steps_only', 'corrupted_cot_steps_only',
        'partial_cot_steps', 'llama_partial_cot_response'
    ]
    missing = [c for c in required if c not in df_llama.columns]
    if missing:
        print(f"Error: Missing columns: {missing}")
        print(f"Available columns: {df_llama.columns.tolist()}")
        return


    # Check for overwrite
    if os.path.exists(FINAL_RESULTS_CSV_PATH):
        os.remove(FINAL_RESULTS_CSV_PATH)


    combined = []


    for idx, row in tqdm(df_llama.iterrows(), total=len(df_llama), desc="Processing"):
        try:
            # Safely extract values
            q = safe_get_value(row, 'question')
            cot = safe_get_value(row, 'llama_cot_steps_only')
            corrupted = safe_get_value(row, 'corrupted_cot_steps_only')
            partial = safe_get_value(row, 'partial_cot_steps')


            # Skip if question is empty
            if not q.strip():
                print(f"Skipping row {idx}: empty question")
                continue


            # Generate Gemma3 responses with error handling
            gemma3_direct = generate_llm_response(GEMMA3_MODEL, PROMPT_DIRECT.format(question=q))
            gemma3_with_cot = generate_llm_response(GEMMA3_MODEL, PROMPT_GEMMA3_WITH_COT.format(question=q, cot=cot))
            gemma3_with_corr = generate_llm_response(GEMMA3_MODEL, PROMPT_GEMMA3_WITH_COT.format(question=q, cot=corrupted))
            gemma3_with_part = generate_llm_response(GEMMA3_MODEL, PROMPT_GEMMA3_WITH_COT.format(question=q, cot=partial))


            combined.append({
                'question': q,
                'original_answer_raw': safe_get_value(row, 'original_answer_raw'),
                'llama_direct_raw': safe_get_value(row, 'llama_direct_raw'),
                'llama_cot_raw': safe_get_value(row, 'llama_cot_raw'),
                'llama_corrupted_cot_raw': safe_get_value(row, 'llama_corrupted_cot_raw'),
                'llama_cot_steps_only': cot,
                'corrupted_cot_steps_only': corrupted,
                'partial_cot_steps': partial,
                'llama_partial_cot_response': safe_get_value(row, 'llama_partial_cot_response'),
                'gemma3_direct_raw': gemma3_direct,
                'gemma3_with_cot_raw': gemma3_with_cot,
                'gemma3_with_corrupted_cot_raw': gemma3_with_corr,
                'gemma3_with_partial_cot_raw': gemma3_with_part
            })


            # Save in chunks
            if (idx + 1) % CHUNK_SAVE_SIZE == 0 or idx + 1 == len(df_llama):
                df_chunk = pd.DataFrame(combined)
                mode = 'a' if os.path.exists(FINAL_RESULTS_CSV_PATH) else 'w'
                header = not os.path.exists(FINAL_RESULTS_CSV_PATH)
                df_chunk.to_csv(FINAL_RESULTS_CSV_PATH, mode=mode, header=header, index=False)
                combined = []


            time.sleep(0.5)


        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            print(f"Row data: {dict(row)}")
            continue


    # Load and evaluate results
    try:
        df_combined = pd.read_csv(FINAL_RESULTS_CSV_PATH)
        
        # Extract answers
        answer_columns = {
            'original_answer_raw': 'original_answer_ans',
            'llama_direct_raw': 'llama_direct_ans',
            'llama_cot_raw': 'llama_cot_ans',
            'llama_corrupted_cot_raw': 'llama_corrupted_cot_ans',
            'llama_partial_cot_response': 'llama_partial_cot_response_ans',
            'gemma3_direct_raw': 'gemma3_direct_ans',
            'gemma3_with_cot_raw': 'gemma3_with_cot_ans',
            'gemma3_with_corrupted_cot_raw': 'gemma3_with_corrupted_cot_ans',
            'gemma3_with_partial_cot_raw': 'gemma3_with_partial_cot_ans'
        }
        
        for raw_col, ans_col in answer_columns.items():
            if raw_col in df_combined.columns:
                df_combined[ans_col] = df_combined[raw_col].apply(extract_math_answer)


        # Calculate accuracies
        comparisons = {
            'llama_vs_original': 'llama_direct_ans',
            'llama_cot_vs_original': 'llama_cot_ans',
            'llama_corruptedCOT_vs_original': 'llama_corrupted_cot_ans',
            'llama_partial_cot_vs_original': 'llama_partial_cot_response_ans',
            'gemma3_direct_vs_original': 'gemma3_direct_ans',
            'gemma3_w_cot_vs_original': 'gemma3_with_cot_ans',
            'gemma3_w_corrupt_cot_vs_original': 'gemma3_with_corrupted_cot_ans',
            'gemma3_w_partial_cot_vs_original': 'gemma3_with_partial_cot_ans'
        }


        for flag, ans_col in comparisons.items():
            if ans_col in df_combined.columns:
                df_combined[flag] = df_combined.apply(
                    lambda r: math_answers_equal(r.get(ans_col), r.get('original_answer_ans')), axis=1
                )


        # Display results
        results = {
            "LLaMA Direct vs. Original": df_combined.get('llama_vs_original', pd.Series()).mean(),
            "LLaMA CoT vs. Original": df_combined.get('llama_cot_vs_original', pd.Series()).mean(),
            "LLaMA Corrupted CoT vs. Original": df_combined.get('llama_corruptedCOT_vs_original', pd.Series()).mean(),
            "LLaMA Partial CoT vs. Original": df_combined.get('llama_partial_cot_vs_original', pd.Series()).mean(),
            "Gemma3 Direct vs. Original": df_combined.get('gemma3_direct_vs_original', pd.Series()).mean(),
            "Gemma3 w/ CoT vs. Original": df_combined.get('gemma3_w_cot_vs_original', pd.Series()).mean(),
            "Gemma3 w/ Corrupted CoT vs. Original": df_combined.get('gemma3_w_corrupt_cot_vs_original', pd.Series()).mean(),
            "Gemma3 w/ Partial CoT vs. Original": df_combined.get('gemma3_w_partial_cot_vs_original', pd.Series()).mean()
        }


        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        for name, acc in results.items():
            if not pd.isna(acc):
                print(f"{name}: {acc:.2%}")


        # Save results
        df_combined.to_csv(FINAL_RESULTS_CSV_PATH, index=False)
        pd.DataFrame([
            {'setting': k, 'accuracy': v} for k, v in results.items() if not pd.isna(v)
        ]).to_csv(SUMMARY_CSV_PATH, index=False)


        print(f"\nSaved results to {FINAL_RESULTS_CSV_PATH}")
        print(f"Saved summary to {SUMMARY_CSV_PATH}")


    except Exception as e:
        print(f"Error in evaluation: {e}")


if __name__ == '__main__':
    main()
