import requests
import json
import re
import pandas as pd
from tqdm import tqdm
import time
import os
from datasets import load_dataset
import concurrent.futures
import traceback

# --- Configuration ---
API_URL = 'http://127.0.0.1:11434/api/generate'
LLAMA_MODEL = 'llama3.1:8b'
LLAMA_RESULTS_CSV_PATH = 'llama_math_dataset_results.csv'
CHUNK_SAVE_SIZE = 50
MAX_WORKERS = 4
REQUEST_TIMEOUT = 120
BATCH_SIZE = 4

# --- FIXED Prompt Templates (removed problematic placeholders) ---

PROMPT_LLAMA_COT = """You are a precise and logical math assistant. Solve the following problem by thinking step-by-step.
Your reasoning should clearly break down the problem into individual steps and calculations.
After presenting your complete reasoning, state the final answer in LaTeX boxed format: \\boxed{{answer}}

Important formatting guidelines:
- For matrices, use \\begin{{pmatrix}}...\\end{{pmatrix}} or \\begin{{bmatrix}}...\\end{{bmatrix}}
- For fractions, use \\frac{{numerator}}{{denominator}}
- For sets, use \\{{element1, element2, ...\\}}
- For complex expressions, maintain proper LaTeX formatting
- For simple numbers, just use the number: \\boxed{{42}}

Here is an example of the desired format:
---
Question: Find the value of x if 2x + 5 = 13.
Reasoning:
1. Start with the equation: 2x + 5 = 13
2. Subtract 5 from both sides: 2x = 13 - 5 = 8
3. Divide both sides by 2: x = 8/2 = 4
\\boxed{{4}}
---

Now, solve this problem:
Question: {question}
Reasoning:
"""

PROMPT_DIRECT = """You are an expert mathematician solving competition-level problems.
Solve the problem and provide your final answer in LaTeX boxed format: \\boxed{{answer}}

Important formatting guidelines:
- For matrices, use \\begin{{pmatrix}}...\\end{{pmatrix}} or \\begin{{bmatrix}}...\\end{{bmatrix}}
- For fractions, use \\frac{{numerator}}{{denominator}}
- For sets, use \\{{element1, element2, ...\\}}
- For complex expressions, use proper LaTeX formatting
- For simple numbers, just use the number: \\boxed{{42}}

Question: {question}

Solution:"""

PROMPT_LLAMA_CORRUPT_COT = """Given the math question: "{question}", provide a step-by-step reasoning process (Chain of Thought) to solve it. The steps must:

- Include deliberate mathematical errors (wrong operations, misplaced signs, incorrect assumptions, etc.)
- Present steps in an incorrect or shuffled order (not logically flowing)
- End with a final answer that is always incorrect
- Do not mention that the reasoning is wrong
- Do not correct any step
- Output only the steps and final answer (no intro, explanation, or conclusion)

Final Answer format: \\boxed{{incorrect_answer}}
"""

PROMPT_LLAMA_WITH_PARTIAL_COT = """You are a precise and logical math assistant. Below is a partially completed reasoning for a math problem. Continue from where the reasoning stops and complete the solution to provide the final answer.

Question: {question}

Partial Reasoning:
{partial_cot}

Continue the reasoning and provide the final answer in LaTeX boxed format: \\boxed{{answer}}

Remember to use proper LaTeX formatting:
- For matrices: \\begin{{pmatrix}}...\\end{{pmatrix}}
- For fractions: \\frac{{numerator}}{{denominator}}
- For sets: \\{{element1, element2, ...\\}}
"""

# --- Helper Functions ---

def remove_final_answer(cot_text):
    """Remove the final answer line from CoT text, keeping only the reasoning steps."""
    if cot_text is None:
        return ""
    
    lines = cot_text.strip().split('\n')
    filtered_lines = []
    
    for line in lines:
        # Stop at any line that contains boxed answers
        if re.search(r'\\boxed\{', line, re.IGNORECASE):
            break
        # Also check for old format just in case
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

def download_math_dataset() -> list:
    """Download MATH dataset with proper error handling"""
    print("Downloading MATH dataset from alternative repository...")
    try:
        dataset = load_dataset("nlile/hendrycks-MATH-benchmark")
        train_data = dataset['train']
        
        print(f"Total train samples available: {len(train_data)}")
        
        # Debug: Check the actual structure
        if len(train_data) > 0:
            sample = train_data[0]
            available_fields = list(sample.keys())
            print(f"Available fields in dataset: {available_fields}")

        data = []
        max_samples = min(1300, len(train_data))
        
        for i in range(max_samples):
            item = train_data[i]
            
            # Handle different possible field names
            question = item.get('problem', item.get('question', ''))
            answer = item.get('solution', item.get('answer', ''))
            level = item.get('level', 'Unknown')
            problem_type = item.get('type', item.get('subject', 'Unknown'))
            
            if question and answer:  # Only add if both exist
                data.append({
                    'question': question,
                    'answer': answer,
                    'level': level,
                    'type': problem_type
                })
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{max_samples} samples...")
        
        print(f"Successfully downloaded {len(data)} problems from MATH dataset (train split).")
        return data
        
    except Exception as e:
        print(f"Error downloading MATH dataset: {e}")
        traceback.print_exc()
        return []

def generate_llm_response_sync(llm_name: str, prompt: str) -> str:
    """Synchronous LLM response generation with better error handling"""
    payload = {"model": llm_name, "prompt": prompt, "stream": False}
    try:
        response = requests.post(API_URL, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        result = response.json()
        return result.get('response', '').strip()
    except requests.exceptions.RequestException as e:
        return f"Error: API request failed: {str(e)}"

def process_single_question(question_data):
    """Process a single question - FINAL FIXED VERSION"""
    idx, q = question_data
    
    try:
        # Validate input data structure
        if not isinstance(q, dict):
            print(f"Error Q{idx}: Invalid question data structure: {type(q)}")
            return None
            
        question_text = q.get('question', '')
        original_answer_raw = q.get('answer', '')
        problem_level = q.get('level', 'Unknown')
        problem_type = q.get('type', 'Unknown')
        
        if not question_text or not original_answer_raw:
            print(f"Error Q{idx}: Missing question or answer data")
            return None
        
        print(f"Processing Q{idx}: {question_text[:50]}...")
        
        # Generate all LLM responses with FIXED formatting
        try:
            llama_cot = generate_llm_response_sync(LLAMA_MODEL, PROMPT_LLAMA_COT.format(question=question_text))
            
            if "Error:" in llama_cot:
                print(f"API Error for Q{idx} CoT: {llama_cot}")
                return None
                
            llama_direct_raw = generate_llm_response_sync(LLAMA_MODEL, PROMPT_DIRECT.format(question=question_text))
            corrupted_cot = generate_llm_response_sync(LLAMA_MODEL, PROMPT_LLAMA_CORRUPT_COT.format(question=question_text))
            
            # Process CoTs
            llama_cot_steps_only = remove_final_answer(llama_cot)
            corrupted_cot_steps_only = remove_final_answer(corrupted_cot)
            partial_cot_steps = get_partial_cot(llama_cot, fraction=0.5)
            
            # Generate partial CoT response
            llama_partial_cot_response = generate_llm_response_sync(
                LLAMA_MODEL, 
                PROMPT_LLAMA_WITH_PARTIAL_COT.format(question=question_text, partial_cot=partial_cot_steps)
            )
            
        except Exception as format_error:
            print(f"Error formatting prompts for Q{idx}: {format_error}")
            return None
        
        return {
            'idx': idx,
            'question': question_text,
            'original_answer_raw': original_answer_raw,
            'problem_level': problem_level,
            'problem_type': problem_type,
            'llama_direct_raw': llama_direct_raw,
            'llama_cot_raw': llama_cot,
            'llama_corrupted_cot_raw': corrupted_cot,
            'llama_cot_steps_only': llama_cot_steps_only,
            'corrupted_cot_steps_only': corrupted_cot_steps_only,
            'partial_cot_steps': partial_cot_steps,
            'llama_partial_cot_response': llama_partial_cot_response,
        }
        
    except Exception as e:
        print(f"Error processing question {idx}: {e}")
        traceback.print_exc()
        return None

def validate_response_format(response, question_type=""):
    """Validate that the response contains properly formatted boxed answer."""
    if not response or "Error:" in response:
        return False
    
    # Check if response contains \boxed{} format
    boxed_pattern = r'\\boxed\{[^}]+\}'
    has_boxed = bool(re.search(boxed_pattern, response))
    
    return has_boxed

def save_results_chunk(results, chunk_num):
    """Save results chunk to CSV efficiently"""
    if not results:
        print(f"Warning: No results to save for chunk {chunk_num}")
        return
        
    df_chunk = pd.DataFrame(results)
    
    # Sort by original index to maintain order
    df_chunk = df_chunk.sort_values('idx').drop('idx', axis=1)
    
    # Append to the CSV if file exists, else create new
    if os.path.exists(LLAMA_RESULTS_CSV_PATH):
        df_chunk.to_csv(LLAMA_RESULTS_CSV_PATH, mode='a', header=False, index=False)
        print(f"✅ Saved chunk {chunk_num} with {len(df_chunk)} questions")
    else:
        df_chunk.to_csv(LLAMA_RESULTS_CSV_PATH, index=False)
        print(f"✅ Created new CSV with chunk {chunk_num} ({len(df_chunk)} questions)")

def main():
    """High-Speed LLaMA Data Collection - FINAL WORKING VERSION"""
    print("=== 🚀 FINAL WORKING LLaMA Data Collection Script for MATH Dataset ===")
    print(f"Max concurrent workers: {MAX_WORKERS}")
    print(f"Batch size: {BATCH_SIZE}")
    
    # Check if file already exists
    if os.path.exists(LLAMA_RESULTS_CSV_PATH):
        response = input(f"File '{LLAMA_RESULTS_CSV_PATH}' already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Exiting without overwriting.")
            return
        os.remove(LLAMA_RESULTS_CSV_PATH)
    
    questions = download_math_dataset()
    if not questions: 
        print("Failed to download dataset. Exiting.")
        return

    # For testing, uncomment next line:
    # questions = questions[:10]

    print(f"\n🏃‍♂️ Starting data collection with {len(questions)} questions...")
    
    # Prepare question data with indices
    question_data = [(idx, q) for idx, q in enumerate(questions)]
    
    all_results = []
    format_validation_stats = {
        'cot_valid': 0,
        'direct_valid': 0,
        'corrupt_valid': 0,
        'partial_valid': 0,
        'total_processed': 0,
        'total_failed': 0
    }
    
    start_time = time.time()
    
    # Process questions in batches using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Create batches
        batches = [question_data[i:i + BATCH_SIZE] for i in range(0, len(question_data), BATCH_SIZE)]
        
        for batch_idx, batch in enumerate(tqdm(batches, desc="Processing Batches")):
            # Submit all questions in current batch
            future_to_question = {
                executor.submit(process_single_question, q_data): q_data 
                for q_data in batch
            }
            
            batch_results = []
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_question, timeout=REQUEST_TIMEOUT * 3):
                try:
                    result = future.result(timeout=60)
                    if result is not None:
                        batch_results.append(result)
                        
                        # Validate formats and collect stats
                        if validate_response_format(result['llama_cot_raw'], "CoT"):
                            format_validation_stats['cot_valid'] += 1
                        if validate_response_format(result['llama_direct_raw'], "Direct"):
                            format_validation_stats['direct_valid'] += 1
                        if validate_response_format(result['llama_corrupted_cot_raw'], "Corrupt CoT"):
                            format_validation_stats['corrupt_valid'] += 1
                        if validate_response_format(result['llama_partial_cot_response'], "Partial CoT"):
                            format_validation_stats['partial_valid'] += 1
                        format_validation_stats['total_processed'] += 1
                    else:
                        format_validation_stats['total_failed'] += 1
                        
                except Exception as e:
                    print(f"❌ Error collecting result: {e}")
                    format_validation_stats['total_failed'] += 1
                    continue
            
            all_results.extend(batch_results)
            
            # Save results periodically
            chunk_threshold = max(1, CHUNK_SAVE_SIZE // BATCH_SIZE)
            if (batch_idx + 1) % chunk_threshold == 0 or batch_idx == len(batches) - 1:
                save_results_chunk(all_results, batch_idx + 1)
                all_results = []  # Clear memory
            
            # Progress update
            processed = min((batch_idx + 1) * BATCH_SIZE, len(questions))
            elapsed = time.time() - start_time
            rate = processed / elapsed if elapsed > 0 else 0
            eta = (len(questions) - processed) / rate if rate > 0 else 0
            
            success_count = format_validation_stats['total_processed']
            failed_count = format_validation_stats['total_failed']
            
            print(f"⚡ Processed: {processed}/{len(questions)} | Rate: {rate:.1f} q/s | ETA: {eta/60:.1f}m | Success: {success_count} | Failed: {failed_count}")
    
    total_time = time.time() - start_time
    
    print(f"\n🎉 LLaMA data collection completed!")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    if total_time > 0:
        print(f"⚡ Average rate: {len(questions)/total_time:.1f} questions/second")
    print(f"💾 Results saved to: '{LLAMA_RESULTS_CSV_PATH}'")
    print(f"📊 Total questions processed: {len(questions)}")
    print(f"✅ Successfully processed: {format_validation_stats['total_processed']}")
    print(f"❌ Failed: {format_validation_stats['total_failed']}")
    
    # Print format validation statistics
    if format_validation_stats['total_processed'] > 0:
        print(f"\n📈 Format Validation Statistics:")
        total = format_validation_stats['total_processed']
        print(f"- CoT responses with \\boxed{{}}: {format_validation_stats['cot_valid']}/{total} ({format_validation_stats['cot_valid']/total*100:.1f}%)")
        print(f"- Direct responses with \\boxed{{}}: {format_validation_stats['direct_valid']}/{total} ({format_validation_stats['direct_valid']/total*100:.1f}%)")
        print(f"- Corrupt CoT responses with \\boxed{{}}: {format_validation_stats['corrupt_valid']}/{total} ({format_validation_stats['corrupt_valid']/total*100:.1f}%)")
        print(f"- Partial CoT responses with \\boxed{{}}: {format_validation_stats['partial_valid']}/{total} ({format_validation_stats['partial_valid']/total*100:.1f}%)")
    
    # Print dataset statistics
    try:
        if os.path.exists(LLAMA_RESULTS_CSV_PATH):
            df_final = pd.read_csv(LLAMA_RESULTS_CSV_PATH)
            print(f"\n📊 Final Dataset Statistics:")
            print(f"- Total problems saved: {len(df_final)}")
            if 'problem_level' in df_final.columns:
                print(f"- Problems by level: {df_final['problem_level'].value_counts().to_dict()}")
            if 'problem_type' in df_final.columns:
                print(f"- Problems by type: {df_final['problem_type'].value_counts().to_dict()}")
        else:
            print("⚠️  No CSV file was created - all processing failed")
    except Exception as e:
        print(f"⚠️  Could not load final statistics: {e}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️  Process interrupted by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        traceback.print_exc()
