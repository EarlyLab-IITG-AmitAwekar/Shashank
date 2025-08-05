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
GEMMA_MODEL = 'gemma3:27b'  # Changed from 'llama3.1:8b'
GEMMA_RESULTS_CSV_PATH = 'gemma3_strategyqa_dataset_results.csv'  # Updated file name
CHUNK_SAVE_SIZE = 50
MAX_WORKERS = 4
REQUEST_TIMEOUT = 120
BATCH_SIZE = 4



# --- Prompt Templates for StrategyQA ---



PROMPT_GEMMA_COT = """You are a logical reasoning assistant. Answer the following question by thinking step-by-step.
Your reasoning should clearly break down the problem into individual steps and considerations.
After presenting your complete reasoning, state the final answer as either "Yes" or "No".


Important guidelines:
- Consider all relevant facts and context
- Think through the logical implications
- Provide clear step-by-step reasoning
- End with a definitive "Yes" or "No" answer


Here is an example of the desired format:
---
Question: Could a penguin survive in the Sahara Desert?
Reasoning:
1. Penguins are adapted to cold climates, primarily living in Antarctica and other cold regions
2. The Sahara Desert has extremely hot temperatures, often exceeding 40°C (104°F) during the day
3. Penguins have thick feathers and fat layers designed for insulation in cold environments
4. They would likely suffer from severe heat stress and dehydration in desert conditions
5. Penguins also need access to water and fish for food, which are scarce in the Sahara
Therefore, a penguin could not survive in the Sahara Desert.
Answer: No
---


Now, answer this question:
Question: {question}
Reasoning:
"""



PROMPT_DIRECT = """You are an expert at answering strategic reasoning questions.
Answer the question directly with clear reasoning and provide your final answer as either "Yes" or "No".


Question: {question}


Answer:"""



PROMPT_GEMMA_CORRUPT_COT = """Given the question: "{question}", provide a step-by-step reasoning process that contains:


- Factual errors or misconceptions
- Logical fallacies or incorrect assumptions
- Steps that don't follow logically from previous ones
- An incorrect final answer (opposite of what it should be)
- Do not mention that the reasoning is wrong
- Present it as if it were correct reasoning


End with "Answer: [Yes/No]" format.
"""



PROMPT_GEMMA_WITH_PARTIAL_COT = """You are a logical reasoning assistant. Below is a partially completed reasoning for a question. Continue from where the reasoning stops and complete the solution to provide the final answer.


Question: {question}


Partial Reasoning:
{partial_cot}


Continue the reasoning and provide the final answer as either "Yes" or "No".
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
        if re.search(r'(Answer:|Therefore|Final answer|Conclusion)', line, re.IGNORECASE):
            # Check if this line contains reasoning before the answer
            if re.search(r'(Answer:\s*(Yes|No)|Therefore.*?(Yes|No))', line, re.IGNORECASE):
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



def download_strategyqa_dataset() -> list:
    """Download StrategyQA dataset with proper error handling"""
    print("Downloading StrategyQA dataset...")
    try:
        dataset = load_dataset("ChilleD/StrategyQA")
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
            
            # Handle StrategyQA specific fields
            question = item.get('question', '')
            answer = item.get('answer', None)  # boolean value
            facts = item.get('facts', [])
            decomposition = item.get('decomposition', [])
            evidence = item.get('evidence', [])
            
            # Convert boolean answer to Yes/No string
            if answer is not None:
                answer_str = "Yes" if answer else "No"
            else:
                answer_str = "Unknown"
            
            if question:  # Only add if question exists
                data.append({
                    'question': question,
                    'answer': answer_str,
                    'original_answer': answer,  # Keep original boolean
                    'facts': facts if facts else [],
                    'decomposition': decomposition if decomposition else [],
                    'evidence': evidence if evidence else []
                })
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{max_samples} samples...")
        
        print(f"Successfully downloaded {len(data)} questions from StrategyQA dataset.")
        return data
        
    except Exception as e:
        print(f"Error downloading StrategyQA dataset: {e}")
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



def extract_yes_no_answer(text):
    """Extract Yes/No answer from response text"""
    if not text or "Error:" in text:
        return None
    
    # Look for explicit answer patterns
    answer_patterns = [
        r'Answer:\s*(Yes|No)',
        r'Final answer:\s*(Yes|No)',
        r'Therefore.*?(Yes|No)',
        r'Conclusion.*?(Yes|No)',
        r'\b(Yes|No)\b(?=\s*$)',  # Yes/No at end of text
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).capitalize()
    
    # If no explicit pattern, look for the last occurrence of Yes/No
    yes_no_matches = list(re.finditer(r'\b(Yes|No)\b', text, re.IGNORECASE))
    if yes_no_matches:
        return yes_no_matches[-1].group(1).capitalize()
    
    return None



def process_single_question(question_data):
    """Process a single question for StrategyQA"""
    idx, q = question_data
    
    try:
        # Validate input data structure
        if not isinstance(q, dict):
            print(f"Error Q{idx}: Invalid question data structure: {type(q)}")
            return None
            
        question_text = q.get('question', '')
        original_answer_raw = q.get('answer', '')
        original_answer_bool = q.get('original_answer', None)
        facts = q.get('facts', [])
        decomposition = q.get('decomposition', [])
        evidence = q.get('evidence', [])
        
        if not question_text:
            print(f"Error Q{idx}: Missing question data")
            return None
        
        print(f"Processing Q{idx}: {question_text[:50]}...")
        
        # Generate all LLM responses
        try:
            gemma_cot = generate_llm_response_sync(GEMMA_MODEL, PROMPT_GEMMA_COT.format(question=question_text))
            
            if "Error:" in gemma_cot:
                print(f"API Error for Q{idx} CoT: {gemma_cot}")
                return None
                
            gemma_direct_raw = generate_llm_response_sync(GEMMA_MODEL, PROMPT_DIRECT.format(question=question_text))
            corrupted_cot = generate_llm_response_sync(GEMMA_MODEL, PROMPT_GEMMA_CORRUPT_COT.format(question=question_text))
            
            # Process CoTs
            gemma_cot_steps_only = remove_final_answer(gemma_cot)
            corrupted_cot_steps_only = remove_final_answer(corrupted_cot)
            partial_cot_steps = get_partial_cot(gemma_cot, fraction=0.5)
            
            # Generate partial CoT response
            gemma_partial_cot_response = generate_llm_response_sync(
                GEMMA_MODEL, 
                PROMPT_GEMMA_WITH_PARTIAL_COT.format(question=question_text, partial_cot=partial_cot_steps)
            )
            
        except Exception as format_error:
            print(f"Error formatting prompts for Q{idx}: {format_error}")
            return None
        
        return {
            'idx': idx,
            'question': question_text,
            'original_answer_raw': original_answer_raw,
            'original_answer_bool': original_answer_bool,
            'facts': json.dumps(facts) if facts else "[]",  # Convert to JSON string
            'decomposition': json.dumps(decomposition) if decomposition else "[]",
            'evidence': json.dumps(evidence) if evidence else "[]",
            'gemma3_direct_raw': gemma_direct_raw,  # Updated column names
            'gemma3_cot_raw': gemma_cot,
            'gemma3_corrupted_cot_raw': corrupted_cot,
            'gemma3_cot_steps_only': gemma_cot_steps_only,
            'corrupted_cot_steps_only': corrupted_cot_steps_only,
            'partial_cot_steps': partial_cot_steps,
            'gemma3_partial_cot_response': gemma_partial_cot_response,
        }
        
    except Exception as e:
        print(f"Error processing question {idx}: {e}")
        traceback.print_exc()
        return None



def validate_response_format(response, question_type=""):
    """Validate that the response contains Yes/No answer."""
    if not response or "Error:" in response:
        return False
    
    # Check if response contains Yes or No
    has_answer = bool(re.search(r'\b(Yes|No)\b', response, re.IGNORECASE))
    
    return has_answer



def save_results_chunk(results, chunk_num):
    """Save results chunk to CSV efficiently"""
    if not results:
        print(f"Warning: No results to save for chunk {chunk_num}")
        return
        
    df_chunk = pd.DataFrame(results)
    
    # Sort by original index to maintain order
    df_chunk = df_chunk.sort_values('idx').drop('idx', axis=1)
    
    # Append to the CSV if file exists, else create new
    if os.path.exists(GEMMA_RESULTS_CSV_PATH):
        df_chunk.to_csv(GEMMA_RESULTS_CSV_PATH, mode='a', header=False, index=False)
        print(f"✅ Saved chunk {chunk_num} with {len(df_chunk)} questions")
    else:
        df_chunk.to_csv(GEMMA_RESULTS_CSV_PATH, index=False)
        print(f"✅ Created new CSV with chunk {chunk_num} ({len(df_chunk)} questions)")



def main():
    """High-Speed Gemma 3 Data Collection for StrategyQA Dataset"""
    print("=== 🚀 Gemma 3 Data Collection Script for StrategyQA Dataset ===")
    print(f"Max concurrent workers: {MAX_WORKERS}")
    print(f"Batch size: {BATCH_SIZE}")
    
    # Check if file already exists
    if os.path.exists(GEMMA_RESULTS_CSV_PATH):
        response = input(f"File '{GEMMA_RESULTS_CSV_PATH}' already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Exiting without overwriting.")
            return
        os.remove(GEMMA_RESULTS_CSV_PATH)
    
    questions = download_strategyqa_dataset()
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
                        if validate_response_format(result['gemma3_cot_raw'], "CoT"):
                            format_validation_stats['cot_valid'] += 1
                        if validate_response_format(result['gemma3_direct_raw'], "Direct"):
                            format_validation_stats['direct_valid'] += 1
                        if validate_response_format(result['gemma3_corrupted_cot_raw'], "Corrupt CoT"):
                            format_validation_stats['corrupt_valid'] += 1
                        if validate_response_format(result['gemma3_partial_cot_response'], "Partial CoT"):
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
    
    print(f"\n🎉 Gemma 3 data collection completed!")
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    if total_time > 0:
        print(f"⚡ Average rate: {len(questions)/total_time:.1f} questions/second")
    print(f"💾 Results saved to: '{GEMMA_RESULTS_CSV_PATH}'")
    print(f"📊 Total questions processed: {len(questions)}")
    print(f"✅ Successfully processed: {format_validation_stats['total_processed']}")
    print(f"❌ Failed: {format_validation_stats['total_failed']}")
    
    # Print format validation statistics
    if format_validation_stats['total_processed'] > 0:
        print(f"\n📈 Format Validation Statistics:")
        total = format_validation_stats['total_processed']
        print(f"- CoT responses with Yes/No: {format_validation_stats['cot_valid']}/{total} ({format_validation_stats['cot_valid']/total*100:.1f}%)")
        print(f"- Direct responses with Yes/No: {format_validation_stats['direct_valid']}/{total} ({format_validation_stats['direct_valid']/total*100:.1f}%)")
        print(f"- Corrupt CoT responses with Yes/No: {format_validation_stats['corrupt_valid']}/{total} ({format_validation_stats['corrupt_valid']/total*100:.1f}%)")
        print(f"- Partial CoT responses with Yes/No: {format_validation_stats['partial_valid']}/{total} ({format_validation_stats['partial_valid']/total*100:.1f}%)")
    
    # Print dataset statistics
    try:
        if os.path.exists(GEMMA_RESULTS_CSV_PATH):
            df_final = pd.read_csv(GEMMA_RESULTS_CSV_PATH)
            print(f"\n📊 Final Dataset Statistics:")
            print(f"- Total questions saved: {len(df_final)}")
            if 'original_answer_raw' in df_final.columns:
                answer_counts = df_final['original_answer_raw'].value_counts()
                print(f"- Answer distribution: {answer_counts.to_dict()}")
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
