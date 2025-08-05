import os
import time
import json
import requests
import pandas as pd
import regex
import re
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
import multiprocessing as mp

# --- Configuration ---
API_URL = 'http://127.0.0.1:11434/api/generate'
SMOLLM2_MODEL = 'smollm2:1.7b'
GEMMA3_RESULTS_CSV_PATH = '../gemma3_math_dataset_results.csv'
FINAL_RESULTS_CSV_PATH = 'gemma3_smollm2_math_final_results.csv'
SUMMARY_CSV_PATH = 'gemma3_smollm2_math_accuracy_summary.csv'
CHUNK_SAVE_SIZE = 20
MAX_WORKERS = min(8, mp.cpu_count())  # Limit concurrent API calls
SIMILARITY_BATCH_SIZE = 100  # Process similarity in batches

# --- Prompt Templates with Fixed JSON Format ---
PROMPT_DIRECT = """You are an expert mathematician solving competition-level problems.
Your task is to solve the given math problem step by step.

At the end, return ONLY valid JSON in this exact format:
{{"answer": "your final answer here"}}

Question: {question}

Solution:"""

PROMPT_SMOLLM2_WITH_COT = """You must ONLY use the provided reasoning below to answer the question. 
Do not add your own reasoning. Continue from where the provided reasoning stops 
and complete the solution.

At the end, return ONLY valid JSON in this exact format:
{{"answer": "your final answer here"}}

Question: {question}

Reasoning:
{cot}

Continue the solution:"""

def remove_final_answer_from_cot(cot_text):
    """
    Remove final answers (boxed content, final answer statements) from CoT reasoning
    to ensure only reasoning steps are passed to SmolLM2
    """
    if not cot_text or pd.isna(cot_text):
        return ""
    
    cot_text = str(cot_text).strip()
    
    # Remove boxed content (LaTeX \boxed{...})
    cot_text = regex.sub(
        r'\\boxed\s*(?P<brace>\{(?:[^{}]|(?&brace))*\})',
        '',
        cot_text,
        flags=regex.VERBOSE
    )
    
    # Remove final answer patterns
    final_answer_patterns = [
        r'(Therefore,?\s*the\s*(final\s*)?answer\s*is.*?)(?=\n\n|\n---|\n$|$)',
        r'(The\s*(final\s*)?answer\s*is.*?)(?=\n\n|\n---|\n$|$)',
        r'(Final\s*answer:.*?)(?=\n\n|\n---|\n$|$)',
        r'(Answer:.*?)(?=\n\n|\n---|\n$|$)',
        r'(\$\$.*?\$\$\s*$)',  # Final mathematical expressions at the end
        r'(So\s*the\s*answer\s*is.*?)(?=\n\n|\n---|\n$|$)',
        r'(Hence,?\s*the\s*answer\s*is.*?)(?=\n\n|\n---|\n$|$)',
        r'(Thus,?\s*the\s*answer\s*is.*?)(?=\n\n|\n---|\n$|$)',
    ]
    
    for pattern in final_answer_patterns:
        cot_text = re.sub(pattern, '', cot_text, flags=re.IGNORECASE | re.DOTALL)
    
    # Remove concluding statements that typically appear at the end
    concluding_patterns = [
        r'(Therefore,?\s*.*?)$',
        r'(In conclusion,?\s*.*?)$',
        r'(Finally,?\s*.*?)$',
        r'(The\s*probability.*?is.*?)$',
        r'(The\s*result.*?is.*?)$',
    ]
    
    for pattern in concluding_patterns:
        # Only remove if it's the last sentence/paragraph
        lines = cot_text.split('\n')
        if lines:
            last_line = lines[-1].strip()
            if re.search(pattern, last_line, flags=re.IGNORECASE):
                lines[-1] = re.sub(pattern, '', last_line, flags=re.IGNORECASE).strip()
                cot_text = '\n'.join(lines)
    
    # Clean up extra whitespace and empty lines at the end
    cot_text = re.sub(r'\n\s*\n\s*$', '\n', cot_text)
    cot_text = cot_text.strip()
    
    # If the CoT ends with a calculation that looks like a final answer, remove it
    lines = cot_text.split('\n')
    if lines:
        last_line = lines[-1].strip()
        # Remove lines that are just calculations or fractions at the end
        if re.match(r'^[=\s]*\$?\\?frac\{.*?\}\{.*?\}\$?\s*[.]?\s*$', last_line) or \
           re.match(r'^[=\s]*\d+/\d+\s*[.]?\s*$', last_line) or \
           re.match(r'^[=\s]*\d+\.\d+\s*[.]?\s*$', last_line) or \
           re.match(r'^[=\s]*\d+\s*[.]?\s*$', last_line):
            lines = lines[:-1]
            cot_text = '\n'.join(lines).strip()
    
    return cot_text

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

def process_single_row(args):
    """Process a single row for parallel execution"""
    idx, row_data, smollm2_model, prompt_direct, prompt_cot = args
    
    try:
        q, cot_steps_only, corrupted_steps_only, partial_steps_only = row_data
        
        # Generate SmolLM2 responses
        smollm2_direct = generate_llm_response(smollm2_model, prompt_direct.format(question=q))
        smollm2_with_cot = generate_llm_response(smollm2_model, prompt_cot.format(question=q, cot=cot_steps_only))
        smollm2_with_corr = generate_llm_response(smollm2_model, prompt_cot.format(question=q, cot=corrupted_steps_only))
        smollm2_with_part = generate_llm_response(smollm2_model, prompt_cot.format(question=q, cot=partial_steps_only))
        
        return idx, {
            'smollm2_direct_raw': smollm2_direct,
            'smollm2_with_cot_raw': smollm2_with_cot,
            'smollm2_with_corrupted_cot_raw': smollm2_with_corr,
            'smollm2_with_partial_cot_raw': smollm2_with_part
        }
    except Exception as e:
        return idx, {
            'smollm2_direct_raw': f"Error: {e}",
            'smollm2_with_cot_raw': f"Error: {e}",
            'smollm2_with_corrupted_cot_raw': f"Error: {e}",
            'smollm2_with_partial_cot_raw': f"Error: {e}"
        }

def safe_get_value(row, column, default=""):
    """Safely get value from DataFrame row"""
    try:
        value = row[column]
        if pd.isna(value):
            return default
        return str(value)
    except (KeyError, AttributeError):
        print(f"Warning: Column '{column}' not found")
        return default

def find_boxed_content(text: str):
    """Extract content from \\boxed{} with nested brace support"""
    if not isinstance(text, str) or not text:
        return []
    try:
        pattern = regex.compile(
            r'\\boxed\s*(?P<brace>\{(?:[^{}]|(?&brace))*\})',
            regex.VERBOSE
        )
        matches = pattern.findall(text)
        return [m[1:-1].strip() for m in matches]
    except Exception as e:
        print(f"Warning: Error in find_boxed_content: {e}")
        return []

def parse_json_answer(raw: str):
    """Enhanced JSON answer extraction with multiple fallback strategies"""
    if not raw or pd.isna(raw):
        return ""
    
    raw = str(raw).strip()
    
    # Strategy 1: Try JSON parsing - look for the last JSON object
    json_matches = re.findall(r'\{[^{}]*"answer"[^{}]*\}', raw, re.IGNORECASE)
    for json_str in reversed(json_matches):
        try:
            obj = json.loads(json_str)
            answer = obj.get("answer", "")
            if answer:
                return str(answer).strip()
        except:
            continue
    
    # Strategy 2: Try simpler JSON extraction with quotes
    json_pattern = r'\{\s*"answer"\s*:\s*"([^"]+)"\s*\}'
    match = re.search(json_pattern, raw, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    
    # Strategy 3: Try without quotes around the answer
    json_pattern2 = r'\{\s*"answer"\s*:\s*([^,}]+)\s*\}'
    match = re.search(json_pattern2, raw, re.IGNORECASE)
    if match:
        return match.group(1).strip().strip('"')
    
    # Strategy 4: Look for "Answer:" pattern (common in SmolLM2 responses)
    answer_patterns = [
        r'Answer:\s*(.+?)(?:\n|$)',
        r'The answer is:\s*(.+?)(?:\n|$)',
        r'Therefore,?\s*the answer is\s*(.+?)(?:\n|$)',
        r'Final answer:\s*(.+?)(?:\n|$)',
        r'The probability.*?is\s*(.+?)(?:\n|$|\.)',
        r'probability.*?=\s*(.+?)(?:\n|$|\.)',
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, raw, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            # Clean up common endings
            answer = re.sub(r'[.,$]$', '', answer)
            if answer:
                return answer
    
    # Strategy 5: Look for fraction patterns
    fraction_patterns = [
        r'(\d+/\d+)',
        r'(\\frac\{[^}]+\}\{[^}]+\})',
        r'(\d+\.\d+)',
    ]
    
    for pattern in fraction_patterns:
        matches = re.findall(pattern, raw)
        if matches:
            return matches[-1]  # Take the last fraction found
    
    # Strategy 6: Fallback to boxed content
    boxed = find_boxed_content(raw)
    if boxed:
        return boxed[-1]
    
    # Strategy 7: Look for numerical values at the end
    number_match = re.search(r'(\d+(?:\.\d+)?)\s*[.!]?\s*$', raw)
    if number_match:
        return number_match.group(1)
    
    # Strategy 8: Extract key mathematical expressions
    math_expressions = re.findall(r'(\d+/\d+|\d+\.\d+|\d+)', raw)
    if math_expressions:
        return math_expressions[-1]
    
    # Final fallback - return cleaned raw text (first 50 chars)
    cleaned = re.sub(r'[^\w\s/\\{}.]', ' ', raw)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned[:50]

def extract_answer_text(text):
    """Extract clean answer text for similarity comparison with better preprocessing"""
    if not text or pd.isna(text):
        return ""
    
    text = str(text).strip()
    
    # Handle common fraction formats
    text = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'\1/\2', text)
    
    # Remove common LaTeX commands and formatting
    text = re.sub(r'\\[a-zA-Z]+\{([^}]*)\}', r'\1', text)  # Remove \command{...}
    text = re.sub(r'\\[a-zA-Z]+', '', text)  # Remove standalone commands
    text = re.sub(r'[{}]', '', text)  # Remove braces
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    
    # Clean up common phrases
    text = re.sub(r'the probability.*?is\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'answer:\s*', '', text, flags=re.IGNORECASE)
    
    return text.strip()

def check_exact_match(ref_text, model_text):
    """Check for exact matches (case-insensitive)"""
    if not ref_text or not model_text:
        return False
    return ref_text.lower().strip() == model_text.lower().strip()

def check_numerical_equivalence(ref_text, model_text):
    """Check for numerical equivalence"""
    try:
        # Try to evaluate as fractions or decimals
        ref_num = eval(ref_text.replace('/', '/')) if '/' in ref_text else float(ref_text)
        model_num = eval(model_text.replace('/', '/')) if '/' in model_text else float(model_text)
        return abs(ref_num - model_num) < 1e-6
    except:
        return False

def compute_semantic_similarity_batch(text_pairs):
    """Compute semantic similarity for a batch of text pairs using TF-IDF"""
    if not text_pairs:
        return []
    
    try:
        # Flatten pairs for vectorization
        all_texts = []
        for ref, model in text_pairs:
            all_texts.extend([ref, model])
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1, 2), min_df=1)
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Compute similarities for each pair
        similarities = []
        for i in range(0, len(all_texts), 2):
            ref_vector = tfidf_matrix[i:i+1]
            model_vector = tfidf_matrix[i+1:i+2]
            sim = cosine_similarity(ref_vector, model_vector)[0][0] * 100
            similarities.append(sim)
        
        return similarities
    except Exception as e:
        print(f"Warning: Batch TF-IDF computation failed: {e}")
        return [0.0] * len(text_pairs)

def compute_similarity_optimized(reference_answers, model_answers):
    """Optimized similarity computation with streamlined logic"""
    # Extract clean text
    ref_texts = [extract_answer_text(ans) for ans in reference_answers]
    model_texts = [extract_answer_text(ans) for ans in model_answers]
    
    similarities = []
    semantic_pairs = []
    semantic_indices = []
    
    # First pass: Check exact matches and numerical equivalence
    for i, (ref, model) in enumerate(zip(ref_texts, model_texts)):
        if not ref or not model:
            similarities.append(0.0)
        elif check_exact_match(ref, model):
            similarities.append(100.0)
        elif check_numerical_equivalence(ref, model):
            similarities.append(100.0)
        else:
            # Queue for semantic similarity computation
            semantic_pairs.append((ref, model))
            semantic_indices.append(i)
            similarities.append(None)  # Placeholder
    
    # Second pass: Compute semantic similarities in batches
    if semantic_pairs:
        # Process in batches to avoid memory issues
        for i in range(0, len(semantic_pairs), SIMILARITY_BATCH_SIZE):
            batch_pairs = semantic_pairs[i:i+SIMILARITY_BATCH_SIZE]
            batch_indices = semantic_indices[i:i+SIMILARITY_BATCH_SIZE]
            batch_similarities = compute_semantic_similarity_batch(batch_pairs)
            
            # Fill in the semantic similarities
            for idx, sim in zip(batch_indices, batch_similarities):
                similarities[idx] = sim
    
    # Ensure no None values remain
    similarities = [sim if sim is not None else 0.0 for sim in similarities]
    
    return similarities

def check_if_processing_needed(csv_path):
    """Check if SmolLM2 processing is already complete"""
    if not os.path.exists(csv_path):
        return True
    
    try:
        df = pd.read_csv(csv_path)
        smollm2_cols = [
            'smollm2_direct_raw',
            'smollm2_with_cot_raw', 
            'smollm2_with_corrupted_cot_raw',
            'smollm2_with_partial_cot_raw'
        ]
        
        # Check if all SmolLM2 columns exist and have data
        for col in smollm2_cols:
            if col not in df.columns:
                return True
            if df[col].isna().all():
                return True
        
        print(f"SmolLM2 processing already complete. Found {len(df)} rows with SmolLM2 data.")
        return False
    except Exception as e:
        print(f"Error checking existing CSV: {e}")
        return True

def main():
    print("=== Enhanced SmolLM2 Processing with Parallel Execution (Gemma3 Dataset) ===")

    # Check if processing is needed
    if not check_if_processing_needed(FINAL_RESULTS_CSV_PATH):
        print("Skipping SmolLM2 processing, moving directly to evaluation...")
    else:
        # Check CSV existence
        if not os.path.exists(GEMMA3_RESULTS_CSV_PATH):
            print(f"Error: Gemma3 results file '{GEMMA3_RESULTS_CSV_PATH}' not found.")
            return

        # Load CSV with error handling
        try:
            df_gemma3 = pd.read_csv(GEMMA3_RESULTS_CSV_PATH)
            print(f"Loaded {len(df_gemma3)} rows")
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return

        # Validate required columns (updated for Gemma3)
        required = [
            'question', 'original_answer_raw', 'gemma3_direct_raw', 'gemma3_cot_raw',
            'gemma3_corrupted_cot_raw', 'gemma3_cot_steps_only', 'corrupted_cot_steps_only',
            'partial_cot_steps', 'gemma3_partial_cot_response'
        ]
        missing = [c for c in required if c not in df_gemma3.columns]
        if missing:
            print(f"Error: Missing columns: {missing}")
            print(f"Available columns: {df_gemma3.columns.tolist()}")
            return

        # Check for overwrite
        if os.path.exists(FINAL_RESULTS_CSV_PATH):
            os.remove(FINAL_RESULTS_CSV_PATH)

        # Prepare data for parallel processing
        print(f"Preparing data for parallel processing with {MAX_WORKERS} workers...")
        
        # Preprocess all CoT texts
        processed_data = []
        for idx, row in df_gemma3.iterrows():
            q = safe_get_value(row, 'question')
            if not q.strip():
                continue
                
            cot_raw = safe_get_value(row, 'gemma3_cot_steps_only')
            corrupted_raw = safe_get_value(row, 'corrupted_cot_steps_only')
            partial_raw = safe_get_value(row, 'partial_cot_steps')

            # Remove final answers from CoT reasoning
            cot_steps_only = remove_final_answer_from_cot(cot_raw)
            corrupted_steps_only = remove_final_answer_from_cot(corrupted_raw)
            partial_steps_only = remove_final_answer_from_cot(partial_raw)
            
            processed_data.append({
                'idx': idx,
                'row': row,
                'question': q,
                'cot_steps_only': cot_steps_only,
                'corrupted_steps_only': corrupted_steps_only,
                'partial_steps_only': partial_steps_only
            })

        # Process in chunks with parallel execution
        combined = []
        chunk_size = min(MAX_WORKERS * 2, len(processed_data))
        
        for chunk_start in range(0, len(processed_data), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(processed_data))
            chunk_data = processed_data[chunk_start:chunk_end]
            
            print(f"Processing chunk {chunk_start//chunk_size + 1}/{(len(processed_data) + chunk_size - 1)//chunk_size}")
            
            # Prepare arguments for parallel processing
            args_list = []
            for item in chunk_data:
                row_data = (
                    item['question'],
                    item['cot_steps_only'],
                    item['corrupted_steps_only'],
                    item['partial_steps_only']
                )
                args_list.append((item['idx'], row_data, SMOLLM2_MODEL, PROMPT_DIRECT, PROMPT_SMOLLM2_WITH_COT))
            
            # Process chunk in parallel
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                results = list(tqdm(
                    executor.map(process_single_row, args_list),
                    total=len(args_list),
                    desc=f"Processing SmolLM2 responses"
                ))
            
            # Combine results with original data
            for (idx, smollm2_results), item in zip(results, chunk_data):
                row = item['row']
                combined.append({
                    'question': item['question'],
                    'original_answer_raw': safe_get_value(row, 'original_answer_raw'),
                    'gemma3_direct_raw': safe_get_value(row, 'gemma3_direct_raw'),
                    'gemma3_cot_raw': safe_get_value(row, 'gemma3_cot_raw'),
                    'gemma3_corrupted_cot_raw': safe_get_value(row, 'gemma3_corrupted_cot_raw'),
                    'gemma3_cot_steps_only': item['cot_steps_only'],
                    'corrupted_cot_steps_only': item['corrupted_steps_only'],
                    'partial_cot_steps': item['partial_steps_only'],
                    'gemma3_partial_cot_response': safe_get_value(row, 'gemma3_partial_cot_response'),
                    **smollm2_results
                })
            
            # Save chunk
            if combined:
                df_chunk = pd.DataFrame(combined)
                mode = 'a' if os.path.exists(FINAL_RESULTS_CSV_PATH) else 'w'
                header = not os.path.exists(FINAL_RESULTS_CSV_PATH)
                df_chunk.to_csv(FINAL_RESULTS_CSV_PATH, mode=mode, header=header, index=False)
                combined = []
                print(f"Saved chunk to {FINAL_RESULTS_CSV_PATH}")

    # Load and evaluate results with optimized similarity computation
    try:
        if not os.path.exists(FINAL_RESULTS_CSV_PATH):
            print(f"Error: Results file {FINAL_RESULTS_CSV_PATH} not found")
            return
            
        df_combined = pd.read_csv(FINAL_RESULTS_CSV_PATH)
        print(f"Loaded {len(df_combined)} rows for evaluation")
        
        # Extract clean answer texts
        print("Extracting answers...")
        
        # Extract original answer (reference)
        original_boxed = df_combined['original_answer_raw'].apply(lambda x: find_boxed_content(str(x)))
        df_combined['original_answer_text'] = original_boxed.apply(lambda x: x[-1] if x else "")
        
        # Extract Gemma3 answers
        gemma3_direct_boxed = df_combined['gemma3_direct_raw'].apply(lambda x: find_boxed_content(str(x)))
        df_combined['gemma3_direct_text'] = gemma3_direct_boxed.apply(lambda x: x[-1] if x else "")
        
        gemma3_cot_boxed = df_combined['gemma3_cot_raw'].apply(lambda x: find_boxed_content(str(x)))
        df_combined['gemma3_cot_text'] = gemma3_cot_boxed.apply(lambda x: x[-1] if x else "")
        
        gemma3_corrupted_boxed = df_combined['gemma3_corrupted_cot_raw'].apply(lambda x: find_boxed_content(str(x)))
        df_combined['gemma3_corrupted_cot_text'] = gemma3_corrupted_boxed.apply(lambda x: x[-1] if x else "")
        
        gemma3_partial_boxed = df_combined['gemma3_partial_cot_response'].apply(lambda x: find_boxed_content(str(x)))
        df_combined['gemma3_partial_cot_text'] = gemma3_partial_boxed.apply(lambda x: x[-1] if x else "")
        
        # Extract SmolLM2 answers using enhanced JSON/pattern parsing
        print("Parsing SmolLM2 answers with enhanced extraction...")
        df_combined['smollm2_direct_text'] = df_combined['smollm2_direct_raw'].apply(parse_json_answer)
        df_combined['smollm2_with_cot_text'] = df_combined['smollm2_with_cot_raw'].apply(parse_json_answer)
        df_combined['smollm2_with_corrupted_cot_text'] = df_combined['smollm2_with_corrupted_cot_raw'].apply(parse_json_answer)
        df_combined['smollm2_with_partial_cot_text'] = df_combined['smollm2_with_partial_cot_raw'].apply(parse_json_answer)

        # Compute optimized similarities
        print("Computing optimized similarities...")
        
        reference_answers = df_combined['original_answer_text'].tolist()
        
        # Gemma3 comparisons
        print("Computing Gemma3 similarities...")
        gemma3_direct_sim = compute_similarity_optimized(reference_answers, df_combined['gemma3_direct_text'].tolist())
        gemma3_cot_sim = compute_similarity_optimized(reference_answers, df_combined['gemma3_cot_text'].tolist())
        gemma3_corrupted_sim = compute_similarity_optimized(reference_answers, df_combined['gemma3_corrupted_cot_text'].tolist())
        gemma3_partial_sim = compute_similarity_optimized(reference_answers, df_combined['gemma3_partial_cot_text'].tolist())
        
        # SmolLM2 comparisons
        print("Computing SmolLM2 similarities...")
        smollm2_direct_sim = compute_similarity_optimized(reference_answers, df_combined['smollm2_direct_text'].tolist())
        smollm2_cot_sim = compute_similarity_optimized(reference_answers, df_combined['smollm2_with_cot_text'].tolist())
        smollm2_corrupted_sim = compute_similarity_optimized(reference_answers, df_combined['smollm2_with_corrupted_cot_text'].tolist())
        smollm2_partial_sim = compute_similarity_optimized(reference_answers, df_combined['smollm2_with_partial_cot_text'].tolist())

        # Add similarity scores to dataframe
        df_combined['gemma3_direct_similarity'] = gemma3_direct_sim
        df_combined['gemma3_cot_similarity'] = gemma3_cot_sim
        df_combined['gemma3_corrupted_cot_similarity'] = gemma3_corrupted_sim
        df_combined['gemma3_partial_cot_similarity'] = gemma3_partial_sim
        df_combined['smollm2_direct_similarity'] = smollm2_direct_sim
        df_combined['smollm2_cot_similarity'] = smollm2_cot_sim
        df_combined['smollm2_corrupted_cot_similarity'] = smollm2_corrupted_sim
        df_combined['smollm2_partial_cot_similarity'] = smollm2_partial_sim

        # Calculate average similarities
        results = {
            "Gemma3 Direct vs. Original": np.mean(gemma3_direct_sim),
            "Gemma3 CoT vs. Original": np.mean(gemma3_cot_sim),
            "Gemma3 Corrupted CoT vs. Original": np.mean(gemma3_corrupted_sim),
            "Gemma3 Partial CoT vs. Original": np.mean(gemma3_partial_sim),
            "SmolLM2 Direct vs. Original": np.mean(smollm2_direct_sim),
            "SmolLM2 w/ CoT vs. Original": np.mean(smollm2_cot_sim),
            "SmolLM2 w/ Corrupted CoT vs. Original": np.mean(smollm2_corrupted_sim),
            "SmolLM2 w/ Partial CoT vs. Original": np.mean(smollm2_partial_sim)
        }

        # Display results
        print("\n" + "="*60)
        print("FINAL RESULTS (SmolLM2 vs Gemma3 Optimized Processing)")
        print("="*60)
        for name, acc in results.items():
            print(f"{name}: {acc:.2f}%")

        # Save results
        df_combined.to_csv(FINAL_RESULTS_CSV_PATH, index=False)
        pd.DataFrame([
            {'setting': k, 'accuracy': v} for k, v in results.items()
        ]).to_csv(SUMMARY_CSV_PATH, index=False)

        print(f"\nSaved results to {FINAL_RESULTS_CSV_PATH}")
        print(f"Saved summary to {SUMMARY_CSV_PATH}")

    except Exception as e:
        print(f"Error in evaluation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
