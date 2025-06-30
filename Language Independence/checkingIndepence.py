import pandas as pd
import numpy as np
import google.generativeai as genai
from bert_score import score
import requests
import json
import time
from typing import List, Dict, Tuple, Optional
import os
from tqdm import tqdm
import warnings
from config import Config
warnings.filterwarnings('ignore')

class GeminiCOTAnalyzer:
    def __init__(self, api_key: Optional[str] = None):

        if api_key is None:
            api_key = Config.get_api_key()
        
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
    def download_gsm8k_dataset(self, num_samples: Optional[int] = None) -> pd.DataFrame:
        
        if num_samples is None:
            num_samples = Config.NUM_SAMPLES
            
        print("Downloading GSM8K dataset...")
        
        try:
            response = requests.get(Config.DATASET_URL)
            response.raise_for_status()
            
            # Parse JSONL format
            data = []
            for line in response.text.strip().split('\n'):
                if line:
                    data.append(json.loads(line))
            
            # Convert to DataFrame
            df = pd.DataFrame(data)
            
            # Take first num_samples
            df = df.head(num_samples)
            
            print(f"Successfully downloaded {len(df)} samples from GSM8K dataset")
            return df
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("Creating sample data for testing...")
            
            # Create sample data for testing
            sample_data = []
            for i in range(num_samples):
                sample_data.append({
                    'question': f'What is {i+1} + {i+2}?',
                    'answer': f'To solve {i+1} + {i+2}, we add the numbers: {i+1} + {i+2} = {i+1+i+2}'
                })
            
            return pd.DataFrame(sample_data)
    
    def call_gemini_api(self, prompt: str, max_retries: Optional[int] = None) -> str:

        if max_retries is None:
            max_retries = Config.MAX_RETRIES
            
        for attempt in range(max_retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return f"Error: {str(e)}"
        
        return "Error: Max retries exceeded"
    
    def generate_english_cot(self, question: str) -> str:
   
        prompt = f"""Given the question, let's give the step by step solution:

Question: {question}

Please provide a detailed step-by-step solution in English:"""
        
        return self.call_gemini_api(prompt)
    
    def generate_hindi_cot(self, question: str) -> str:

        prompt = f"""Given the question, let's give the step by step solution in Hindi:

Question: {question}

कृपया हिंदी में विस्तृत चरण-दर-चरण समाधान प्रदान करें:"""
        
        return self.call_gemini_api(prompt)
    
    def convert_hindi_to_english(self, hindi_text: str) -> str:

        prompt = f"""Convert the following Hindi Chain of Thought to English:

Hindi Text:
{hindi_text}

Please provide the English translation:"""
        
        return self.call_gemini_api(prompt)
    
    def calculate_bert_score(self, refs: List[str], hyps: List[str]) -> Tuple[List[float], float]:

        try:
            P, R, F1 = score(hyps, refs, lang='en', verbose=True)
            return F1.tolist(), F1.mean().item()
        except Exception as e:
            print(f"Error calculating BERT score: {e}")
            return [0.0] * len(refs), 0.0
    
    def process_dataset(self, num_samples: Optional[int] = None) -> pd.DataFrame:

        # Download dataset
        df = self.download_gsm8k_dataset(num_samples)
        
        results = []
        
        print(f"Processing {len(df)} samples...")
        
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing samples"):
            question = str(row['question'])
            answer = str(row['answer'])
            
            print(f"\nProcessing sample {idx + 1}/{len(df)}")
            print(f"Question: {question[:100]}...")
            
            # Generate English COT
            print("Generating English COT...")
            english_cot = self.generate_english_cot(question)
            print(english_cot)
            time.sleep(Config.API_DELAY)  # Rate limiting
            
            # Generate Hindi COT
            print("Generating Hindi COT...")
            hindi_cot = self.generate_hindi_cot(question)
            print(hindi_cot)
            time.sleep(Config.API_DELAY)  # Rate limiting
            
            # Convert Hindi to English
            print("Converting Hindi to English...")
            hindi_to_english = self.convert_hindi_to_english(hindi_cot)
            print(hindi_to_english)
            time.sleep(Config.API_DELAY)  # Rate limiting
            
            # Store results
            result = {
                'sample_id': int(idx) + 1,
                'question': question,
                'original_answer': answer,
                'english_cot': english_cot,
                'hindi_cot': hindi_cot,
                'hindi_to_english': hindi_to_english
            }
            
            results.append(result)
            
            print(f"Completed sample {int(idx) + 1}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate BERT scores
        print("\nCalculating BERT scores...")
        english_cots = results_df['english_cot'].tolist()
        hindi_to_english_texts = results_df['hindi_to_english'].tolist()
        
        bert_scores, mean_bert_score = self.calculate_bert_score(english_cots, hindi_to_english_texts)

        results_df['bert_score'] = bert_scores
        results_df['mean_bert_score'] = mean_bert_score
        
        return results_df
    
    def save_results(self, results_df: pd.DataFrame, filename: Optional[str] = None):

        if filename is None:
            filename = Config.OUTPUT_FILENAME
            
        results_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
    
    def print_summary(self, results_df: pd.DataFrame):

        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        
        print(f"Total samples processed: {len(results_df)}")
        print(f"Mean BERT Score: {results_df['mean_bert_score'].iloc[0]:.4f}")
        print(f"BERT Score Std Dev: {results_df['bert_score'].std():.4f}")
        print(f"Min BERT Score: {results_df['bert_score'].min():.4f}")
        print(f"Max BERT Score: {results_df['bert_score'].max():.4f}")
        
        # Show some examples
        print("\n" + "="*50)
        print("SAMPLE RESULTS")
        print("="*50)
        
        for i in range(min(3, len(results_df))):
            row = results_df.iloc[i]
            print(f"\nSample {row['sample_id']}:")
            print(f"Question: {row['question'][:100]}...")
            print(f"BERT Score: {row['bert_score']:.4f}")
            print(f"English COT: {row['english_cot'][:200]}...")
            print(f"Hindi to English: {row['hindi_to_english'][:200]}...")

def main():
    """
    Main function to run the analysis
    """
    try:
        # Initialize analyzer (API key will be loaded automatically)
        analyzer = GeminiCOTAnalyzer()
        
        # Process dataset
        results = analyzer.process_dataset()
        
        # Save results
        analyzer.save_results(results)
        
        # Print summary
        analyzer.print_summary(results)
        
        print("\nAnalysis completed successfully!")
        
    except Exception as e:
        print(f"Error running analysis: {e}")
        print("Please check your setup and try again.")

if __name__ == "__main__":
    main() 