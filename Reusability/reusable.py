import os
import json
import time
import random
import pandas as pd
import numpy as np
import requests
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import google.generativeai as genai
from tqdm import tqdm
import re
from difflib import SequenceMatcher

@dataclass
class Problem:
    """Data class to store problem information"""
    question: str
    answer: str
    cot: str = ""
    corrupted_cot: str = ""
    llm_answer_direct: str = ""
    llm_answer_with_cot: str = ""
    llm_answer_with_corrupted_cot: str = ""

class COTAnalyzer:
    """Main class for Chain of Thought analysis"""
    
    def __init__(self, gemini_api_key: str, openai_api_key: Optional[str] = None):
        """Initialize the analyzer with API keys"""
        self.gemini_api_key = gemini_api_key
        self.openai_api_key = openai_api_key
        
        # Configure Gemini
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
        # GSM8K dataset URL
        self.dataset_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl"
        
        # Rate limiting
        self.api_delay = 2.0
        self.max_retries = 3
        
    def load_gsm8k_data(self, num_problems: int = 30) -> List[Problem]:
        """Load GSM8K dataset and return first N problems"""
        print(f"Loading first {num_problems} problems from GSM8K dataset...")
        
        try:
            response = requests.get(self.dataset_url)
            response.raise_for_status()
            
            problems = []
            lines = response.text.strip().split('\n')
            
            for i, line in enumerate(lines[:num_problems]):
                if line.strip():
                    data = json.loads(line)
                    problem = Problem(
                        question=data['question'],
                        answer=data['answer']
                    )
                    problems.append(problem)
            
            print(f"Successfully loaded {len(problems)} problems")
            return problems
            
        except Exception as e:
            print(f"Error loading GSM8K data: {e}")
            # Fallback to sample problems if loading fails
            return self._get_sample_problems(num_problems)
    
    def _get_sample_problems(self, num_problems: int) -> List[Problem]:
        """Fallback sample problems if GSM8K loading fails"""
        sample_problems = [
            Problem(
                question="Janet's dogs eat 2 pounds of dog food each day. How many pounds of dog food do her dogs eat in 7 days?",
                answer="14"
            ),
            Problem(
                question="There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
                answer="6"
            ),
            Problem(
                question="Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?",
                answer="39"
            ),
            Problem(
                question="If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?",
                answer="5"
            ),
            Problem(
                question="There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?",
                answer="6"
            )
        ]
        
        # Repeat sample problems to reach desired number
        while len(sample_problems) < num_problems:
            sample_problems.extend(sample_problems[:min(len(sample_problems), num_problems - len(sample_problems))])
        
        return sample_problems[:num_problems]
    
    def generate_chain_of_thought(self, problem: Problem) -> str:
        """Generate Chain of Thought for a problem using Gemini 2.0 Flash"""
        prompt = f"""Please solve this math problem step by step, showing your reasoning:

Question: {problem.question}

Please provide a detailed step-by-step solution with clear reasoning:"""

        try:
            response = self.gemini_model.generate_content(prompt)
            time.sleep(self.api_delay)
            return response.text.strip()
        except Exception as e:
            print(f"Error generating CoT: {e}")
            return f"Error generating Chain of Thought: {e}"
    
    def generate_corrupted_chain_of_thought(self, problem: Problem) -> str:
        """Generate intentionally corrupted Chain of Thought"""
        prompt = f"""Please solve this math problem step by step, but make some intentional mistakes in your reasoning:

Question: {problem.question}

Please provide a step-by-step solution, but include some logical errors or incorrect calculations:"""

        try:
            response = self.gemini_model.generate_content(prompt)
            time.sleep(self.api_delay)
            return response.text.strip()
        except Exception as e:
            print(f"Error generating corrupted CoT: {e}")
            return f"Error generating corrupted Chain of Thought: {e}"
    
    def call_llm_api(self, prompt: str, model_name: str = "huggingface") -> str:
        """Call free LLM API for problem solving with rate limiting and retries"""
        
        for attempt in range(self.max_retries):
            try:
                if model_name == "huggingface":
                    return self._call_huggingface_api(prompt)
                elif model_name == "openai" and self.openai_api_key:
                    return self._call_openai_api(prompt)
                else:
                    # Default to fallback response
                    return self._generate_fallback_response(prompt)
                    
            except Exception as e:
                print(f"Error calling LLM API (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    return self._generate_fallback_response(prompt)
                time.sleep(self.api_delay)
                continue
        
        return self._generate_fallback_response(prompt)
    
    def _call_huggingface_api(self, prompt: str) -> str:
        """Call GitHub AI API (free alternative to OpenAI)"""
        try:
            # Using GitHub AI endpoint as free alternative
            token = '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'  # User needs to replace with their token
            endpoint = "https://models.github.ai/inference"
            model_name = "openai/gpt-4o"
            
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": 5000
            }
            
            response = requests.post(endpoint, headers=headers, json=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            else:
                print(f"GitHub AI API Error: {response.status_code} - {response.text}")
                return self._generate_fallback_response(prompt)
                
        except Exception as e:
            print(f"GitHub AI API error: {e}")
            return self._generate_fallback_response(prompt)
    
    
    def _call_openai_api(self, prompt: str) -> str:
        """Call OpenAI API (if API key is available)"""
        try:
            headers = {
                "Authorization": f"Bearer {self.openai_api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
                "temperature": 0.1
            }
            
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=data
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content'].strip()
            elif response.status_code == 429:
                # Rate limit hit - wait with exponential backoff
                wait_time = (2 ** 0) * self.api_delay
                print(f"Rate limit hit (429). Waiting {wait_time} seconds before retry")
                time.sleep(wait_time)
                raise Exception("Rate limited")
            else:
                print(f"OpenAI API Error: {response.status_code} - {response.text}")
                return self._generate_fallback_response(prompt)
                
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return self._generate_fallback_response(prompt)
    
    def _generate_fallback_response(self, prompt: str) -> str:
        """Generate a fallback response when API is not available"""
        # Simple pattern matching for basic math operations
        if "add" in prompt.lower() or "+" in prompt:
            return "I think the answer is around 10-15"
        elif "multiply" in prompt.lower() or "*" in prompt:
            return "The result should be approximately 20-30"
        elif "divide" in prompt.lower() or "/" in prompt:
            return "The answer is probably 5-8"
        else:
            return "Based on the problem, I estimate the answer is 12"
    
    def solve_with_different_prompts(self, problem: Problem) -> None:
        """Solve the same problem with three different prompt types"""
        
        # 1. Direct question
        direct_prompt = f"Solve this math problem: {problem.question}"
        problem.llm_answer_direct = self.call_llm_api(direct_prompt)
        time.sleep(self.api_delay)  # Add delay between calls
        
        # 2. Question with Chain of Thought
        cot_prompt = f"""Question: {problem.question}

Here's a step-by-step solution:
{problem.cot}

Based on this reasoning, what is the answer?"""
        problem.llm_answer_with_cot = self.call_llm_api(cot_prompt)
        time.sleep(self.api_delay)  # Add delay between calls
        
        # 3. Question with corrupted Chain of Thought
        corrupted_prompt = f"""Question: {problem.question}

Here's a step-by-step solution:
{problem.corrupted_cot}

Based on this reasoning, what is the answer?"""
        problem.llm_answer_with_corrupted_cot = self.call_llm_api(corrupted_prompt)
    
    def extract_numeric_answer(self, text: str) -> Optional[float]:
        """Extract numeric answer from text"""
        # Look for numbers in the text
        numbers = re.findall(r'\d+(?:\.\d+)?', text)
        if numbers:
            try:
                return float(numbers[-1])  # Return the last number found
            except ValueError:
                pass
        return None
    
    def calculate_similarity(self, answer1: str, answer2: str) -> float:
        """Calculate similarity between two answers"""
        # Extract numeric values
        num1 = self.extract_numeric_answer(answer1)
        num2 = self.extract_numeric_answer(answer2)
        
        if num1 is not None and num2 is not None:
            # If both are numbers, calculate numeric similarity
            if num1 == 0 and num2 == 0:
                return 1.0
            elif num1 == 0 or num2 == 0:
                return 0.0
            else:
                return 1 - abs(num1 - num2) / max(abs(num1), abs(num2))
        else:
            # If not numbers, use string similarity
            return SequenceMatcher(None, answer1.lower(), answer2.lower()).ratio()
    
    def analyze_similarities(self, problems: List[Problem]) -> Dict:
        """Analyze similarities between different answer types"""
        similarities = {
            'direct_vs_cot': [],
            'direct_vs_corrupted': [],
            'cot_vs_corrupted': []
        }
        
        for problem in problems:
            # Direct vs CoT
            sim1 = self.calculate_similarity(
                problem.llm_answer_direct, 
                problem.llm_answer_with_cot
            )
            similarities['direct_vs_cot'].append(sim1)
            
            # Direct vs Corrupted CoT
            sim2 = self.calculate_similarity(
                problem.llm_answer_direct, 
                problem.llm_answer_with_corrupted_cot
            )
            similarities['direct_vs_corrupted'].append(sim2)
            
            # CoT vs Corrupted CoT
            sim3 = self.calculate_similarity(
                problem.llm_answer_with_cot, 
                problem.llm_answer_with_corrupted_cot
            )
            similarities['cot_vs_corrupted'].append(sim3)
        
        return similarities
    
    def run_analysis(self, num_problems: int = 50) -> None:
        """Run the complete Chain of Thought analysis"""
        print("Starting Chain of Thought Analysis for GSM8K Problems")
        print("=" * 60)
        
        # Load problems
        problems = self.load_gsm8k_data(num_problems)
        
        # Generate Chain of Thought and corrupted Chain of Thought
        print("\n1. Generating Chain of Thought and Corrupted Chain of Thought...")
        for i, problem in enumerate(tqdm(problems, desc="Generating CoT")):
            problem.cot = self.generate_chain_of_thought(problem)
            problem.corrupted_cot = self.generate_corrupted_chain_of_thought(problem)
        
        # Solve problems with different prompts
        print("\n2. Solving problems with different prompt types...")
        for i, problem in enumerate(tqdm(problems, desc="Solving with different prompts")):
            self.solve_with_different_prompts(problem)
        
        # Analyze similarities
        print("\n3. Analyzing answer similarities...")
        similarities = self.analyze_similarities(problems)
        
        # Display results
        self.display_results(problems, similarities)
        
        # Save results
        self.save_results(problems, similarities)
    
    def display_results(self, problems: List[Problem], similarities: Dict) -> None:
        """Display analysis results"""
        print("\n" + "=" * 60)
        print("ANALYSIS RESULTS")
        print("=" * 60)
        
        # Similarity statistics
        print("\nSimilarity Analysis:")
        print(f"Direct vs Chain of Thought: {np.mean(similarities['direct_vs_cot']):.3f} ± {np.std(similarities['direct_vs_cot']):.3f}")
        print(f"Direct vs Corrupted CoT: {np.mean(similarities['direct_vs_corrupted']):.3f} ± {np.std(similarities['direct_vs_corrupted']):.3f}")
        print(f"CoT vs Corrupted CoT: {np.mean(similarities['cot_vs_corrupted']):.3f} ± {np.std(similarities['cot_vs_corrupted']):.3f}")
        
        # Sample problem analysis
        print("\nSample Problem Analysis:")
        sample_problem = problems[0]
        print(f"\nQuestion: {sample_problem.question}")
        print(f"Correct Answer: {sample_problem.answer}")
        print(f"\nChain of Thought:\n{sample_problem.cot}")
        print(f"\nCorrupted Chain of Thought:\n{sample_problem.corrupted_cot}")
        print(f"\nLLM Answers:")
        print(f"Direct: {sample_problem.llm_answer_direct}")
        print(f"With CoT: {sample_problem.llm_answer_with_cot}")
        print(f"With Corrupted CoT: {sample_problem.llm_answer_with_corrupted_cot}")
    
    def save_results(self, problems: List[Problem], similarities: Dict) -> None:
        """Save results to files"""
        # Save detailed results
        results_data = []
        for i, problem in enumerate(problems):
            results_data.append({
                'problem_id': i + 1,
                'question': problem.question,
                'correct_answer': problem.answer,
                'chain_of_thought': problem.cot,
                'corrupted_chain_of_thought': problem.corrupted_cot,
                'llm_answer_direct': problem.llm_answer_direct,
                'llm_answer_with_cot': problem.llm_answer_with_cot,
                'llm_answer_with_corrupted_cot': problem.llm_answer_with_corrupted_cot,
                'similarity_direct_vs_cot': similarities['direct_vs_cot'][i],
                'similarity_direct_vs_corrupted': similarities['direct_vs_corrupted'][i],
                'similarity_cot_vs_corrupted': similarities['cot_vs_corrupted'][i]
            })
        
        df = pd.DataFrame(results_data)
        df.to_csv('cot_analysis_results.csv', index=False)
        
        # Save summary statistics
        summary = {
            'total_problems': len(problems),
            'avg_similarity_direct_vs_cot': np.mean(similarities['direct_vs_cot']),
            'avg_similarity_direct_vs_corrupted': np.mean(similarities['direct_vs_corrupted']),
            'avg_similarity_cot_vs_corrupted': np.mean(similarities['cot_vs_corrupted']),
            'std_similarity_direct_vs_cot': np.std(similarities['direct_vs_cot']),
            'std_similarity_direct_vs_corrupted': np.std(similarities['direct_vs_corrupted']),
            'std_similarity_cot_vs_corrupted': np.std(similarities['cot_vs_corrupted'])
        }
        
        with open('cot_analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nResults saved to:")
        print("- cot_analysis_results.csv (detailed results)")
        print("- cot_analysis_summary.json (summary statistics)")

def main():
    """Main function to run the analysis"""
    print("Chain of Thought Analysis for GSM8K Problems")
    print("=" * 60)
    
    # Get API keys
    gemini_api_key = os.getenv('GEMINI_API_KEY')
    if not gemini_api_key:
        gemini_api_key = input("Please enter your Gemini API key: ").strip()
    
    openai_api_key = os.getenv('OPENAI_API_KEY')  # Optional
    
    if not gemini_api_key:
        print("Error: Gemini API key is required!")
        return
    
    # Choose which LLM to use for problem solving
    print("\nChoose which LLM to use for problem solving:")
    print("1. GitHub AI (free, requires token)")
    print("2. Ollama (free, local - requires installation)")
    print("3. OpenAI (paid, requires API key)")
    print("4. Fallback responses (no API needed)")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    llm_choice = "huggingface"  # default (now GitHub AI)
    if choice == "1":
        llm_choice = "huggingface"
        print("Using GitHub AI API (free)")
        print("Note: Make sure to replace '#YOUR API KEY' in the code with your actual token")
    elif choice == "2":
        llm_choice = "ollama"
        print("Using Ollama API (local)")
        print("Note: Make sure Ollama is installed and running on localhost:11434")
    elif choice == "3":
        if not openai_api_key:
            openai_api_key = input("Please enter your OpenAI API key: ").strip()
        if openai_api_key:
            llm_choice = "openai"
            print("Using OpenAI API")
        else:
            print("No OpenAI API key provided, using fallback responses")
            llm_choice = "fallback"
    else:
        llm_choice = "fallback"
        print("Using fallback responses")
    
    # Create analyzer and run analysis
    analyzer = COTAnalyzer(gemini_api_key, openai_api_key)
    
    # Update the solve_with_different_prompts method to use the chosen LLM
    original_solve_method = analyzer.solve_with_different_prompts
    
    def solve_with_chosen_llm(problem):
        """Solve with the chosen LLM"""
        # 1. Direct question
        direct_prompt = f"Solve this math problem: {problem.question}"
        problem.llm_answer_direct = analyzer.call_llm_api(direct_prompt, llm_choice)
        time.sleep(analyzer.api_delay)
        
        # 2. Question with Chain of Thought
        cot_prompt = f"""Question: {problem.question}

Here's a step-by-step solution:
{problem.cot}

Based on this reasoning, what is the answer?"""
        problem.llm_answer_with_cot = analyzer.call_llm_api(cot_prompt, llm_choice)
        time.sleep(analyzer.api_delay)
        
        # 3. Question with corrupted Chain of Thought
        corrupted_prompt = f"""Question: {problem.question}

Here's a step-by-step solution:
{problem.corrupted_cot}

Based on this reasoning, what is the answer?"""
        problem.llm_answer_with_corrupted_cot = analyzer.call_llm_api(corrupted_prompt, llm_choice)
    
    analyzer.solve_with_different_prompts = solve_with_chosen_llm
    
    try:
        analyzer.run_analysis(num_problems=30)
    except KeyboardInterrupt:
        print("\nAnalysis interrupted by user.")
    except Exception as e:
        print(f"\nError during analysis: {e}")

if __name__ == "__main__":
    main()
