import requests
import json
import re

# --- Configuration ---
API_URL = 'http://127.0.0.1:11434/api/generate'
# GEMMA_MODEL = 'gemma3:27b'
GEMMA_MODEL = 'gemma3:1b'

# --- Prompt Template ---
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

# --- Helper Functions ---
def generate_llm_response(llm_name: str, prompt: str) -> str:
    payload = {"model": llm_name, "prompt": prompt, "stream": False}
    try:
        print(f"Sending request to {llm_name}...")
        response = requests.post(API_URL, json=payload, timeout=300)
        print(f"Response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"Error response: {response.text}")
            return f"Error: HTTP {response.status_code}: {response.text}"
            
        result = response.json()
        return result.get('response', '').strip()
    except requests.exceptions.RequestException as e:
        print(f"Request exception for {llm_name}: {e}")
        return f"Error: API request failed: {str(e)}"
    except json.JSONDecodeError as e:
        print(f"JSON decode error for {llm_name}: {e}")
        return f"Error: Invalid JSON response: {str(e)}"

def extract_answer(text: str):
    """Extract numerical answer from model response"""
    if text is None or "Error:" in str(text):
        return None
    
    # First try to find #### pattern
    m = re.search(r'####(.*)', text)
    if m:
        after_hashes = m.group(1)
        num_match = re.search(r'[-+]?\d*\.?\d+', after_hashes)
        if num_match:
            try:
                return float(num_match.group(0))
            except ValueError:
                pass
    
    # Fallback: find all numbers and return the last one
    all_nums = re.findall(r'[-+]?\d*\.?\d+', text)
    if all_nums:
        try:
            return float(all_nums[-1])
        except ValueError:
            pass
    
    return None

# --- Main Script ---
def main():
    # Your question here
    question = "A store sells 3 types of fruit: apples at $2 each, oranges at $3 each, and bananas at $1 each. If someone buys 4 apples, 2 oranges, and 5 bananas, how much do they spend in total?"
    
    print(f"Question: {question}")
    print(f"Model: {GEMMA_MODEL}")
    print("-" * 50)
    
    # Generate response
    formatted_prompt = PROMPT_GEMMA_COT.format(question=question)
    response = generate_llm_response(GEMMA_MODEL, formatted_prompt)
    
    # Display results
    print("\n--- Model Response ---")
    print(response)
    
    # Extract answer
    extracted_answer = extract_answer(response)
    print(f"\n--- Extracted Answer ---")
    print(f"Answer: {extracted_answer}")
    
    print("\n--- Script Complete ---")

if __name__ == "__main__":
    main()
