import pandas as pd
from bert_score import score
import requests
import json
import time
import os

API_URL = 'http://127.0.0.1:11434/api/generate'
GSM8K_URL = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"

# Assume these functions are defined elsewhere in your project
# from your_llm_library import generate_llm_response, LLM_LIST
# from your_dataset_library import download_gsm8k_last_n_questions

# --- MOCK FUNCTIONS FOR DEMONSTRATION ---
# Replace these with your actual implementations
LLM_LIST = ["gemma3:1b", 'mistral', 'llama3', 'phi3', 'deepseek-llm', 'stablelm2',"qwen3:1.7b"]


def download_gsm8k_last_n_questions(n):
    # This is a mock. In your code, this will download real data.
    response = requests.get(GSM8K_URL)
    response.raise_for_status()
    data = [json.loads(line) for line in response.text.strip().split('\n') if line]
    return data

def generate_llm_response(llm_name, prompt, context=None):
    """
    Generate response from a specific LLM via Ollama API.
    Returns (response_text, new_context, generation_time).
    """
    if context is None:
        context = []
        
    start_time = time.time()
    
    try:
        payload = {
            "model": llm_name,
            "prompt": prompt,
            "stream": False,
            "context": context
        }
        
        response = requests.post(API_URL, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        generation_time = time.time() - start_time
        
        return result.get('response', ''), result.get('context', []), generation_time
    
    except Exception as e:
        # Always return a tuple, with error message in the first element
        return str(e), [], 0.0

# --- END MOCK FUNCTIONS ---


# ==============================================================================
# >> ENHANCED PROMPTS <<
# ==============================================================================
# By defining prompts here, we keep the main function clean and can easily
# modify them.

PROMPT_TEMPLATES = {
    # This prompt sets a clear role and asks for a specific format.
    "english_cot": """You are a meticulous math assistant. Your task is to solve the following word problem by thinking step-by-step.

Question: {question}

Provide your chain of thought below:
""",

    # This is the most critical change. It uses a "few-shot" approach.
    # 1. It's entirely in one language (English instructions, but the task is specified for Hindi).
    # 2. It gives a clear example of the desired input/output format.
    # 3. It ends with the label "सोच की प्रक्रिया:" to guide the model to start writing in Hindi immediately.
    "hindi_cot_few_shot": """You are an expert math assistant who thinks and responds in Hindi.

Here is an example of how to solve a problem:
---
Question: A fruit seller had 20 apples. He sold 5. How many are left?
सोच की प्रक्रिया:
1. फल विक्रेता के पास शुरुआत में 20 सेब थे।
2. उसने 5 सेब बेच दिए।
3. बचे हुए सेबों की संख्या जानने के लिए, हमें 20 में से 5 घटाना होगा।
4. 20 - 5 = 15.
5. तो, उसके पास 15 सेब बचे हैं।
---

Now, solve the following problem. Provide the step-by-step thinking process entirely in Hindi.

Question: {question}

सोच की प्रक्रिया:
""",

    # Marathi prompt with few-shot example
    "marathi_cot_few_shot": """You are an expert math assistant who thinks and responds in Marathi.

Here is an example of how to solve a problem:
---
Question: A fruit seller had 20 apples. He sold 5. How many are left?
विचार प्रक्रिया:
1. फळ विक्रेत्याकडे सुरुवातीला 20 सफरचंद होते।
2. त्याने 5 सफरचंद विकली।
3. उरलेल्या सफरचंदांची संख्या शोधण्यासाठी, आपल्याला 20 मधून 5 वजा करावे लागतील।
4. 20 - 5 = 15.
5. तर, त्याच्याकडे 15 सफरचंद शिल्लक आहेत।
---

Now, solve the following problem. Provide the step-by-step thinking process entirely in Marathi.

Question: {question}

विचार प्रक्रिया:
""",

    # Spanish prompt with few-shot example
    "spanish_cot_few_shot": """You are an expert math assistant who thinks and responds in Spanish.

Here is an example of how to solve a problem:
---
Question: A fruit seller had 20 apples. He sold 5. How many are left?
Proceso de pensamiento:
1. El vendedor de frutas tenía inicialmente 20 manzanas.
2. Vendió 5 manzanas.
3. Para encontrar cuántas manzanas quedan, necesitamos restar 5 de 20.
4. 20 - 5 = 15.
5. Entonces, le quedan 15 manzanas.
---

Now, solve the following problem. Provide the step-by-step thinking process entirely in Spanish.

Question: {question}

Proceso de pensamiento:
""",

    # German prompt with few-shot example
    "german_cot_few_shot": """You are an expert math assistant who thinks and responds in German.

Here is an example of how to solve a problem:
---
Question: A fruit seller had 20 apples. He sold 5. How many are left?
Denkprozess:
1. Der Obsthändler hatte anfangs 20 Äpfel.
2. Er verkaufte 5 Äpfel.
3. Um herauszufinden, wie viele Äpfel übrig sind, müssen wir 5 von 20 abziehen.
4. 20 - 5 = 15.
5. Also bleiben ihm 15 Äpfel übrig.
---

Now, solve the following problem. Provide the step-by-step thinking process entirely in German.

Question: {question}

Denkprozess:
""",

    # This prompt is more specific than "convert". It clearly asks to "translate"
    # and provides structure with labels.
    "translate_hindi_to_english": """Translate the following Hindi text, which represents a step-by-step thought process, into clear and concise English.

Hindi Chain of Thought:
{hindi_cot}

English Chain of Thought:
""",

    "translate_marathi_to_english": """Translate the following Marathi text, which represents a step-by-step thought process, into clear and concise English.

Marathi Chain of Thought:
{marathi_cot}

English Chain of Thought:
""",

    "translate_spanish_to_english": """Translate the following Spanish text, which represents a step-by-step thought process, into clear and concise English.

Spanish Chain of Thought:
{spanish_cot}

English Chain of Thought:
""",

    "translate_german_to_english": """Translate the following German text, which represents a step-by-step thought process, into clear and concise English.

German Chain of Thought:
{german_cot}

English Chain of Thought:
""",

    "sanskrit_cot_few_shot": """You are an expert math assistant who thinks and responds in Sanskrit.

Here is an example of how to solve a problem:
---
Question: A fruit seller had 20 apples. He sold 5. How many are left?
चिन्तनप्रक्रिया:
1. फलविक्रेतुः आरम्भे विंशतिः सेवफलानि आसन्।
2. सः पञ्च सेवफलानि विक्रीतवान्।
3. अवशिष्टानि सेवफलानि ज्ञातुं वयं विंशतेः पञ्च ऊनं करणीयम्।
4. 20 - 5 = 15.
5. अतः तस्य पञ्चदश सेवफलानि अवशिष्टानि।
---

Now, solve the following problem. Provide the step-by-step thinking process entirely in Sanskrit.

Question: {question}

चिन्तनप्रक्रिया:
""",

    "translate_sanskrit_to_english": """Translate the following Sanskrit text, which represents a step-by-step thought process, into clear and concise English.

Sanskrit Chain of Thought:
{sanskrit_cot}

English Chain of Thought:
"""
}


def evaluate_models_on_gsm8k_enhanced():
    """
    For each model in LLM_LIST, run the prompts on GSM8K questions, compute BERT Score,
    and print the model with the best mean BERT Score. Store all results in existing CSV files.
    """
    questions = download_gsm8k_last_n_questions(20)
    summary_rows = []
    model_scores = {}  # Store mean BERT score per model

    for model in LLM_LIST:
        print(f"\n{'='*20} Processing Model: {model.upper()} {'='*20}")
        per_question_rows = []

        for i, q in enumerate(questions):
            question = q['question']
            print(f"\n--- Model: {model} | Question {i+1}/{len(questions)}: {question} ---")

            # 1. English COT (Clear, role-based prompt)
            prompt_en = PROMPT_TEMPLATES["english_cot"].format(question=question)
            en_cot, _, _ = generate_llm_response(model, prompt_en)
            print(f"[{model} Q{i+1}] Generated English CoT:\n{en_cot}")

            # 2. Hindi COT (Robust, few-shot prompt)
            prompt_hi = PROMPT_TEMPLATES["hindi_cot_few_shot"].format(question=question)
            hi_cot, _, _ = generate_llm_response(model, prompt_hi)
            print(f"[{model} Q{i+1}] Generated Hindi CoT:\n{hi_cot}")

            # 3. Marathi COT
            prompt_ma = PROMPT_TEMPLATES["marathi_cot_few_shot"].format(question=question)
            ma_cot, _, _ = generate_llm_response(model, prompt_ma)
            print(f"[{model} Q{i+1}] Generated Marathi CoT:\n{ma_cot}")

            # 4. Spanish COT
            prompt_sp = PROMPT_TEMPLATES["spanish_cot_few_shot"].format(question=question)
            sp_cot, _, _ = generate_llm_response(model, prompt_sp)
            print(f"[{model} Q{i+1}] Generated Spanish CoT:\n{sp_cot}")

            # 5. German COT
            prompt_ge = PROMPT_TEMPLATES["german_cot_few_shot"].format(question=question)
            ge_cot, _, _ = generate_llm_response(model, prompt_ge)
            print(f"[{model} Q{i+1}] Generated German CoT:\n{ge_cot}")

            # 6. Sanskrit COT
            prompt_sa = PROMPT_TEMPLATES["sanskrit_cot_few_shot"].format(question=question)
            sa_cot, _, _ = generate_llm_response(model, prompt_sa)
            print(f"[{model} Q{i+1}] Generated Sanskrit CoT:\n{sa_cot}")

            # Convert all languages to English
            hi2en_cot = ""
            ma2en_cot = ""
            sp2en_cot = ""
            ge2en_cot = ""
            sa2en_cot = ""

            if hi_cot and hi_cot.strip():
                prompt_hi2en = PROMPT_TEMPLATES["translate_hindi_to_english"].format(hindi_cot=hi_cot)
                hi2en_cot, _, _ = generate_llm_response(model, prompt_hi2en)
                print(f"[{model} Q{i+1}] Generated Hindi-to-English CoT:\n{hi2en_cot}")

            if ma_cot and ma_cot.strip():
                prompt_ma2en = PROMPT_TEMPLATES["translate_marathi_to_english"].format(marathi_cot=ma_cot)
                ma2en_cot, _, _ = generate_llm_response(model, prompt_ma2en)
                print(f"[{model} Q{i+1}] Generated Marathi-to-English CoT:\n{ma2en_cot}")

            if sp_cot and sp_cot.strip():
                prompt_sp2en = PROMPT_TEMPLATES["translate_spanish_to_english"].format(spanish_cot=sp_cot)
                sp2en_cot, _, _ = generate_llm_response(model, prompt_sp2en)
                print(f"[{model} Q{i+1}] Generated Spanish-to-English CoT:\n{sp2en_cot}")

            if ge_cot and ge_cot.strip():
                prompt_ge2en = PROMPT_TEMPLATES["translate_german_to_english"].format(german_cot=ge_cot)
                ge2en_cot, _, _ = generate_llm_response(model, prompt_ge2en)
                print(f"[{model} Q{i+1}] Generated German-to-English CoT:\n{ge2en_cot}")

            if sa_cot and sa_cot.strip():
                prompt_sa2en = PROMPT_TEMPLATES["translate_sanskrit_to_english"].format(sanskrit_cot=sa_cot)
                sa2en_cot, _, _ = generate_llm_response(model, prompt_sa2en)
                print(f"[{model} Q{i+1}] Generated Sanskrit-to-English CoT:\n{sa2en_cot}")
            
            # Word counts for all languages
            en_cot_word_count = len(str(en_cot).split()) if en_cot else 0
            hi_cot_word_count = len(str(hi_cot).split()) if hi_cot else 0
            ma_cot_word_count = len(str(ma_cot).split()) if ma_cot else 0
            sp_cot_word_count = len(str(sp_cot).split()) if sp_cot else 0
            ge_cot_word_count = len(str(ge_cot).split()) if ge_cot else 0
            sa_cot_word_count = len(str(sa_cot).split()) if sa_cot else 0
            hi2en_cot_word_count = len(str(hi2en_cot).split()) if hi2en_cot else 0
            ma2en_cot_word_count = len(str(ma2en_cot).split()) if ma2en_cot else 0
            sp2en_cot_word_count = len(str(sp2en_cot).split()) if sp2en_cot else 0
            ge2en_cot_word_count = len(str(ge2en_cot).split()) if ge2en_cot else 0
            sa2en_cot_word_count = len(str(sa2en_cot).split()) if sa2en_cot else 0
            
            per_question_rows.append({
                'model': model,
                'question': question,
                'english_cot': en_cot,
                'english_cot_word_count': en_cot_word_count,
                'hindi_cot': hi_cot,
                'hindi_cot_word_count': hi_cot_word_count,
                'marathi_cot': ma_cot,
                'marathi_cot_word_count': ma_cot_word_count,
                'spanish_cot': sp_cot,
                'spanish_cot_word_count': sp_cot_word_count,
                'german_cot': ge_cot,
                'german_cot_word_count': ge_cot_word_count,
                'sanskrit_cot': sa_cot,
                'sanskrit_cot_word_count': sa_cot_word_count,
                'hindi_to_english_cot': hi2en_cot,
                'hindi_to_english_cot_word_count': hi2en_cot_word_count,
                'marathi_to_english_cot': ma2en_cot,
                'marathi_to_english_cot_word_count': ma2en_cot_word_count,
                'spanish_to_english_cot': sp2en_cot,
                'spanish_to_english_cot_word_count': sp2en_cot_word_count,
                'german_to_english_cot': ge2en_cot,
                'german_to_english_cot_word_count': ge2en_cot_word_count,
                'sanskrit_to_english_cot': sa2en_cot,
                'sanskrit_to_english_cot_word_count': sa2en_cot_word_count
            })

        # Compute BERT Scores for all languages
        bert_scores = {}
        english_cots = [row['english_cot'] for row in per_question_rows]
        hindi_translations = [row['hindi_to_english_cot'] for row in per_question_rows]
        marathi_translations = [row['marathi_to_english_cot'] for row in per_question_rows]
        spanish_translations = [row['spanish_to_english_cot'] for row in per_question_rows]
        german_translations = [row['german_to_english_cot'] for row in per_question_rows]
        sanskrit_translations = [row['sanskrit_to_english_cot'] for row in per_question_rows]

        if english_cots and hindi_translations:
            P, R, F1 = score(hindi_translations, english_cots, lang='en', verbose=False)
            for idx, row in enumerate(per_question_rows):
                row['hindi_bert_score'] = F1[idx].item()
            bert_scores['hindi'] = F1.mean().item()
        if english_cots and marathi_translations:
            P, R, F1 = score(marathi_translations, english_cots, lang='en', verbose=False)
            for idx, row in enumerate(per_question_rows):
                row['marathi_bert_score'] = F1[idx].item()
            bert_scores['marathi'] = F1.mean().item()
        if english_cots and spanish_translations:
            P, R, F1 = score(spanish_translations, english_cots, lang='en', verbose=False)
            for idx, row in enumerate(per_question_rows):
                row['spanish_bert_score'] = F1[idx].item()
            bert_scores['spanish'] = F1.mean().item()
        if english_cots and german_translations:
            P, R, F1 = score(german_translations, english_cots, lang='en', verbose=False)
            for idx, row in enumerate(per_question_rows):
                row['german_bert_score'] = F1[idx].item()
            bert_scores['german'] = F1.mean().item()
        if english_cots and sanskrit_translations:
            P, R, F1 = score(sanskrit_translations, english_cots, lang='en', verbose=False)
            for idx, row in enumerate(per_question_rows):
                row['sanskrit_bert_score'] = F1[idx].item()
            bert_scores['sanskrit'] = F1.mean().item()

        if per_question_rows:
            df = pd.DataFrame(per_question_rows)
            model_csv_path = f'llm_gsm8k_cot_results_full_{model}_enhanced.csv'
            df.to_csv(model_csv_path, index=False)
            print(f'Created new file and saved {model} results to {model_csv_path}')

        # Save summary row for this model
        for lang, score_val in bert_scores.items():
            summary_rows.append({'model': model, 'language': lang, 'mean_bert_score': score_val})
            model_scores[f"{model}_{lang}"] = score_val

        # Clear per-model results to save RAM
        del per_question_rows
        print(f"[LOCK] Finished processing {model}. RAM cleared for next model.\n")

    # --- Final Reporting ---
    print("\n==================== FINAL RESULTS ====================")
    print(model_scores)
    
    # Save summary to existing CSV (append mode)
    summary_df = pd.DataFrame(summary_rows)
    summary_path = 'llm_gsm8k_cot_summary_enhanced.csv'
    
    if os.path.exists(summary_path):
        summary_df.to_csv(summary_path, mode='a', header=False, index=False)
        print(f'Appended summary to {summary_path}')
    else:
        summary_df.to_csv(summary_path, index=False)
        print(f'Created new summary file: {summary_path}')

    # Find best model for each language
    if model_scores:
        for lang in ['hindi', 'marathi', 'spanish', 'german', 'sanskrit']:
            lang_scores = {k: v for k, v in model_scores.items() if k.endswith(f'_{lang}')}
            if lang_scores:
                best_model = max(lang_scores, key=lambda x: float(lang_scores[x]))
                print(f"\n🏆 Best model for {lang}: {best_model} with Mean BERT Score: {lang_scores[best_model]:.4f}")
    else:
        print("\nNo models were successfully evaluated.")


# To run the function:
if __name__ == '__main__':
    # Make sure you have the required libraries installed:
    # pip install pandas bert-score torch transformers
    evaluate_models_on_gsm8k_enhanced()