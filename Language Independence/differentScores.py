import os
import pandas as pd
from tqdm import tqdm
from alignscore import AlignScore
from rouge_score import rouge_scorer

# Set up AlignScore
ALIGN_CKPT_PATH = '/workspace/amit/Shashank/COT/Language Independence/AlignScore-large.ckpt'
align_scorer = AlignScore(model='roberta-base', batch_size=32, device=0, ckpt_path=ALIGN_CKPT_PATH, evaluation_mode='nli_sp')

# Helper to compute ROUGE-L F1
rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
def rouge_l_f1(ref, cand):
    try:
        return rouge.score(ref, cand)['rougeL'].fmeasure
    except Exception:
        return 0.0

# Directory containing the model CSVs
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'All Language Results')

# List of languages to check
LANGS = ['hindi', 'marathi', 'spanish', 'german', 'sanskrit']

for fname in os.listdir(RESULTS_DIR):
    if not fname.endswith('.csv'):
        continue
    csv_path = os.path.join(RESULTS_DIR, fname)
    print(f'Processing {csv_path}')
    df = pd.read_csv(csv_path)
    updated = False
    for lang in LANGS:
        ref_col = 'english_cot'
        cand_col = f'{lang}_to_english_cot'
        align_col = f'english_{lang}_align_score'
        rouge_col = f'english_{lang}_rouge_score'
        if ref_col in df.columns and cand_col in df.columns:
            refs = df[ref_col].astype(str).fillna("").tolist()
            cands = df[cand_col].astype(str).fillna("").tolist()
            # Filter out pairs where either is too short
            filtered = [(r, c, i) for i, (r, c) in enumerate(zip(refs, cands)) if len(r.strip()) > 10 and len(c.strip()) > 10]
            if not filtered:
                print(f"No valid pairs for {lang} in {fname}")
                continue
            idxs = [i for _, _, i in filtered]
            refs_filt = [r for r, _, _ in filtered]
            cands_filt = [c for _, c, _ in filtered]
            # Compute AlignScore
            try:
                align_scores = align_scorer.score(contexts=refs_filt, claims=cands_filt)
            except Exception as e:
                print(f"AlignScore error for {lang} in {fname}: {e}")
                align_scores = [0.0] * len(refs_filt)
            # Compute ROUGE-L
            rouge_scores = [rouge_l_f1(r, c) for r, c in tqdm(zip(refs_filt, cands_filt), total=len(refs_filt), desc=f'ROUGE-L {lang}')]            
            # Fill new columns with NaN, then assign computed values
            df[align_col] = float('nan')
            df[rouge_col] = float('nan')
            for idx, a, r in zip(idxs, align_scores, rouge_scores):
                df.at[idx, align_col] = a
                df.at[idx, rouge_col] = r
            updated = True
    if updated:
        df.to_csv(csv_path, index=False)
        print(f'Updated: {csv_path}')
    else:
        print(f'No updates made to {csv_path}')