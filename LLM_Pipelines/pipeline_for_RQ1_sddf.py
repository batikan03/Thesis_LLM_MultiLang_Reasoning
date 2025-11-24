import requests
import json
import pandas as pd
import os
import re
import time
from glob import glob

# ----------------------------
# Config (single input CSV)
# ----------------------------
INPUT_CSV = "datasets/MMLU_translated_v2/processed_questions_google_translated_en.csv"   # <- pick your EN file here
PROMPT_DIR = "prompts_for_MMLU"                      # check_truth_en.txt, check_truth_fr.txt, ...
OUTPUT_BASE_DIR = "/home/batikan/llm_pipeline/spesific_results/RQ1_Results"
MODEL_NAMES = ["qwq:latest", "qwen2.5:7b", "deepseek-r1:14b-qwen-distill-q8_0"]
COLUMN_TO_EVALUATE = "input"

# ----------------------------
# HTTP / LLM helpers
# ----------------------------
def query_ollama(prompt, model, retries=3, delay=10, timeout=120, session=None):
    url = "http://localhost:11434/api/generate"
    data = {"prompt": prompt, "model": model, "stream": False}
    sess = session or requests.Session()

    for attempt in range(retries):
        try:
            resp = sess.post(url, json=data, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
            return (payload.get("response") or "").strip()
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            print(f"[{model}] Attempt {attempt + 1}/{retries} failed: {e}")
            if attempt < retries - 1:
                sleep_for = delay * (1.5 ** attempt)
                print(f"Retrying in {sleep_for:.1f}s...")
                time.sleep(sleep_for)
            else:
                print("Max retries reached. Skipping this prompt.")
    return None

def evaluate_headline_multilang(question_text, model, prompt_instruction, session=None):
    prompt = f"{prompt_instruction.strip()}\n\nQuestion:\n{question_text}"
    full_response = query_ollama(prompt, model, session=session)
    if full_response is None:
        return None, None
    # Expect final line to be 1-4
    lines = full_response.strip().splitlines()
    last_line = lines[-1].strip() if lines else ""
    match = re.fullmatch(r"[1-4]", last_line)
    if match:
        return match.group(), full_response
    return None, full_response

# ----------------------------
# Utilities
# ----------------------------
LANG_FROM_PROMPT_RE = re.compile(r"check_truth_([a-z]{2})\.txt$", re.IGNORECASE)

def discover_languages_from_prompts(prompt_dir):
    mapping = {}
    for p in glob(os.path.join(prompt_dir, "check_truth_*.txt")):
        m = LANG_FROM_PROMPT_RE.search(os.path.basename(p))
        if m:
            mapping[m.group(1)] = p
    return mapping

# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # Load the single input CSV (e.g., EN)
    try:
        df = pd.read_csv(INPUT_CSV)
    except Exception as e:
        raise SystemExit(f"Failed to read INPUT_CSV '{INPUT_CSV}': {e}")

    if COLUMN_TO_EVALUATE not in df.columns:
        raise SystemExit(f"Column '{COLUMN_TO_EVALUATE}' not found in {INPUT_CSV}")

    lang_to_prompt = discover_languages_from_prompts(PROMPT_DIR)
    if not lang_to_prompt:
        raise SystemExit(f"No prompt files found in {PROMPT_DIR} (expected: check_truth_*.txt)")

    base_name = os.path.basename(INPUT_CSV).replace(".csv", "")
    session = requests.Session()

    print(f"Input rows: {len(df)} from {INPUT_CSV}")

    for lang_code, prompt_file in lang_to_prompt.items():
        # Read prompt text for this language
        try:
            with open(prompt_file, "r", encoding="utf-8") as pf:
                prompt_instruction = pf.read()
        except Exception as e:
            print(f"! Could not read prompt for '{lang_code}': {e}")
            continue

        for model_name in MODEL_NAMES:
            print(f"\n=== Language: {lang_code} | Model: {model_name} ===")
            model_safe = model_name.replace(":", "_")
            out_dir = os.path.join(OUTPUT_BASE_DIR, model_safe)
            os.makedirs(out_dir, exist_ok=True)

            # Output file: single input -> one result file per (lang, model)
            out_filename = f"{base_name}_{model_safe}_{lang_code}_eval.csv"
            out_path = os.path.join(out_dir, out_filename)

            # Resume logic
            if os.path.exists(out_path):
                try:
                    df_results = pd.read_csv(out_path)
                    for col in ("evaluation", "response", "language"):
                        if col not in df_results.columns:
                            df_results[col] = None
                    evaluated_indices = set(df_results.index[df_results["evaluation"].notna()])
                    print(f"Resuming from {out_path}: {len(evaluated_indices)} rows already evaluated.")
                except Exception as e:
                    print(f"! Failed to resume from {out_path}: {e}")
                    df_results = df.copy()
                    for col in ("evaluation", "response"):
                        if col not in df_results.columns:
                            df_results[col] = None
                    df_results["language"] = lang_code
                    evaluated_indices = set()
            else:
                df_results = df.copy()
                for col in ("evaluation", "response"):
                    if col not in df_results.columns:
                        df_results[col] = None
                df_results["language"] = lang_code
                evaluated_indices = set()

            # Evaluate each row using this language prompt
            for idx, row in df.iterrows():
                if idx in evaluated_indices:
                    continue
                content = row[COLUMN_TO_EVALUATE]
                if pd.isna(content) or str(content).strip() == "":
                    df_results.at[idx, "evaluation"] = None
                    df_results.at[idx, "response"] = None
                    df_results.at[idx, "language"] = lang_code
                    continue

                print(f"  Row {idx + 1}/{len(df)}")
                evaluation, full_response = evaluate_headline_multilang(
                    str(content),
                    model=model_name,
                    prompt_instruction=prompt_instruction,
                    session=session
                )

                df_results.at[idx, "evaluation"] = evaluation
                df_results.at[idx, "response"] = full_response
                df_results.at[idx, "language"] = lang_code

                # Save after each row for robustness
                try:
                    df_results.to_csv(out_path, index=False)
                except Exception as e:
                    print(f"  ! Failed to save {out_path}: {e}")

            print(f"✅ Saved: {out_path}")

    print("\nAll done.")
