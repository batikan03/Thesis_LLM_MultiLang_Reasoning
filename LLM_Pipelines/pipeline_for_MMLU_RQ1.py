import requests
import json
import pandas as pd
import os
from datetime import datetime
import re
import time
from glob import glob

def query_ollama(prompt, model, retries=3, delay=10):
    url = "http://localhost:11434/api/generate"
    data = {
        "prompt": prompt,
        "model": model,
        "stream": False
    }
    for attempt in range(retries):
        try:
            response = requests.post(url, json=data, timeout=120)
            response.raise_for_status()
            return response.json()['response'].strip()
        except (requests.exceptions.RequestException, json.JSONDecodeError) as e:
            print(f"Attempt {attempt + 1}/{retries} failed for model '{model}': {e}")
            if attempt < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("Max retries reached. Skipping this prompt.")
    return None

def evaluate_headline_multilang(question_text, model, prompt_instruction):
    prompt = (
        f"{prompt_instruction.strip()}\n\n"
        f"Question:\n{question_text}"
    )
    full_response = query_ollama(prompt, model)
    if full_response:
        lines = full_response.strip().splitlines()
        last_line = lines[-1].strip()
        match = re.fullmatch(r"[1-4]", last_line)
        if match:
            return match.group(), full_response
        else:
            return None, full_response
    return None, None

if __name__ == "__main__":
    input_dir = "MMLU_translated_new"
    prompt_dir = "prompts_for_MMLU"
    output_base_dir = "/home/batikan/llm_pipeline/spesific_results/RQ1_alldiff_Results"
    model_names = ["qwq:latest", "qwen2.5:7b", "deepseek-r1:14b-qwen-distill-q8_0"]
    column_to_evaluate = "input"

    csv_files = glob(os.path.join(input_dir, "*.csv"))

    for csv_file in csv_files:
        try:
            # Extract language code from filename
            lang_code_match = re.search(r"_([a-z]{2})\.csv$", csv_file)
            if not lang_code_match:
                print(f"Skipping unrecognized file format: {csv_file}")
                continue

            lang_code = lang_code_match.group(1)
            prompt_file = os.path.join(prompt_dir, f"check_truth_{lang_code}.txt")

            if not os.path.exists(prompt_file):
                print(f"Prompt file not found for language '{lang_code}': {prompt_file}")
                continue

            # Read input data and prompt
            df = pd.read_csv(csv_file)
            with open(prompt_file, "r", encoding="utf-8") as pf:
                prompt_instruction = pf.read()

            for model_name in model_names:
                print(f"\nEvaluating model '{model_name}' on language: {lang_code}")
                output_dir = os.path.join(output_base_dir, model_name.replace(":", "_"))
                os.makedirs(output_dir, exist_ok=True)

                base_filename = os.path.basename(csv_file).replace(".csv", "")
                output_filename = f"{base_filename}_{model_name.replace(':', '_')}_{lang_code}_eval.csv"
                output_path = os.path.join(output_dir, output_filename)

                if os.path.exists(output_path):
                    df_results = pd.read_csv(output_path)
                    evaluated_indices = set(df_results.index[df_results["evaluation"].notna()])
                    print(f"Resuming from saved file. {len(evaluated_indices)} rows already evaluated.")
                else:
                    df_results = df.copy()
                    df_results["evaluation"] = None
                    df_results["response"] = None
                    df_results["language"] = lang_code
                    evaluated_indices = set()

                for idx, row in df.iterrows():
                    if idx in evaluated_indices:
                        continue

                    content = row[column_to_evaluate]
                    print(f"  Evaluating row {idx + 1}...")

                    evaluation, full_response = evaluate_headline_multilang(
                        content, model=model_name, prompt_instruction=prompt_instruction
                    )

                    df_results.at[idx, "evaluation"] = evaluation
                    df_results.at[idx, "response"] = full_response
                    df_results.at[idx, "language"] = lang_code

                    df_results.to_csv(output_path, index=False)

                print(f"✅ Finished evaluating with model '{model_name}' in language: {lang_code}")
                print(f"📁 Results saved to: {output_path}")

        except Exception as e:
            print(f"❌ Error processing file {csv_file}: {e}")
