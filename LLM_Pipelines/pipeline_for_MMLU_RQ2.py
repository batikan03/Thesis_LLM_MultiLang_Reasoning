import requests
import json
import pandas as pd
import os
from datetime import datetime
import re
import time

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

def evaluate_headline_multilang(question_text, model, language_code):
    prompt = (
        f"You are an AI assistant solving multiple-choice math or logic questions. "
        f"Read the question carefully, then choose the correct answer from the options provided. "
        f"Provide your answer by stating the number of the correct option (e.g., answer: 1, 2, 3, or 4). "
        f"You must respond in the following language: {language_code}.\n\n"
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
    input_file = "datasets/MMLU_translated_v2/processed_questions_google_translated_en.csv"
    model_names = ["qwq:latest", "qwen2.5:7b", "deepseek-r1:14b-qwen-distill-q8_0"]
    target_languages = {
    "bulgarian": "bg",
    "bosnian": "bs",
    "maltese": "mt",
    "albanian": "sq"}    
    column_to_evaluate = "input"
    output_base_dir = "/home/batikan/llm_pipeline/spesific_results/RQ2_Results_default"

    try:
        df = pd.read_csv(input_file)

        for model_name in model_names:
            for lang_name, lang_code in target_languages.items():
                print(f"\nEvaluating with model '{model_name}' in language: {lang_name}")
                output_dir = os.path.join(output_base_dir, model_name.replace(":", "_"))
                os.makedirs(output_dir, exist_ok=True)

                output_filename = f"covid_en_input_{model_name.replace(':', '_')}_{lang_code}_eval.csv"
                output_path = os.path.join(output_dir, output_filename)

                # Check if we already have partial results saved
                if os.path.exists(output_path):
                    df_results = pd.read_csv(output_path)
                    evaluated_indices = set(df_results.index[df_results["evaluation"].notna()])
                    print(f"Resuming from saved file. {len(evaluated_indices)} rows already evaluated.")
                else:
                    df_results = df.copy()
                    df_results["evaluation"] = None
                    df_results["response"] = None
                    df_results["language"] = lang_name
                    evaluated_indices = set()

                for idx, row in df.iterrows():
                    if idx in evaluated_indices:
                        continue

                    content = row[column_to_evaluate]
                    print(f"  Evaluating row {idx + 1}...")

                    evaluation, full_response = evaluate_headline_multilang(
                        content, model=model_name, language_code=lang_name
                    )

                    df_results.at[idx, "evaluation"] = evaluation
                    df_results.at[idx, "response"] = full_response
                    df_results.at[idx, "language"] = lang_name

                    # Save progress after each row
                    df_results.to_csv(output_path, index=False)

                print(f"✅ Finished evaluating with model '{model_name}' in language: {lang_name}")
                print(f"📁 Results saved to: {output_path}")

    except FileNotFoundError:
        print(f"❌ Error: File not found - {input_file}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
