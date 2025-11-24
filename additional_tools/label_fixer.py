import os
import re
import pandas as pd

# Path to the folder with CSV files
folder_path = r'COVID_RESULTS\prompt_variation_results\qwen2.5_7b'

# Multilingual equivalents of True/False
true_false_map = {
    'true': ['true', 'wahr', 'vrai', 'verdadero', 'prawda', 'sandt', 'doğru', 'правда'],
    'false': ['false', 'falsch', 'faux', 'falso', 'fałsz', 'falsk', 'yanlış', 'неправда']
}

# Compile regex pattern
pattern = re.compile(r'\b(' + '|'.join(true_false_map['true'] + true_false_map['false']) + r')\b', flags=re.IGNORECASE)

# Function to normalize multilingual response
def extract_final_decision(text):
    if not isinstance(text, str):
        return None

    matches = pattern.findall(text)
    if not matches:
        return None

    last_match = matches[-1].lower()

    # Determine whether it's a True or False value
    if last_match in map(str.lower, true_false_map['true']):
        return 'True'
    elif last_match in map(str.lower, true_false_map['false']):
        return 'False'
    return None

# Process all CSVs in the folder
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        file_path = os.path.join(folder_path, filename)
        print(f"Processing: {file_path}")

        try:
            df = pd.read_csv(file_path)

            # Make sure 'response' column exists
            if 'response' in df.columns:
                df['evaluation'] = df['response'].apply(extract_final_decision)
                df.to_csv(file_path, index=False)
                print(f"✔ Updated: {filename}")
            else:
                print(f"⚠ 'response' column not found in: {filename}")

        except Exception as e:
            print(f"❌ Error processing {filename}: {e}")
