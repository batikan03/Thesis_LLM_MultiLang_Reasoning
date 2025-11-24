import os
import re
import pandas as pd
import matplotlib.pyplot as plt

# ======================
# CONFIG
# ======================
ROOT_DIR = r"confidence_results\new_results_200d\confidence_results_RQ1point2_newresults"  # input folder
OUTPUT_DIR = r"instability_charts\RQ1point2"  # where to save plots
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================
# Heuristics
# ======================
# Regex for Chinese characters
chinese_char_pattern = re.compile(
    r'[\u4E00-\u9FFF\u3400-\u4DBF\U00020000-\U0002A6DF'
    r'\U0002A700-\U0002B73F\U0002B740-\U0002B81F'
    r'\U0002B820-\U0002CEAF\U0002CEB0-\U0002EBEF'
    r'\U00030000-\U0003134F]'
)

def detect_hallucination(text: str) -> bool:
    if pd.isna(text):
        return False
    # Chinese: any occurrence
    if chinese_char_pattern.search(text):
        return True
    # Cyrillic: 5+ contiguous
    #if re.search(r"[\u0400-\u04FF]{5,}", text):
        return True
    return False

def detect_indecision(text: str) -> bool:
    if pd.isna(text):
        return False
    text = text.lower()
    refusal_patterns = [
        "i don't know", "i do not know",
        "i cannot answer", "can't answer",
        "unable to determine", "no idea"
    ]
    return any(p in text for p in refusal_patterns)

# ======================
# Main Loop
# ======================
for model in os.listdir(ROOT_DIR):
    model_dir = os.path.join(ROOT_DIR, model)
    if not os.path.isdir(model_dir):
        continue
    
    instability_counts = {}
    
    # Loop over language CSVs
    for file in os.listdir(model_dir):
        if not file.endswith(".csv"):
            continue
        lang = file.replace("confidence_", "").replace(".csv", "")
        # Read CSV and normalize columns
        df = pd.read_csv(os.path.join(model_dir, file))
        df.columns = [c.strip().lower() for c in df.columns]

        if "response" not in df.columns:
            print(f"⚠️ Skipping {file} (no 'response' column found)")
            continue

        hallucinations = df["response"].apply(detect_hallucination)
        indecisions = df["response"].apply(detect_indecision)

        instability = (hallucinations | indecisions).sum()
        instability_counts[lang] = instability
                
        hallucinations = df["response"].apply(detect_hallucination)
        indecisions = df["response"].apply(detect_indecision)
        
        instability = (hallucinations | indecisions).sum()
        instability_counts[lang] = instability
    
    # Create bar chart
    if instability_counts:
        langs = list(instability_counts.keys())
        counts = [instability_counts[l] for l in langs]
        
        plt.figure(figsize=(8, 5))
        plt.bar(langs, counts, color="steelblue")
        plt.title(f"Instability Counts per Language – {model}")
        plt.xlabel("Language")
        plt.ylabel("Instability Count")
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        out_path = os.path.join(OUTPUT_DIR, f"{model}_instability.png")
        plt.savefig(out_path)
        plt.close()

print(f"✅ Instability charts saved to {OUTPUT_DIR}")
