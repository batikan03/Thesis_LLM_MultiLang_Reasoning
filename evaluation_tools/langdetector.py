import os
import pandas as pd
from langdetect import detect_langs, DetectorFactory, LangDetectException
import re
from collections import Counter

# Set a seed for reproducibility in language detection
DetectorFactory.seed = 0

# Regex pattern to identify Chinese characters(ı.e bleedout)
chinese_char_pattern = re.compile(
    r'[\u4E00-\u9FFF\u3400-\u4DBF\U00020000-\U0002A6DF'
    r'\U0002A700-\U0002B73F\U0002B740-\U0002B81F'
    r'\U0002B820-\U0002CEAF\U0002CEB0-\U0002EBEF'
    r'\U00030000-\U0003134F]'
)

def get_language_label(paragraph):
    """
    Detects the language(s) of a given paragraph and returns a descriptive label.
    Handles single dominant languages, mixes, and presence of Chinese characters.
    """
    if not isinstance(paragraph, str) or not paragraph.strip():
        return 'unknown'

    try:
        detections = detect_langs(paragraph)
        langs_probs = [(lang.lang, lang.prob) for lang in detections]
    except LangDetectException:
        # If no language can be reliably detected
        # print(paragraph)
        return 'unknown'
    except Exception as e:
        # Catch other potential errors during detection
        print(f"Error detecting language in: '{paragraph[:50]}...' — {e}")
        return 'error'

    # Normalize probabilities to sum to 1
    total_prob = sum(prob for _, prob in langs_probs)
    if total_prob == 0: # Avoid division by zero if all probabilities are 0 (unlikely but good safeguard)
        return 'unknown'
    langs_probs = [(lang, prob / total_prob) for lang, prob in langs_probs]

    primary_lang, primary_prob = langs_probs[0]
    others = langs_probs[1:]

    has_chinese = bool(chinese_char_pattern.search(paragraph))
    label = "" # Initialize label

    # Determine the language label based on probability thresholds
    if primary_prob >= 0.85:
        label = primary_lang
    elif primary_prob >= 0.75:
        fragments = [f"{lang}" for lang, prob in others if prob >= 0.05]
        label = f"{primary_lang} with {' and '.join(fragments)} fragments" if fragments else primary_lang
    elif primary_prob >= 0.5:
        fragments = [f"{lang}" for lang, prob in others if prob >= 0.05]
        label = f"{primary_lang} and {' and '.join(fragments)} mix" if fragments else f"{primary_lang} mix"
        #print(paragraph)
    else:
        # For more balanced or less dominant mixes
        mix_langs = [lang for lang, prob in langs_probs if prob >= 0.15]
        label = ' and '.join(mix_langs) + " mix" if mix_langs else 'mixed_no_dominant'

    # Append "this string" if the alpaheth is weird
    if has_chinese:
        label += " with foreign/irregular characters"

    return label

def analyze_dataset_languages(filepath, response_column='response'):
    """
    Loads a CSV, applies language detection to the specified response column,
    and returns a summary of language label frequencies and the updated DataFrame.
    """
    print(f"\n--- Analyzing file: {os.path.basename(filepath)} ---")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None, None
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return None, None

    if response_column not in df.columns:
        print(f"Skipping {os.path.basename(filepath)}: Column '{response_column}' not found.")
        return None, None

    print("Starting detailed language analysis...")
    df['language_label'] = df[response_column].apply(get_language_label)
    print("Language analysis complete.")

    language_summary = df['language_label'].value_counts()
    return language_summary, df

# --- Main execution ---
if __name__ == "__main__":
    # Directory containing your CSV files
    directory_path = r"confidence_results\raw_results\confidence_results_RQ1\deepseek-r1_14b-qwen-distill-q8_0" # Changed to original directory

    # Dictionary to store aggregated results from all files
    all_language_summaries = Counter()
    processed_files_count = 0

    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' not found.")
    else:
        for filename in os.listdir(directory_path):
            if filename.endswith(".csv"):
                filepath = os.path.join(directory_path, filename)
                summary, updated_df = analyze_dataset_languages(filepath, response_column='response')

                if summary is not None:
                    print("\n--- Aggregated Language Labels for Current File ---")
                    print(summary)
                    all_language_summaries.update(summary) # Aggregate counts
                    processed_files_count += 1

                    # Optionally, save the updated DataFrame for each file
                    # updated_df.to_csv(f'./processed_datasets/{filename.replace(".csv", "_with_lang.csv")}', index=False)

    print("\n" + "="*50)
    print(f"=== Overall Language Analysis Across {processed_files_count} Processed Files ===")
    if all_language_summaries:
        for label, count in all_language_summaries.most_common():
            print(f"- {label}: {count}")
    else:
        print("No valid CSV files were processed or no language data found.")
    print("="*50)