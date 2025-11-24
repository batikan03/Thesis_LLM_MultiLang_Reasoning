import pandas as pd
import re
import os

# Set your directory here
directory = r"confidence_results\new_results_200d\confidence_results_RQ1point2_newresults\qwq_latest"

# --- Combined and Refined Regex Patterns ---

# Expanded answer keywords for explicit answer declarations (e.g., "Answer: X", "Svar: X")
# Includes both the original list and "Réponse est", "Doğru cevap", "Poprawna odpowiedź"
answer_keywords_combined = [
    "Svar",              # Danish
    "Answer",            # English
    "Antwort",           # German
    "Respuesta",         # Spanish
    "Réponse", "Réponse est", # French
    "Cevap", "Doğru cevap",  # Turkish
    "Odpowiedź", "Poprawna odpowiedź" # Polish
]

# Pattern for explicit answer declarations (e.g., "Answer: X", "Réponse est: X")
# This pattern is made more robust to handle optional punctuation and formatting.
explicit_answer_pattern = rf'(?:{"|".join(answer_keywords_combined)})\s*[:：]?\s*[\*#\-]?\s*(\d+)\s*\.?'

# Final conclusion phrases (e.g., "The correct answer is X", "Die korrekte Antwort lautet X")
# This list is a combination of both scripts' phrases.
final_conclusion_phrases_combined = [
    r'The\s+correct\s+answer\s+is',
    r'Die\s+korrekte\s+Antwort\s+lautet',
    r'Poprawna\s+odpowiedź\s+to',
    r'Réponse\s+est', # This was also in answer_keywords_combined, but specifically targeted here for conclusions
    r'Doğru\s+cevap(?:\s+nedir)?',
    r'Det\s+rigtige\s+svar\s+er',
    # Added from the second script's `final_conclusion_pattern` logic for broader match
    r'(?:The\s+correct\s+|Die\s+korrekte\s+)?(?:Antwort(?:option)?|answer|solution)(?:\s+is|\s+lautet|\s+ist)?'
]

# Pattern for more specific conclusive statements, now more comprehensive
final_conclusion_pattern = rf'(?:{"|".join(final_conclusion_phrases_combined)})\s*[:：]?\s*[\*#\-]?\s*(\d+)\s*\.?'

# Pattern for "Option X" or "Choice X" in English and German, now also including "Antwortoption"
option_choice_pattern = r'(?:[Oo]ption|[Cc]hoice|Antwortoption)\s*[:：]?\s*[\*#\-]?\s*(\d+)\s*\.?'

def extract_final_answer(response):
    """
    Extracts the final numerical answer from a given text response using a prioritized
    set of regex patterns.
    """
    if pd.isna(response):
        return None

    # Ensure response is a string and clean up leading/trailing whitespace
    response = str(response).strip()

    # 1. Look for LaTeX-style boxed answers, e.g., \boxed{3}
    match = re.search(r'\\boxed\{(\d+)\}', response)
    if match:
        return int(match.group(1))

    # 2. Look for explicit final conclusion statements (e.g., "The correct answer is: 2", "Poprawna odpowiedź to: 3")
    # This takes precedence over general explicit answer patterns due to its specificity.
    match = re.search(final_conclusion_pattern, response, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # 3. Look for explicit answer keyword patterns (e.g., "Answer: 2", "Réponse est: 4")
    match = re.search(explicit_answer_pattern, response, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # 4. Look for "Option X", "Choice X", "Antwortoption X"
    match = re.search(option_choice_pattern, response, re.IGNORECASE)
    if match:
        return int(match.group(1))

    # 5. Fallback: Last numeric-only line.
    # This iteration goes through lines in reverse, prioritizing the last short numeric line.
    lines = response.strip().splitlines()
    for line in reversed(lines):
        line = line.strip()
        # Clean the line by removing trailing non-digit characters that might be part of a number (like "4.")
        cleaned_line = re.sub(r'[^\d]+$', '', line)
        
        # Check if the cleaned line is *only* digits and is short (heuristic for an answer number)
        # This is an important heuristic to avoid matching random numbers in the text.
        if re.fullmatch(r'(\d+)', cleaned_line) and len(cleaned_line) <= 2:
            return int(cleaned_line)

    return None

# --- Pipeline Execution ---
print(f"Starting answer extraction for directory: {directory}")

for filename in os.listdir(directory):
    if filename.endswith(".csv"):
        filepath = os.path.join(directory, filename)
        print(f"\nProcessing: {filepath}")

        try:
            df = pd.read_csv(filepath)

            if 'response' in df.columns:
                # Apply the extraction function to the 'response' column
                df['evaluation'] = df['response'].apply(extract_final_answer)
                
                # Overwrite the original CSV file with the updated DataFrame
                df.to_csv(filepath, index=False)
                print(f"Successfully updated: {filename} with 'evaluation' column.")
            else:
                print(f"Skipped {filename}: No 'response' column found.")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("\nAnswer extraction pipeline completed.")