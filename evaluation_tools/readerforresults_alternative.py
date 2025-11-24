import pandas as pd
import re
import os
from typing import Optional, List

# --- Point this to the *RQ directory* that contains multiple model subfolders ---
RQ_ROOT = r"confidence_results\new_results_200d\experiment_results\confidence_results_RQ3_newresults"

print(f"Starting recursive answer extraction under: {RQ_ROOT}")

# ----------------- Patterns (keep your order; only broaden coverage) -----------------

answer_keywords_combined = [
    # original
    "Svar", "Answer", "Antwort", "Respuesta", "Réponse", "Réponse est",
    "Cevap", "Doğru cevap", "Odpowiedź", "Poprawna odpowiedź",
    # added (BS/HR/SR/BG/SQ/MT)
    "Odgovor", "Tačan odgovor", "Točan odgovor",
    "Отговор", "Верен отговор", "Правилен отговор", "Правилният отговор",
    "Përgjigjja", "Përgjigja", "Përgjigjja e saktë", "Përgjigja e saktë",
    "Përgjigjja e duhur", "Përgjigja e duhur",
    "Tweġiba", "It-tweġiba korretta", "It-tweġiba t-tajba"
]
explicit_answer_pattern = re.compile(
    rf'(?:{"|".join(answer_keywords_combined)})\s*[:：=\-]?\s*[\(\[\{{]?\s*(\d+)\s*[\)\]\}}]?\s*\.?',
    re.IGNORECASE | re.UNICODE
)

final_conclusion_phrases_combined = [
    # original
    r'The\s+correct\s+answer\s+is',
    r'Die\s+korrekte\s+Antwort\s+lautet',
    r'Poprawna\s+odpowiedź\s+to',
    r'Réponse\s+est',
    r'Doğru\s+cevap(?:\s+nedir)?',
    r'Det\s+rigtige\s+svar\s+er',
    r'(?:The\s+correct\s+|Die\s+korrekte\s+)?(?:Antwort(?:option)?|answer|solution)(?:\s+is|\s+lautet|\s+ist)?',
    # added
    r'Tačan\s+odgovor\s+je', r'Točan\s+odgovor\s+je', r'Tačan\s+je\s+odgovor', r'Točan\s+je\s+odgovor',
    r'Верният\s+отговор\s+е', r'Правилният\s+отговор\s+е',
    r'Përgjigjja\s+e\s+saktë\s+është', r'Përgjigja\s+e\saktë\s+është',
    r'Përgjigjja\s+e\s+duhur\s+është', r'Përgjigja\s+e\s+duhur\s+është',
    r'It\-tweġiba\s+korretta\s+hi', r'It\-tweġiba\s+t\-tajba\s+hi', r'It\-tweġiba\s+t\-tajba\s+hija'
]
final_conclusion_pattern = re.compile(
    rf'(?:{"|".join(final_conclusion_phrases_combined)})\s*[:：=\-]?\s*[\(\[\{{]?\s*(\d+)\s*[\)\]\}}]?\s*\.?',
    re.IGNORECASE | re.UNICODE
)

option_choice_words = [
    r'[Oo]ption', r'[Cc]hoice', r'Antwortoption',
    r'[Oo]pcija', r'[Ii]zbor',
    r'Опция',
    r'[Zz]gjedhja',
    r'[Gg]ħażla'
]
option_choice_pattern = re.compile(
    rf'(?:{"|".join(option_choice_words)})\s*[:：=\-]?\s*[\(\[\{{]?\s*(\d+)\s*[\)\]\}}]?\s*\.?',
    re.IGNORECASE | re.UNICODE
)

boxed_pattern = re.compile(r'\\boxed\{(\d+)\}')

# NEW: strong “end-of-text number after colon/equals”
end_colon_number_pattern = re.compile(r'[:=]\s*(\d{1,2})\s*[\)\].;,:-]?\s*$', re.IGNORECASE | re.MULTILINE)
# NEW: permissive “last number at end”
end_loose_number_pattern  = re.compile(r'(\d{1,2})\s*[\)\].;,:-]?\s*$', re.IGNORECASE | re.MULTILINE)

line_tail_numeric = re.compile(r'^\s*[\(\[]?\s*(\d{1,2})\s*[\)\]]?\s*[\.\-–—,:;]?\s*$')

def extract_final_answer(response: Optional[str]) -> Optional[int]:
    if pd.isna(response):
        return None
    response = str(response).strip()
    if not response:
        return None

    # 1) LaTeX boxed
    m = boxed_pattern.search(response)
    if m:
        return int(m.group(1))

    # 2) Final conclusion
    m = final_conclusion_pattern.search(response)
    if m:
        return int(m.group(1))

    # 3) "Answer:"-style
    m = explicit_answer_pattern.search(response)
    if m:
        return int(m.group(1))

    # 4) Option/Choice
    m = option_choice_pattern.search(response)
    if m:
        return int(m.group(1))

    # 5) NEW: number after ":" or "=" near the end (e.g., "Odgovor je :2", "Përgjigja është: 3")
    m = end_colon_number_pattern.search(response)
    if m:
        return int(m.group(1))

    # 6) NEW: any final number at the end (1–2 digits)
    m = end_loose_number_pattern.search(response)
    if m:
        return int(m.group(1))

    # 7) Fallback: last numeric-only line (≤2 digits)
    for line in reversed(response.splitlines()):
        line = line.strip()
        m = line_tail_numeric.fullmatch(line)
        if m:
            return int(m.group(1))
    return None

# ----------------- Robust CSV reader -----------------
ENCODINGS: List[str] = ["utf-8", "utf-8-sig", "cp1250", "cp1251", "latin-1"]
SEPARATORS: List[Optional[str]] = [",", ";", "\t", None]
ENGINES: List[str] = ["python", "c"]

def robust_read_csv(path: str) -> pd.DataFrame:
    last_err = None
    for engine in ENGINES:
        for sep in SEPARATORS:
            for enc in ENCODINGS:
                try:
                    kwargs = dict(encoding=enc)
                    if engine == "python":
                        kwargs.update(dict(engine="python", on_bad_lines="skip", sep=sep))
                    else:
                        if sep is None:
                            continue
                        kwargs.update(dict(engine="c", sep=sep))
                    df = pd.read_csv(path, **kwargs)
                    if not len(df.columns):
                        raise ValueError("No columns parsed.")
                    # Normalize possible BOM/spacing in headers
                    df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]
                    return df
                except Exception as e:
                    last_err = e
                    continue
    raise RuntimeError(f"Failed to read CSV robustly: {path}\nLast error: {last_err}")

def find_response_column(df: pd.DataFrame) -> Optional[str]:
    # exact case-insensitive match first
    lower_map = {c.lower(): c for c in df.columns}
    for key in ("response", "model_response", "output"):
        if key in lower_map:
            return lower_map[key]
    # then partial contains "response"
    cands = [c for c in df.columns if "response" in c.lower()]
    return cands[0] if cands else None

processed, updated, skipped, errors = 0, 0, 0, []

for root, _, files in os.walk(RQ_ROOT):
    csvs = [f for f in files if f.lower().endswith(".csv")]
    for filename in csvs:
        path = os.path.join(root, filename)
        processed += 1
        print(f"\nProcessing: {path}")
        try:
            df = robust_read_csv(path)
            resp_col = find_response_column(df)
            if not resp_col:
                skipped += 1
                print(f"⤴ Skipped (no 'response' column): {filename}")
                continue

            df["evaluation"] = df[resp_col].apply(extract_final_answer)
            df.to_csv(path, index=False)
            updated += 1
            print(f"✔ Updated: {filename} (evaluation written)")
        except Exception as e:
            errors.append(f"{path} -> {e}")
            print(f"✖ Error: {e}")

print("\n=== Summary ===")
print(f"Processed CSVs: {processed}")
print(f"Updated: {updated}")
print(f"Skipped (no response col): {skipped}")
if errors:
    print("Errors:")
    for e in errors:
        print("  -", e)
print("Done.")
