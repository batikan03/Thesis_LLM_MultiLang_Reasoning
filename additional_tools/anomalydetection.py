import os
import re
import math
import pandas as pd
from collections import Counter, defaultdict

# ========= Config: column names in your CSVs =========
RESPONSE_COL = "response"          # required
CHOICES_COL  = "choices"           # optional (A–D etc.). If absent, related checks are skipped.
ANSWER_COL   = "answer"            # optional; unused here but can be helpful later.

# ========= Helpers =========
def safe_str(x):
    return x if isinstance(x, str) else ""

def norm_per_100_tokens(text, count):
    # simple length normalization
    tokens = max(1, len(re.findall(r"\w+", text)))
    return 100.0 * count / tokens

def has_commitment_marker(text):
    return bool(re.search(
        r"(?i)\b(final answer|the answer is|correct answer is|my answer|i choose|i pick|selected answer|"
        r"cevap|antwort|réponse|respuesta)\b", text))

def extract_final_choices(text):
    """
    Extract explicit A-D choices when marked as a final commitment.
    Returns set like {'A','C'}.
    """
    finals = set()
    # explicit markers like "final answer: B", "answer is C"
    for m in re.finditer(r"(?i)(?:final answer|answer is|correct answer is|cevap|antwort|réponse|respuesta)\s*[:\-]?\s*([A-D])\b", text):
        finals.add(m.group(1).upper())
    return finals

def mentioned_choices(text):
    # any bare A-D mention (avoid false positives by word boundaries)
    return set(ch.upper() for ch in re.findall(r"\b([A-D])\b", text, flags=re.IGNORECASE))

def has_reversal_connector(text):
    return bool(re.search(
        r"(?i)\b(but|however|yet|nevertheless|still|though|actually|in fact|on second thought|"
        r"yine de|ancak|jednak|pourtant)\b", text))

# Hedging/indecision lexicon (language-agnostic + some EU cues).
HEDGE_RE = re.compile(
    r"(?i)\b("
    r"maybe|perhaps|possibly|probably|might|could|unsure|uncertain|not sure|hard to say|"
    r"i think|i guess|i believe|it depends|ambiguous|lean(?:ing)? toward|"
    r"vielleicht|unsicher|nicht sicher|könnte|wahrscheinlich|"
    r"quizás|tal vez|podría|probablemente|"
    r"peut[- ]être|il se peut|"
    r"emin değilim|belki|muhtemelen|olabilir|kararsız(ım)?|"
    r"måske|ikke sikker|usikker|"
    r"może|nie jestem pewn\w+|prawdopodobnie"
    r")\b"
)

QUESTION_RE = re.compile(r"\?\s*$")

# Fabrication / citation style
CITATION_RE = re.compile(r"(?i)\b(source|citation|references?|doi|according to (?:my|an|internal|private) (?:db|database)|"
                         r"according to wikipedia|as per (?:internal|private) data)\b")
URL_RE = re.compile(r"https?://[^\s)]+")
FAKE_TOOLING_RE = re.compile(r"(?i)\b(i (?:browsed|searched) the web|i looked it up online|"
                             r"i executed code|i ran a script|i queried an api)\b")

OPTION_LABEL_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)
EXTRA_OPTION_RE = re.compile(r"\b([E-Z])\b")  # introducing E, F, ...

def detect_instability(text):
    """
    Returns (severity, subtypes)
    severity: 0 none, 1 minor, 2 moderate, 3 severe
    """
    t = safe_str(text)
    if not t.strip():
        return 0, []

    sub = []

    finals = extract_final_choices(t)
    if len(finals) >= 2:
        sub.append("multi_final")
    # option oscillation with reversal language
    ment = mentioned_choices(t)
    if len(ment) >= 2 and has_reversal_connector(t):
        sub.append("option_oscillation")
    # self-correction / revision markers without explicit options
    if has_reversal_connector(t) and re.search(r"(?i)\b(i was wrong|on second thought|actually|"
                                               r"i change my mind|let me correct|correction)\b", t):
        sub.append("self_correction")
    # inconsistent polarity proxy: "is X" and later "is not X" (very conservative)
    neg_flip = re.search(r"(?i)\bis\s+([a-z]{3,})\b.*?\bis\s+not\s+\1\b", t)
    if neg_flip:
        sub.append("polarity_flip")

    # severity scoring
    sev = 0
    if "multi_final" in sub:
        sev = max(sev, 3)
    if "option_oscillation" in sub:
        sev = max(sev, 2)
    if "self_correction" in sub:
        sev = max(sev, 2)
    if "polarity_flip" in sub:
        sev = max(sev, 2)

    return sev, sub

def detect_indecision(text):
    """
    Returns (severity, subtypes)
    """
    t = safe_str(text)
    if not t.strip():
        return 0, []

    sub = []

    # hedging density normalized
    hedges = len(HEDGE_RE.findall(t))
    hedges_norm = norm_per_100_tokens(t, hedges)
    if hedges_norm >= 2.0:
        sub.append("heavy_hedging")
    elif hedges_norm >= 0.7:
        sub.append("light_hedging")

    # alternatives without commitment
    has_alts = bool(re.search(r"\b([A-D])\b.*\b(or|/|,)\b.*\b([A-D])\b", t, flags=re.IGNORECASE))
    committed = has_commitment_marker(t)
    if has_alts and not committed:
        sub.append("alts_no_commit")

    # missing final: no clear final choice and no decisive phrasing
    no_final = not committed and not extract_final_choices(t)
    if no_final:
        sub.append("missing_final")

    # ending as a question (asks reader to choose)
    if QUESTION_RE.search(t):
        sub.append("questioning")

    # severity
    sev = 0
    if "alts_no_commit" in sub or "missing_final" in sub:
        sev = max(sev, 2)
    if "heavy_hedging" in sub:
        sev = max(sev, 2)
    if "questioning" in sub:
        sev = max(sev, 1)
    if "light_hedging" in sub and sev == 0:
        sev = 1

    return sev, sub

def detect_hallucination(text, choices_text=None):
    """
    Returns (severity, subtypes)
    """
    t = safe_str(text)
    if not t.strip():
        return 0, []

    sub = []

    # fabricated refs / URLs / fake tooling
    if CITATION_RE.search(t):
        sub.append("fabricated_citation_style")
    if URL_RE.search(t):
        # URLs in raw generations are often invented; mark but keep severity moderate unless also 'citation style'
        sub.append("url_claim")
    if FAKE_TOOLING_RE.search(t):
        sub.append("fake_tool_use")

    # choices-aware checks
    if isinstance(choices_text, str) and choices_text.strip():
        valid = set(ch.upper() for ch in OPTION_LABEL_RE.findall(choices_text))
        if valid:
            finals = extract_final_choices(t)
            mentions = mentioned_choices(t)
            # outside schema final
            if finals and any(f not in valid for f in finals):
                sub.append("outside_schema_final")
            # introduce extra option labels like E when only A-D exist
            extra = set(x.group(1).upper() for x in EXTRA_OPTION_RE.finditer(t))
            if extra and all(e not in valid for e in extra):
                sub.append("invented_option_label")

    # severity
    sev = 0
    if "outside_schema_final" in sub or "invented_option_label" in sub:
        sev = max(sev, 3)
    if "fabricated_citation_style" in sub and "url_claim" in sub:
        sev = max(sev, 2)
    if "fake_tool_use" in sub:
        sev = max(sev, 2)
    if "url_claim" in sub and sev == 0:
        sev = 1

    return sev, sub

# ========= Per-file processing =========
def analyze_file(filepath):
    df = pd.read_csv(filepath)
    if RESPONSE_COL not in df.columns:
        print(f"Skipping {os.path.basename(filepath)} (missing '{RESPONSE_COL}')")
        return None

    choices_series = df[CHOICES_COL] if CHOICES_COL in df.columns else None

    instab_counts = Counter()
    indec_counts = Counter()
    halluc_counts = Counter()

    instab_total = indec_total = halluc_total = 0
    n = len(df)

    for idx, row in df.iterrows():
        resp = safe_str(row.get(RESPONSE_COL, ""))
        choices_text = safe_str(row.get(CHOICES_COL, "")) if choices_series is not None else None

        sev_i, sub_i = detect_instability(resp)
        sev_d, sub_d = detect_indecision(resp)
        sev_h, sub_h = detect_hallucination(resp, choices_text)

        instab_total += 1 if sev_i > 0 else 0
        indec_total  += 1 if sev_d > 0 else 0
        halluc_total += 1 if sev_h > 0 else 0

        for s in sub_i: instab_counts[s] += 1
        for s in sub_d: indec_counts[s] += 1
        for s in sub_h: halluc_counts[s] += 1

    result = {
        "file": os.path.basename(filepath),
        "total_responses": n,
        # binary occurrence counts
        "instability": instab_total,
        "indecision": indec_total,
        "hallucination": halluc_total,
        # rates
        "instability_rate": round(instab_total / n, 4) if n else 0.0,
        "indecision_rate": round(indec_total / n, 4) if n else 0.0,
        "hallucination_rate": round(halluc_total / n, 4) if n else 0.0,
        # subtype breakdowns
        "instability_subtypes": dict(instab_counts),
        "indecision_subtypes": dict(indec_counts),
        "hallucination_subtypes": dict(halluc_counts),
    }
    return result

def analyze_directory(directory_path):
    totals = []
    for fname in os.listdir(directory_path):
        if fname.lower().endswith(".csv"):
            fp = os.path.join(directory_path, fname)
            res = analyze_file(fp)
            if res:
                totals.append(res)

    # Print per-file
    for r in totals:
        print(f"\n--- {r['file']} ---")
        print(f"Total: {r['total_responses']}")
        print(f"Instability: {r['instability']} (rate {r['instability_rate']})")
        print(f"  subtypes: {r['instability_subtypes']}")
        print(f"Indecision: {r['indecision']} (rate {r['indecision_rate']})")
        print(f"  subtypes: {r['indecision_subtypes']}")
        print(f"Hallucination: {r['hallucination']} (rate {r['hallucination_rate']})")
        print(f"  subtypes: {r['hallucination_subtypes']}")

    # Grand totals
    if totals:
        N = sum(r["total_responses"] for r in totals)
        g_inst = sum(r["instability"] for r in totals)
        g_ind  = sum(r["indecision"] for r in totals)
        g_hal  = sum(r["hallucination"] for r in totals)

        print("\n" + "="*60)
        print("Grand totals across directory")
        print(f"Total responses: {N}")
        print(f"Instability: {g_inst} (rate {round(g_inst/max(1,N),4)})")
        print(f"Indecision: {g_ind} (rate {round(g_ind/max(1,N),4)})")
        print(f"Hallucination: {g_hal} (rate {round(g_hal/max(1,N),4)})")

if __name__ == "__main__":
    # >>>> SET THIS <<<<
    directory_path = r"confidence_results\raw_results\confidence_results_RQ1\qwq_latest"
    if not os.path.isdir(directory_path):
        print(f"Directory not found: {directory_path}")
    else:
        analyze_directory(directory_path)
