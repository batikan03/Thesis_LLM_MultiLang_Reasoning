#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Standalone heatmap maker for response language detection.
Preserves the same storing logic and filenames as your original pipeline.

Usage:
    python make_heatmaps.py /path/to/confidence_results_ROOT
"""

import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors, ticker

# ---------------- CONFIG ----------------
# Intended languages (row order)
ROW_LANGS = [
    "english","german","french","spanish","polish","danish","turkish",
    "bulgarian","bosnian","albanian","maltese"
]

# Map row names to 2-letter codes
ROW2CODE = {
    "english":"en","german":"de","french":"fr","spanish":"es",
    "polish":"pl","danish":"da","turkish":"tr",
    "bulgarian":"bg","bosnian":"bs","albanian":"sq","maltese":"mt",
}

# Detected codes (column order). Keep 'mixed' and 'unknown' at end.
COL_CODES = [
    "en","de","fr","es","pl","da","tr","bg","bs","sq","mt","zh","mixed","unknown"
]

# Normalize close siblings into buckets (e.g., hr/sr/sh -> bs)
NORMALIZE_EQUIV = {"sh":"bs","hr":"bs","sr":"bs"}

# Robust mapping for names/synonyms/self-names → ISO 2-letter codes
NAME2CODE = {
    # English names
    "english":"en","german":"de","french":"fr","spanish":"es",
    "polish":"pl","danish":"da","turkish":"tr",
    "bulgarian":"bg","bosnian":"bs","albanian":"sq","maltese":"mt",
    "chinese":"zh",

    # Common autonyms / variants
    "deutsch":"de","français":"fr","francais":"fr","español":"es","espanol":"es",
    "polski":"pl","dansk":"da","türkçe":"tr","turkce":"tr",
    "български":"bg","bulgarski":"bg",
    "bosanski":"bs","hrvatski":"bs","srpski":"bs",
    "shqip":"sq","malti":"mt",
    "中文":"zh","汉语":"zh","漢語":"zh",

    # Turkish exonyms that sometimes show up
    "ingilizce":"en","almanca":"de","fransızca":"fr","ispanyolca":"es",
    "lehçe":"pl","lehce":"pl","danca":"da","bulgarca":"bg",
    "boşnakça":"bs","bosnakca":"bs","arnavutça":"sq","arnavutca":"sq",
    "malta dili":"mt",

    # Legacy cluster tags
    "sh":"bs","hr":"bs","sr":"bs",
}

# ---------- Optional detectors ----------
_HAVE_LANGID = False
try:
    import langid
    langid.set_languages([c for c in COL_CODES if c not in ("mixed","unknown")])
    _HAVE_LANGID = True
except Exception:
    _HAVE_LANGID = False

_HAVE_LANGDETECT = False
try:
    from langdetect import detect
    _HAVE_LANGDETECT = True
except Exception:
    _HAVE_LANGDETECT = False


# ---------- Heuristic features per language ----------
# Diacritic / script clues (fast & strong)
_DIACRITIC_PATTERNS = {
    "mt": r"[għĦħŻżĠġĊċ]",                       # Maltese unique letters/digraph
    "da": r"[æøåÆØÅ]",                            # Danish
    "pl": r"[ąćęłńóśźżĄĆĘŁŃÓŚŹŻ]",                # Polish
    "tr": r"[ğşıİöçüĞŞÖÇÜ]",                      # Turkish
    "de": r"[äöüÄÖÜß]",                            # German
    "es": r"[áéíóúüñÁÉÍÓÚÜÑ]",                    # Spanish
    "fr": r"[àâæçéèêëîïôœùûüÿÀÂÆÇÉÈÊËÎÏÔŒÙÛÜŸ]",   # French
    "bs": r"[čćđšžČĆĐŠŽ]",                        # Bosnian/Croatian/Serbian (Latin)
    "sq": r"[ëËçÇ]",                               # Albanian
}

# Cyrillic detection; disambiguate Bulgarian vs Russian-ish
_CYRILLIC_RE   = re.compile(r"[\u0400-\u04FF]")
_RU_ONLY_ISH   = re.compile(r"[ыэ]")  # letters used in RU, not in BG (rough but helpful)

# Quick Chinese check
_CHINESE_RE = re.compile(r"[\u4E00-\u9FFF]")

# Light keyword signals (tiny, safe)
_KEYWORDS = {
    "mt": ["għal","għand","għax","li","minn","x'inhu","huwa","tagħha","tagħhom"],
    "da": ["og","ikke","jeg","det","er","til","på","af","med","som"],
    "pl": ["i","że","jest","nie","po","na","to","czy","się"],
    "tr": ["ve","bir","için","değil","bu","şu","mı","mi","çok"],
    "de": ["und","nicht","ich","ist","das","ein","mit","für","zum","den"],
    "es": ["y","no","que","de","el","la","es","en","un"],
    "fr": ["et","pas","que","de","le","la","est","en","un"],
    "bs": ["je","sam","nije","što","koji","ćemo","više","samo","kada"],
    "sq": ["dhe","është","një","të","për","në","ka","si","me"],
    "en": ["the","and","is","of","to","in","that","it","for"],
    "bg": ["и","е","в","на","се","по","за","от","не"],  # generic; cyrillic check matters most
}


# ---------------- Helpers ----------------
def _safe_mkdir(p):
    os.makedirs(p, exist_ok=True)
    return p

def ensure_dirs(root_dir, model_name):
    root_name = os.path.basename(os.path.normpath(root_dir))
    out_root  = os.path.join(os.path.dirname(os.path.normpath(root_dir)),
                             f"graph_results_{root_name}")
    model_out = os.path.join(out_root, model_name)
    heat_out  = os.path.join(model_out, "heatmaps")
    os.makedirs(heat_out, exist_ok=True)
    return out_root, model_out, heat_out

def normalize_code(code: str) -> str:
    """Map detector code -> your matrix code; unseen -> 'unknown'."""
    if not isinstance(code, str):
        return "unknown"
    code = code.strip().lower()
    code = NORMALIZE_EQUIV.get(code, code)
    return code if code in COL_CODES else "unknown"

def primary_code_from_label(label: str) -> str:
    """
    Parse labels like:
    'en', 'bg', 'bg_BG', 'Bulgarian', 'български', 'bulgarca',
    'en with fr fragments', 'mixed_no_dominant', etc.
    """
    if not isinstance(label, str) or not label.strip():
        return "unknown"
    lab = label.strip().lower()

    # explicit mixed/unknown
    if "mix" in lab:
        return "mixed"
    if lab.startswith("unknown"):
        return "unknown"

    # 1) direct 2-letter code (handles 'bg', 'bg_BG', 'tr-TR', etc.)
    m = re.search(r'\b([a-z]{2})(?:(?:[_-])[a-z]{2})?\b', lab)
    if m:
        return normalize_code(m.group(1))

    # 2) name/synonym substring lookup
    for name, code in NAME2CODE.items():
        if name in lab:
            return normalize_code(code)

    # 3) last-resort: ISO-639-2 (three-letter) handful
    three2two = {
        "eng":"en","deu":"de","ger":"de","fra":"fr","fre":"fr","spa":"es",
        "pol":"pl","dan":"da","tur":"tr","bul":"bg","bos":"bs","sqi":"sq","alb":"sq","mlt":"mt","zho":"zh","chi":"zh"
    }
    m3 = re.search(r'\b([a-z]{3})\b', lab)
    if m3 and m3.group(1) in three2two:
        return normalize_code(three2two[m3.group(1)])

    return "unknown"

def _score_diacritics(text: str) -> dict:
    scores = {}
    for code, pat in _DIACRITIC_PATTERNS.items():
        hits = len(re.findall(pat, text))
        if hits:
            scores[code] = scores.get(code, 0) + hits * 2.0  # diacritics are strong
    # Cyrillic => likely BG unless Russian-only letters present
    if _CYRILLIC_RE.search(text):
        if not _RU_ONLY_ISH.search(text):   # no RU-only letters
            scores["bg"] = scores.get("bg", 0) + 3.0
    # Chinese characters
    if _CHINESE_RE.search(text):
        scores["zh"] = scores.get("zh", 0) + 5.0
    return scores

def _score_keywords(text: str) -> dict:
    scores = {}
    low = text.lower()
    for code, words in _KEYWORDS.items():
        c = 0
        for w in words:
            if re.search(rf"\b{re.escape(w)}\b", low):
                c += 1
        if c:
            scores[code] = scores.get(code, 0) + min(c, 5) * 0.6  # soft signal
    return scores

def _langid_guess(text: str):
    if not _HAVE_LANGID:
        return None
    try:
        code, score = langid.classify(text)
        return normalize_code(code), float(score)
    except Exception:
        return None

def _langdetect_guess(text: str):
    if not _HAVE_LANGDETECT:
        return None
    try:
        code = detect(text)
        return normalize_code(code), 0.0  # no score provided
    except Exception:
        return None

def detect_lang_code(text: str) -> str:
    """
    Ensemble detector:
      - diacritics/script -> strong
      - langid/langdetect (if present)
      - small keyword bump
      - 'mixed' if 2 winners within margin; else top
      - 'unknown' for empty/low-signal
    """
    if not isinstance(text, str) or not text.strip():
        return "unknown"
    t = text.strip()

    # 1) Hard Chinese check first
    if _CHINESE_RE.search(t):
        return "zh"

    # 2) Diacritics/script
    scores = _score_diacritics(t)

    # 3) Optional detectors
    li = _langid_guess(t)
    if li and li[0] != "unknown":
        # 1..3 points extra depending on confidence (bounded)
        scores[li[0]] = scores.get(li[0], 0.0) + 1.0 + max(0.0, min(2.0, li[1]))
    ld = _langdetect_guess(t)
    if ld and ld[0] != "unknown":
        scores[ld[0]] = scores.get(ld[0], 0.0) + 1.0

    # 4) Keywords (soft)
    kw = _score_keywords(t)
    for k, v in kw.items():
        scores[k] = scores.get(k, 0.0) + v

    if not scores:
        return "unknown"

    winners = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top_code, top_val = winners[0]
    second_val = winners[1][1] if len(winners) > 1 else -999.0

    MIN_TOP = 1.0
    CLOSE_DELTA = 0.6  # within this => mixed

    if top_val < MIN_TOP or top_code == "unknown":
        return "unknown"
    if len(winners) > 1 and (top_val - second_val) < CLOSE_DELTA:
        return "mixed"
    return top_code


# ---------------- Heatmap plotting ----------------
def plot_matrix(mat_df, title, out_path, fmt="count"):
    values = mat_df.values
    nrows, ncols = values.shape

    # Choose White->Red colormap and sensible normalization
    if fmt == "pct":
        vmin, vmax = 0.0, 100.0
    else:
        vmax = np.nanmax(values) if np.isfinite(values).any() else 1.0
        vmin = 0.0
    cmap = plt.get_cmap("Reds")
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(values, aspect="auto", cmap=cmap, norm=norm)

    # Ticks & labels
    ax.set_xticks(np.arange(ncols))
    ax.set_yticks(np.arange(nrows))
    ax.set_xticklabels(mat_df.columns)
    ax.set_yticklabels(mat_df.index)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    # Draw faint grid lines to separate cells
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which="minor", color="#000000", alpha=0.15, linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Cell text with contrast-aware color
    def _text_color(val):
        rgba = cmap(norm(val))
        r, g, b = rgba[:3]
        L = 0.299*r + 0.587*g + 0.114*b
        return "white" if L < 0.5 else "black"

    for i in range(nrows):
        for j in range(ncols):
            val = values[i, j]
            if fmt == "pct":
                txt = f"{val:.0f}%"
                disp_val = float(val)
            else:
                try:
                    txt = f"{int(val)}"
                except Exception:
                    txt = f"{val:.0f}"
                disp_val = float(val) if np.isfinite(val) else 0.0
            ax.text(
                j, i, txt,
                ha="center", va="center",
                fontsize=11, fontweight="bold",
                color=_text_color(disp_val)
            )

    ax.set_xlabel("Detected response language (primary code)")
    ax.set_ylabel("Intended language (from filename)")
    ax.set_title(title)

    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if fmt == "pct":
        cb.ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=100))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------- Row label & file handling ----------------
def derive_row_label_from_filename(csv_path: str) -> str:
    """
    Accept both 'confidence_english.csv' and code-based 'confidence_bg.csv'.
    Also tolerate extra tokens like 'confidence_bg_input.csv' (keeps first token).
    Map short code -> long name using ROW2CODE inverse.
    """
    base = os.path.basename(csv_path)
    m = re.match(r'^confidence_([^.]+)\.csv$', base, re.IGNORECASE)
    if not m:
        return "unknown"
    raw = m.group(1).lower()

    # keep only the first token if multiple underscores (bg_input -> bg)
    raw = raw.split("_", 1)[0]

    # If already long name
    if raw in ROW_LANGS:
        return raw

    # If it's a code, map back to long name if possible:
    inv = {v: k for k, v in ROW2CODE.items()}  # e.g. 'en' -> 'english'
    # Normalize equivalents first (e.g., sr/hr/sh -> bs)
    raw = NORMALIZE_EQUIV.get(raw, raw)
    return inv.get(raw, raw)


# ---------------- Main heatmap builder ----------------
def primary_code_from_any(row) -> str:
    """
    Hybrid per-row detection:
      - Prefer parsed response_language_label if present and meaningful,
      - Else ensemble on response text,
      - Fallback back to parsed label if ensemble was unknown.
    """
    label = row.get("response_language_label", None)
    if isinstance(label, str) and label.strip():
        code = primary_code_from_label(label)
        if code != "unknown":
            return code

    resp = row.get("response", "")
    code = detect_lang_code(resp)

    if code == "unknown" and isinstance(label, str) and label.strip():
        code = primary_code_from_label(label)

    return code

def heatmap_for_model(model_dir, model_name, root_dir):
    # Fixed axes to guarantee order:
    mat = pd.DataFrame(0, index=ROW_LANGS, columns=COL_CODES, dtype=int)

    total_rows = 0
    detector_used = None

    csvs = glob.glob(os.path.join(model_dir, "confidence_*.csv"))
    if not csvs:
        print(f"  No 'confidence_*.csv' under {model_dir}, skipping.")
        return

    for csv_path in csvs:
        intended = derive_row_label_from_filename(csv_path)
        if intended not in ROW_LANGS:
            # skip anything not in your locked list
            continue

        df = pd.read_csv(csv_path)

        # Column-wise detection with hybrid logic per row
        if "response_language_label" in df.columns or "response" in df.columns:
            codes = df.apply(primary_code_from_any, axis=1)
            src = "hybrid(label+ensemble)"
        else:
            print(f"  WARNING: {os.path.basename(csv_path)} lacks both label and response; skipping.")
            continue

        # Accumulate into the fixed matrix
        vc = codes.value_counts()
        total_rows += int(vc.sum())
        for code, n in vc.items():
            code = normalize_code(code)
            mat.loc[intended, code] += int(n)

        detector_used = detector_used or src

    if total_rows == 0:
        print("  No usable rows found. Nothing to save.")
        return

    # Row-normalized percentage matrix
    row_sums = mat.sum(axis=1).replace(0, np.nan)
    row_pct = mat.div(row_sums, axis=0) * 100.0
    row_pct = row_pct.fillna(0.0)

    # Save CSVs + PNGs (same storing logic)
    _, _, heat_out = ensure_dirs(root_dir, model_name)
    counts_csv = os.path.join(heat_out, f"{model_name}_response_language_counts.csv")
    rowpct_csv = os.path.join(heat_out, f"{model_name}_response_language_rowpct.csv")
    mat.to_csv(counts_csv)
    row_pct.to_csv(rowpct_csv)
    print(f"  Saved: {counts_csv}")
    print(f"  Saved: {rowpct_csv}")
    print(f"  Detector used: {detector_used or 'n/a'}")

    plot_matrix(mat,
                f"{model_name}: Response language counts",
                os.path.join(heat_out, f"{model_name}_response_language_heatmap_counts.png"),
                fmt="count")
    plot_matrix(row_pct,
                f"{model_name}: Response language row-normalized (%)",
                os.path.join(heat_out, f"{model_name}_response_language_heatmap_rowpct.png"),
                fmt="pct")


def main(root_dir):
    model_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                  if os.path.isdir(os.path.join(root_dir, d))]
    if not model_dirs:
        print(f"No model directories under {root_dir}")
        return

    for md in sorted(model_dirs):
        model_name = os.path.basename(md)
        csvs = glob.glob(os.path.join(md, "confidence_*.csv"))
        if not csvs:
            continue
        print(f"[Heatmap] {model_name}")
        heatmap_for_model(md, model_name, root_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Path to confidence results root directory (e.g., confidence_results_RQ3)")
    args = ap.parse_args()
    main(args.root)
