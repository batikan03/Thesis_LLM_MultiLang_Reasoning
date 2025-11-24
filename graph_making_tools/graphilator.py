import os
import glob
import re
import math
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- CONFIG ----------
LANG_ORDER = [
    "english","german","french","spanish","polish","danish","turkish"
]
LANG_TITLE = {
    "english":"English","german":"German","french":"French","spanish":"Spanish",
    "polish":"Polish","danish":"Danish","turkish":"Turkish"
}
# Which columns to visualize (must exist in your CSVs)
METRICS = [
    "response_avg_logp",
    "response_avg_logp_per_char",
    "response_perplexity",
    "response_avg_logit_gap",
]
# --- ridgeline params ---
RIDGE_BINS = 200
RIDGE_SPACING = 1.4
RIDGE_HEIGHT_SCALE = 0.85
RIDGE_SMOOTH = 3
# --bocplot params----

# --- boxplot style (like the example image) ---
SHOW_MEAN_CI_OVERLAY = False   # set True if you still want mean ±95% CI dots
BOX_WIDTH = 0.55               # visual width


# ---------------- CONFIG for HEATMAP (merged) ----------------
# Intended languages (row order)
ROW_LANGS = [
    "english","german","french","spanish","polish","danish","turkish",
    "bulgarian","bosnian","albanian","maltese"
]

# Map row names to 2-letter codes
ROW2CODE = {
    "english":"en", "german":"de", "french":"fr", "spanish":"es",
    "polish":"pl", "danish":"da", "turkish":"tr",
    "bulgarian":"bg", "bosnian":"bs", "albanian":"sq", "maltese":"mt",
}

# Detected codes (column order). Keep 'mixed' and 'unknown' at end.
# Include zh if you want to visualize bleed-outs to Chinese.
COL_CODES = [
    "en","de","fr","es","pl","da","tr","bg","bs","sq","mt","zh","mixed","unknown"
]

# Some detectors return close siblings; normalize them into your chosen buckets.
# E.g., collapse hr/sr/sh -> bs (Bosnian bucket)
NORMALIZE_EQUIV = {
    "sh": "bs",  # Serbo-Croatian -> Bosnian bucket
    "hr": "bs",  # Croatian -> Bosnian bucket
    "sr": "bs",  # Serbian -> Bosnian bucket
}
# ----------------------------------------------------------------

# Try to use langid (restrict to your set)
_HAVE_LANGID = False
try:
    import langid
    langid.set_languages([c for c in COL_CODES if c not in ("mixed","unknown")])
    _HAVE_LANGID = True
except Exception:
    _HAVE_LANGID = False

# Optional fallback: langdetect
_HAVE_LANGDETECT = False
try:
    from langdetect import detect
    _HAVE_LANGDETECT = True
except Exception:
    _HAVE_LANGDETECT = False


# ---------------- Utility helpers ----------------
def _natural_lang_key(lang):
    lang_norm = (lang or "").strip().lower()
    if lang_norm in LANG_ORDER:
        return (0, LANG_ORDER.index(lang_norm), lang_norm)
    return (1, lang_norm)

def _title_lang(lang):
    t = LANG_TITLE.get((lang or "").lower(), None)
    return t if t else (lang or "unknown").capitalize()

def _safe_mkdir(p):
    os.makedirs(p, exist_ok=True)
    return p

def _sanitize_column_numeric(df, colname):
    if colname not in df.columns:
        return None
    s = pd.to_numeric(df[colname], errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    return s

def _parse_int_like(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    try:
        if isinstance(x, (int, np.integer)):
            return int(x)
        if isinstance(x, float):
            return int(round(x))
        s = str(x).strip()
        if s == "":
            return None
        v = float(s)
        return int(round(v))
    except Exception:
        m = re.search(r"-?\d+", str(x))
        return int(m.group(0)) if m else None

def _compute_is_correct_col(df):
    ans = df.get("answer", pd.Series([None]*len(df)))
    eva = df.get("evaluation", pd.Series([None]*len(df)))

    ans_norm = ans.map(_parse_int_like)
    eva_norm = eva.map(_parse_int_like)

    # evaluation is 1-based; adjust to 0-based to compare with 'answer'
    eva_adj = eva_norm.map(lambda v: v - 1 if v is not None else None)

    is_corr = []
    for a, e in zip(ans_norm, eva_adj):
        if a is None or e is None:
            is_corr.append(np.nan)
        else:
            is_corr.append(bool(a == e))
    df["is_correct"] = pd.Series(is_corr, index=df.index)
    return df


# ---------------- Load & aggregate ----------------
def load_model_lang_tables(model_dir):
    per_lang = {}
    for csv_path in glob.glob(os.path.join(model_dir, "confidence_*.csv")):
        lang = re.sub(r"^confidence_|\.csv$", "", os.path.basename(csv_path), flags=re.IGNORECASE)
        df = pd.read_csv(csv_path)

        # derive perplexity if missing
        if "response_perplexity" not in df.columns and "response_avg_logp" in df.columns:
            lp = pd.to_numeric(df.get("response_avg_logp", pd.Series(dtype=float)), errors="coerce")
            df["response_perplexity"] = np.exp(-lp)

        for m in METRICS:
            if m in df.columns:
                df[m] = _sanitize_column_numeric(df, m)

        df = _compute_is_correct_col(df)
        per_lang[lang] = df

    lang_list = sorted(per_lang.keys(), key=_natural_lang_key)
    return per_lang, lang_list

def _agg_stats(df, metric):
    if metric not in df.columns:
        return dict(mean=np.nan, se=np.nan, ci95_low=np.nan, ci95_high=np.nan, n=0)
    vals = df[metric].dropna().to_numpy()
    n = len(vals)
    if n == 0:
        return dict(mean=np.nan, se=np.nan, ci95_low=np.nan, ci95_high=np.nan, n=0)
    mean = float(np.mean(vals))
    se = float(np.std(vals, ddof=1) / math.sqrt(n)) if n > 1 else np.nan
    ci95 = 1.96 * se if not math.isnan(se) else np.nan
    return dict(
        mean=mean,
        se=se,
        ci95_low=mean - ci95 if not math.isnan(ci95) else np.nan,
        ci95_high=mean + ci95 if not math.isnan(ci95) else np.nan,
        n=n
    )

def _agg_accuracy(df):
    if "is_correct" not in df.columns:
        return dict(mean=np.nan, se=np.nan, ci95_low=np.nan, ci95_high=np.nan, n=0)
    vals = df["is_correct"].dropna().astype(bool).to_numpy()
    n = len(vals)
    if n == 0:
        return dict(mean=np.nan, se=np.nan, ci95_low=np.nan, ci95_high=np.nan, n=0)
    p = float(np.mean(vals))
    se = math.sqrt(p * (1 - p) / n) if n > 0 else np.nan
    ci95 = 1.96 * se if not math.isnan(se) else np.nan
    return dict(
        mean=p,
        se=se,
        ci95_low=max(0.0, p - ci95) if not math.isnan(ci95) else np.nan,
        ci95_high=min(1.0, p + ci95) if not math.isnan(ci95) else np.nan,
        n=n
    )

def build_summary_table(per_lang, lang_list):
    rows = []
    for lang in lang_list:
        df = per_lang[lang]
        for m in METRICS:
            if m not in df.columns:
                continue
            stats = _agg_stats(df, m)
            rows.append({"language": lang, "metric": m, **stats})
        # accuracy row
        acc_stats = _agg_accuracy(df)
        rows.append({"language": lang, "metric": "accuracy", **acc_stats})
    return pd.DataFrame(rows)


# ---------------- PLOTS ----------------
def _gaussian_smooth(y, k):
    if k <= 0:
        return y
    r = int(k)
    x = np.arange(-r, r+1)
    w = np.exp(-(x**2) / (2*(k/2.0)**2))
    w /= w.sum()
    return np.convolve(y, w, mode="same")

def plot_ridgeline_per_model(per_lang, lang_list, out_dir, model_name, metric):
    series = []
    for lang in lang_list:
        df = per_lang[lang]
        vals = None
        if metric == "accuracy":
            if "is_correct" in df.columns:
                vals = df["is_correct"].dropna().astype(float).to_numpy()
        else:
            if metric in df.columns:
                vals = pd.to_numeric(df[metric], errors="coerce").dropna().to_numpy()
        if vals is None or len(vals) == 0 or not np.isfinite(vals).any():
            continue
        series.append((lang, vals))

    if not series:
        return

    series.sort(key=lambda t: float(np.nanmean(t[1])))
    all_vals = np.concatenate([v for _, v in series], axis=0)
    x_min, x_max = np.nanmin(all_vals), np.nanmax(all_vals)
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_min == x_max:
        if metric == "accuracy":
            x_min, x_max = 0.0, 1.0
        else:
            return

    if metric != "accuracy":
        span = x_max - x_min
        x_min -= 0.05 * span
        x_max += 0.05 * span
    else:
        x_min, x_max = 0.0, 1.0

    bins = np.linspace(x_min, x_max, RIDGE_BINS + 1)
    centers = 0.5 * (bins[:-1] + bins[1:])

    plt.figure(figsize=(10, 0.75 * len(series) + 2))

    for idx, (lang, vals) in enumerate(series):
        use_smooth = (RIDGE_SMOOTH if metric != "accuracy" else 0)
        counts, _ = np.histogram(vals, bins=bins, density=True)
        if use_smooth > 0:
            counts = _gaussian_smooth(counts.astype(float), use_smooth)
        peak = np.max(counts) if np.isfinite(counts).any() else 1.0
        if peak <= 0 or not np.isfinite(peak):
            peak = 1.0
        y = (counts / peak) * (RIDGE_HEIGHT_SCALE * RIDGE_SPACING)

        offset = idx * RIDGE_SPACING
        plt.plot([x_min, x_max], [offset, offset], linewidth=0.6)
        plt.fill_between(centers, offset, offset + y, alpha=0.7, linewidth=1)

    tick_pos = [i * RIDGE_SPACING + (RIDGE_HEIGHT_SCALE * RIDGE_SPACING) * 0.45 for i in range(len(series))]
    tick_lbl = []
    for lang, _ in series:
        short = lang[:2].title() if len(lang) > 2 else lang.title()
        m = {"english": "En", "german": "De", "french": "Fr",
             "spanish": "Es", "polish": "Pl", "danish": "Da", "turkish": "Tr"}
        tick_lbl.append(m.get(lang.lower(), short))
    plt.yticks(tick_pos, tick_lbl)

    plt.xlim(x_min, x_max)
    plt.ylim(-RIDGE_SPACING * 0.2, (len(series) - 1) * RIDGE_SPACING + RIDGE_SPACING)
    xlabel_map = {
        "response_avg_logp": "Avg log-prob (per token)",
        "response_avg_logp_per_char": "Avg log-prob (per char)",
        "response_perplexity": "Perplexity (lower is better)",
        "response_avg_logit_gap": "Avg logit gap (top1 - top2)",
        "accuracy": "Correct (0) … (1)",
    }
    title_map = {
        "response_avg_logp": "Response avg log-prob",
        "response_avg_logp_per_char": "Response avg log-prob per char",
        "response_perplexity": "Response perplexity",
        "response_avg_logit_gap": "Response avg logit gap",
        "accuracy": "Accuracy",
    }
    plt.xlabel(xlabel_map.get(metric, metric))
    title = f"{model_name}: Ridgeline – " + title_map.get(metric, metric)
    plt.title(title)
    plt.tight_layout()

    out_png = os.path.join(out_dir, f"{model_name}_{metric}_ridgeline.png")
    plt.savefig(out_png, dpi=150)
    plt.close()

def plot_accuracy_per_model(summary_df, out_dir, model_name):
    """
    Dedicated accuracy figure: bars with 95% CI and % labels.
    Saves as {model_name}_accuracy.png
    """
    sub = summary_df[summary_df["metric"] == "accuracy"].copy()
    if sub.empty:
        return
    sub["__key"] = sub["language"].map(lambda x: _natural_lang_key(x))
    sub = sub.sort_values("__key")

    langs = sub["language"].tolist()
    x = np.arange(len(langs))

    acc_mean = (sub["mean"].to_numpy(dtype=float) * 100.0)
    acc_low  = (sub["ci95_low"].to_numpy(dtype=float) * 100.0)
    acc_high = (sub["ci95_high"].to_numpy(dtype=float) * 100.0)

    # 95% CI as asymmetric error bars (top/bottom distances from the mean)
    yerr = np.vstack([
        np.where(np.isnan(acc_mean) | np.isnan(acc_low),  0.0, acc_mean - acc_low),
        np.where(np.isnan(acc_mean) | np.isnan(acc_high), 0.0, acc_high - acc_mean),
    ])

    plt.figure(figsize=(10, 5.5))
    plt.bar(x, acc_mean, width=0.6)
    plt.errorbar(x, acc_mean, yerr=yerr, fmt="none", capsize=4)

    # add labels on bars
    for xi, yi in zip(x, acc_mean):
        if np.isfinite(yi):
            plt.text(xi, yi + 1.0, f"{yi:.1f}%", ha="center", va="bottom", fontsize=9)

    plt.xticks(x, [_title_lang(l) for l in langs], rotation=30, ha="right")
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Language")
    plt.title(f"{model_name}: Accuracy by language")
    plt.tight_layout()

    out_png = os.path.join(out_dir, f"{model_name}_accuracy.png")
    plt.savefig(out_png, dpi=150)
    plt.close()



# ======== CHANGED: per-metric plots -> BOXPLOTS with 95% CI overlay ========
def plot_metric_per_model(summary_df, out_dir, model_name, per_lang):
    """
    For each metric in METRICS, draw per-language boxplots in a classic
    grayscale style (filled boxes, thick black medians, slim whiskers/caps,
    no fliers, whiskers to min–max). Optionally overlay mean ±95% CI.
    Saves: {model_name}_{metric}.png
    """
    for m in METRICS:
        # Collect raw values by language in stable order
        langs_all = sorted(per_lang.keys(), key=_natural_lang_key)
        data, keep_langs, means, cis = [], [], [], []

        for lang in langs_all:
            df = per_lang[lang]
            if m not in df.columns:
                continue
            vals = pd.to_numeric(df[m], errors="coerce").dropna().to_numpy()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            keep_langs.append(lang)
            data.append(vals)

            # for optional CI overlay
            n = vals.size
            mu = float(np.mean(vals))
            if n > 1:
                se = float(np.std(vals, ddof=1) / math.sqrt(n))
                ci = 1.96 * se
            else:
                ci = 0.0
            means.append(mu)
            cis.append(ci)

        if not data:
            continue

        x = np.arange(len(data))

        # --- figure ---
        plt.figure(figsize=(10, 5.5))

        # Classic clinical style
        bp = plt.boxplot(
            data,
            positions=x,
            widths=BOX_WIDTH,
            showfliers=False,
            whis=[0, 100],              # whiskers to min–max
            patch_artist=True
        )

        # Style elements
        # boxes: filled grayscale, black edge; medians: thick black; whiskers/caps: slim black
        for i, box in enumerate(bp['boxes']):
            box.set_facecolor('#CFCFCF' if i % 2 == 0 else '#9E9E9E')  # alternating light/dark gray
            box.set_edgecolor('black')
            box.set_linewidth(1.0)

        for med in bp['medians']:
            med.set_color('black')
            med.set_linewidth(3.0)

        for w in bp['whiskers']:
            w.set_color('black')
            w.set_linewidth(1.0)

        for cap in bp['caps']:
            cap.set_color('black')
            cap.set_linewidth(1.0)

        # Optional: overlay mean ±95% CI as small dots with error bars
        if SHOW_MEAN_CI_OVERLAY:
            means = np.asarray(means, dtype=float)
            cis = np.asarray(cis, dtype=float)
            yerr = np.vstack([cis, cis])  # symmetric
            plt.errorbar(
                x, means, yerr=yerr,
                fmt="o", markersize=4, capsize=3, linewidth=1, color="black"
            )

        # Labels/titles like before
        plt.xticks(x, [_title_lang(l) for l in keep_langs], rotation=30, ha="right")

        if m == "response_avg_logp":
            ylabel = "Avg log-prob (per token)"
            title  = f"{model_name}: Response avg log-prob"
        elif m == "response_avg_logp_per_char":
            ylabel = "Avg log-prob (per char)"
            title  = f"{model_name}: Response avg log-prob per char"
        elif m == "response_perplexity":
            ylabel = "Perplexity (lower is better)"
            title  = f"{model_name}: Response perplexity"
        elif m == "response_avg_logit_gap":
            ylabel = "Avg logit gap (top1 - top2)"
            title  = f"{model_name}: Response avg logit gap"
        else:
            ylabel = m
            title  = f"{model_name}: {m}"

        plt.ylabel(ylabel)
        plt.xlabel("Language")
        plt.title(title)
        plt.tight_layout()

        out_png = os.path.join(out_dir, f"{model_name}_{m}.png")
        plt.savefig(out_png, dpi=150)
        plt.close()

# ===========================================================================


# ---------------- HEATMAP HELPERS (merged) ----------------
def normalize_code(code: str) -> str:
    """Map detector code -> your matrix code; unseen -> 'unknown'."""
    if not isinstance(code, str):
        return "unknown"
    code = code.strip().lower()
    code = NORMALIZE_EQUIV.get(code, code)
    return code if code in COL_CODES else "unknown"

def primary_code_from_label(label: str) -> str:
    """
    Parse 'response_language_label' like:
    'en', 'en with fr fragments', 'tr and en mix', 'mixed_no_dominant', 'unknown'
    """
    if not isinstance(label, str) or not label.strip():
        return "unknown"
    lab = label.strip().lower()
    if lab.startswith("unknown"):
        return "unknown"
    if "mix" in lab:
        return "mixed"
    m = re.search(r'\b([a-z]{2})\b', lab)
    if m:
        return normalize_code(m.group(1))
    return "unknown"

def detect_lang_code(text: str) -> str:
    """
    Detect response language (restricted set). Prefer langid; fallback to langdetect.
    """
    if not isinstance(text, str) or not text.strip():
        return "unknown"
    # 1) langid
    if _HAVE_LANGID:
        try:
            code, score = langid.classify(text)
            return normalize_code(code)
        except Exception:
            pass
    # 2) langdetect
    if _HAVE_LANGDETECT:
        try:
            code = detect(text)
            return normalize_code(code)
        except Exception:
            pass
    return "unknown"

def ensure_dirs(root_dir, model_name):
    root_name = os.path.basename(os.path.normpath(root_dir))
    out_root = os.path.join(os.path.dirname(os.path.normpath(root_dir)),
                            f"graph_results_{root_name}")
    model_out = os.path.join(out_root, model_name)
    heat_out = os.path.join(model_out, "heatmaps")
    os.makedirs(heat_out, exist_ok=True)
    return out_root, model_out, heat_out


def plot_matrix(mat_df, title, out_path, fmt="count"):
    import matplotlib.pyplot as plt
    from matplotlib import colors, ticker
    import numpy as np

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
        # Perceived luminance (ITU-R BT.601)
        L = 0.299*r + 0.587*g + 0.114*b
        return "white" if L < 0.5 else "black"

    for i in range(nrows):
        for j in range(ncols):
            val = values[i, j]
            if fmt == "pct":
                txt = f"{val:.0f}%"
                disp_val = float(val)
            else:
                # Prefer int-like where it makes sense
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

def derive_row_label_from_filename(csv_path: str) -> str:
    """
    Accept both 'confidence_english.csv' and code-based 'confidence_bg.csv'.
    Map short code -> long name using ROW2CODE inverse.
    """
    base = os.path.basename(csv_path)
    m = re.match(r'^confidence_(.+)\.csv$', base, re.IGNORECASE)
    if not m:
        return "unknown"
    raw = m.group(1).lower()

    # If already long name
    if raw in ROW_LANGS:
        return raw

    # If it's a code, map back to long name if possible:
    inv = {v: k for k, v in ROW2CODE.items()}  # e.g. 'en' -> 'english'
    # Normalize equivalents first (e.g., sr/hr/sh -> bs)
    raw = NORMALIZE_EQUIV.get(raw, raw)
    return inv.get(raw, raw)

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

        # Decide how to get detected language codes:
        if "response_language_label" in df.columns and df["response_language_label"].notna().any():
            codes = df["response_language_label"].apply(primary_code_from_label)
            detector_used = detector_used or "column:response_language_label"
        elif "response" in df.columns:
            codes = df["response"].apply(detect_lang_code)
            detector_used = detector_used or ("langid" if _HAVE_LANGID else ("langdetect" if _HAVE_LANGDETECT else "none"))
        else:
            print(f"  WARNING: {os.path.basename(csv_path)} lacks both 'response_language_label' and 'response'; skipping.")
            continue

        # Accumulate into the fixed matrix
        vc = codes.value_counts()
        total_rows += int(vc.sum())
        for code, n in vc.items():
            code = normalize_code(code)
            mat.loc[intended, code] += int(n)

    if total_rows == 0:
        print("  No usable rows found. Nothing to save.")
        return

    # Row-normalized percentage matrix
    row_sums = mat.sum(axis=1).replace(0, np.nan)
    row_pct = mat.div(row_sums, axis=0) * 100.0
    row_pct = row_pct.fillna(0.0)

    # Save CSVs + PNGs
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


# ---------------- Main runner ----------------
def main(root_dir):
    root_name = os.path.basename(os.path.normpath(root_dir))
    out_root = _safe_mkdir(os.path.join(os.path.dirname(os.path.normpath(root_dir)),
                                        f"graph_results_{root_name}"))

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

        print(f"[Model] {model_name}")
        per_lang, lang_list = load_model_lang_tables(md)
        if not per_lang:
            print(f"  No language CSVs in {md}, skipping.")
            continue

        model_out_dir = _safe_mkdir(os.path.join(out_root, model_name))
        plots_out_dir = _safe_mkdir(os.path.join(model_out_dir, "plots"))

        summary_df = build_summary_table(per_lang, lang_list)
        summary_path = os.path.join(model_out_dir, f"{model_name}_summary_metrics.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"  Saved summary -> {summary_path}")

        # NEW: Per-metric BOXPLOTS with mean ± 95% CI overlay
        plot_metric_per_model(summary_df, plots_out_dir, model_name, per_lang)

        # Separate accuracy figure
        plot_accuracy_per_model(summary_df, plots_out_dir, model_name)

        # Optional: metric ridgelines (kept as-is)
        for m in METRICS:
            plot_ridgeline_per_model(per_lang, lang_list, plots_out_dir, model_name, m)

        # Heatmaps (merged pipeline)
        print(f"[Heatmap] {model_name}")
        heatmap_for_model(md, model_name, root_dir)

        print(f"  Saved plots under {plots_out_dir}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="Path to confidence results root directory (e.g., confidence_results_RQ3)")
    args = ap.parse_args()
    main(args.root)
