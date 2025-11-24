import os
import glob
import re
import gc
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from langdetect import detect_langs, DetectorFactory, LangDetectException
from collections import defaultdict
from tqdm import tqdm
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

print(os.path.abspath("./offload"))
print(os.listdir("./offload"))

DetectorFactory.seed = 0
chinese_char_pattern = re.compile(
    r'[\u4E00-\u9FFF\u3400-\u4DBF\U00020000-\U0002A6DF'
    r'\U0002A700-\U0002B73F\U0002B740-\U0002B81F'
    r'\U0002B820-\U0002CEAF\U0002CEB0-\U0002EBEF'
    r'\U00030000-\U0003134F]'
)

def get_language_label(text):
    if not isinstance(text, str) or not text.strip():
        return 'unknown'
    try:
        detections = detect_langs(text)
        langs_probs = [(lang.lang, lang.prob) for lang in detections]
    except LangDetectException:
        return 'unknown'
    total = sum(p for _, p in langs_probs) or 1
    langs_probs = [(l, p / total) for l, p in langs_probs]
    primary, p0 = langs_probs[0]
    fragments = [l for l, p in langs_probs[1:] if p >= 0.05]
    label = primary
    if p0 < 0.85:
        if p0 >= 0.75:
            label = f"{primary} with {' and '.join(fragments)} fragments" if fragments else primary
        elif p0 >= 0.5:
            label = f"{primary} and {' and '.join(fragments)} mix" if fragments else f"{primary} mix"
        else:
            mix_langs = [l for l, p in langs_probs if p >= 0.15]
            label = f"{' and '.join(mix_langs)} mix" if mix_langs else 'mixed_no_dominant'
    if chinese_char_pattern.search(text or ""):
        label += " with foreign/irregular characters"
    return label

# Map Ollama model dir -> HF repo id
ollama_to_hf = {
    'qwen2.5_7b': "Qwen/Qwen2.5-7B",
    'qwq_latest': "Qwen/QwQ-32B",
    'deepseek-r1_14b-qwen-distill-q8_0': "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
}


_lang_from_fname_re = re.compile(r'_result_([a-zA-Z]+)', re.IGNORECASE)

def extract_lang_from_filename(filename: str) -> str:
    """
    Extracts the language from filenames like:
    deepseek-r1_14b-qwen-distill-q8_0_result_danish.csv
    """
    base = os.path.basename(filename)
    m = _lang_from_fname_re.search(base)
    if m:
        return m.group(1).lower()
    return "unknown"

def _safe_logsoftmax_fp32(logits):
    # logits: [*, V] possibly float16 on GPU
    return F.log_softmax(logits.float(), dim=-1)

def score_sequence(model, tokenizer, text: str, max_window: int = 1024, stride: int = 512):
    max_ctx = getattr(model.config, "max_position_embeddings", 2048)
    max_window = min(max_window, max_ctx)

    if not isinstance(text, str) or not text.strip():
        return {"avg_logp": 0.0, "perplexity": float('inf'),
                "avg_logit_gap": 0.0, "token_count": 0, "char_count": 0,
                "avg_logp_per_char": 0.0}

    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(model.device)
    L = input_ids.shape[-1]
    if L < 2:
        return {"avg_logp": 0.0, "perplexity": float('inf'),
                "avg_logit_gap": 0.0, "token_count": int(L), "char_count": len(text),
                "avg_logp_per_char": 0.0}

    sum_logp, sum_gap, count = 0.0, 0.0, 0
    prev_end = 0
    with torch.inference_mode():
        for i in range(0, L - 1, stride):
            end_loc = min(i + max_window, L)              # grow by 'stride' via i
            begin_loc = max(end_loc - max_window, 0)
            trg_len = end_loc - prev_end

            window_input = input_ids[:, begin_loc:end_loc]
            outputs = model(input_ids=window_input)
            logits = outputs.logits                # [1, W, V]
            next_logits = logits[:, :-1, :]        # [1, W-1, V]
            target = window_input.clone()
            if trg_len < target.size(1):
                target[:, :-trg_len] = -100        # mask previously counted tokens
            next_target = target[:, 1:]            # [1, W-1]

            next_target_clamped = next_target.clone()
            next_target_clamped[next_target_clamped == -100] = 0

            logprobs = _safe_logsoftmax_fp32(next_logits)
            gathered = logprobs.gather(-1, next_target_clamped.unsqueeze(-1)).squeeze(-1)
            valid_mask = (next_target != -100)

            if valid_mask.any():
                sum_logp += gathered[valid_mask].sum().item()
                top2 = next_logits.topk(2, dim=-1).values.float()
                gaps = (top2[..., 0] - top2[..., 1])[valid_mask]
                sum_gap += gaps.sum().item()
                count += int(valid_mask.sum().item())

            prev_end = end_loc

    avg_logp = (sum_logp / count) if count else 0.0
    ppl = float(torch.exp(torch.tensor(-avg_logp)).item()) if count else float('inf')
    avg_gap = (sum_gap / count) if count else 0.0
    char_count = len(text)
    avg_logp_per_char = (sum_logp / char_count) if char_count > 0 else 0.0

    return {
        "avg_logp": avg_logp,
        "perplexity": ppl,
        "avg_logit_gap": avg_gap,
        "token_count": int(L),
        "char_count": char_count,
        "avg_logp_per_char": avg_logp_per_char,
    }

def score_response_given_prompt(model, tokenizer, prompt_text: str, response_text: str,
                                max_window: int = 1024, stride: int = 512):
    """
    Conditional scoring: log p(response | prompt).
    We concatenate [prompt][response] and only score tokens belonging to response.
    Returns token- and char-normalized metrics for the response portion only.
    """
    if not isinstance(response_text, str) or not response_text.strip():
        return {"avg_logp": 0.0, "perplexity": float('inf'),
                "avg_logit_gap": 0.0, "token_count": 0, "char_count": 0,
                "avg_logp_per_char": 0.0}

    # Tokenize separately to know the boundary
    enc_prompt = tokenizer(prompt_text or "", return_tensors="pt", add_special_tokens=False)
    enc_resp = tokenizer(response_text, return_tensors="pt", add_special_tokens=False)

    ids_prompt = enc_prompt["input_ids"]
    ids_resp = enc_resp["input_ids"]
    input_ids = torch.cat([ids_prompt, ids_resp], dim=1).to(model.device)
    Lp = ids_prompt.shape[-1]
    Lr = ids_resp.shape[-1]
    L = Lp + Lr
    if L < 2 or Lr == 0:
        return {"avg_logp": 0.0, "perplexity": float('inf'),
                "avg_logit_gap": 0.0, "token_count": int(Lr),
                "char_count": len(response_text), "avg_logp_per_char": 0.0}

    # Boolean mask: which positions belong to response
    resp_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    resp_mask[:, Lp:] = True

    sum_logp, sum_gap, count = 0.0, 0.0, 0
    prev_end = 0
    with torch.inference_mode():
        for i in range(0, L - 1, stride):
            end_loc = min(i + 1, L)
            begin_loc = max(end_loc - max_window, 0)
            trg_len = end_loc - prev_end

            window_input = input_ids[:, begin_loc:end_loc]               # [1, W]
            outputs = model(input_ids=window_input)
            logits = outputs.logits                                      # [1, W, V]
            next_logits = logits[:, :-1, :]                              # [1, W-1, V]

            # targets aligned to next token
            target = window_input.clone()
            if trg_len < target.size(1):
                target[:, :-trg_len] = -100                              # only score new tokens
            next_target = target[:, 1:]                                  # [1, W-1]

            # response mask sliced to the window and shifted to align with next_target
            window_resp_mask = resp_mask[:, begin_loc:end_loc]           # [1, W]
            resp_valid = window_resp_mask[:, 1:]                         # [1, W-1]

            # only count tokens that are BOTH new and belong to the response
            valid_new = (next_target != -100) & resp_valid

            if valid_new.any():
                next_target_clamped = next_target.clone()
                next_target_clamped[next_target_clamped == -100] = 0

                logprobs = _safe_logsoftmax_fp32(next_logits)
                gathered = logprobs.gather(-1, next_target_clamped.unsqueeze(-1)).squeeze(-1)
                sum_logp += gathered[valid_new].sum().item()

                top2 = next_logits.topk(2, dim=-1).values.float()
                gaps = (top2[..., 0] - top2[..., 1])[valid_new]
                sum_gap += gaps.sum().item()
                count += int(valid_new.sum().item())

            prev_end = end_loc

    avg_logp = (sum_logp / count) if count else 0.0
    ppl = float(torch.exp(torch.tensor(-avg_logp)).item()) if count else float('inf')
    avg_gap = (sum_gap / count) if count else 0.0
    char_count = len(response_text)
    avg_logp_per_char = (sum_logp / char_count) if char_count > 0 else 0.0

    return {
        "avg_logp": avg_logp,
        "perplexity": ppl,
        "avg_logit_gap": avg_gap,
        "token_count": int(Lr),
        "char_count": char_count,
        "avg_logp_per_char": avg_logp_per_char,
    }



def score_and_detect(mainfolder, output_root="confidence_results_RQ1point2_newresults"):
    os.makedirs(output_root, exist_ok=True)
    model_dirs = [d for d in os.listdir(mainfolder) if d in ollama_to_hf]

    for model_dir in model_dirs:
        hf_model = ollama_to_hf[model_dir]
        print(f"\nLoading model '{model_dir}' ({hf_model}) ...")

        # Clear memory before loading new model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        try:
            tokenizer = AutoTokenizer.from_pretrained(hf_model, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(
                hf_model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
                low_cpu_mem_usage=True,
                offload_folder="./offload",
                offload_state_dict=True,
            )
        except Exception as e:
            print(f"Error loading model {hf_model}: {e}")
            continue

        model.eval()
        try:
            model.config.use_cache = False
        except Exception:
            pass
        model_output_dir = os.path.join(output_root, model_dir)
        os.makedirs(model_output_dir, exist_ok=True)

        csv_files = glob.glob(os.path.join(mainfolder, model_dir, "*.csv"))
        if not csv_files:
            print(f"No CSV files found for model {model_dir}, skipping.")
            del model, tokenizer
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

        # Accumulate rows per language across ALL files for this model
        lang_buckets = defaultdict(list)

        for csv_path in csv_files:
            filename = os.path.basename(csv_path)
            lang = extract_lang_from_filename(filename)
            print(f"Processing {csv_path} -> language: {lang}")

            df = pd.read_csv(csv_path)

            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{model_dir}-{lang}", leave=False):
                prompt_text = row.get('input_prompt', '')   # <-- use full prompt
                response_text = row.get('response', '')

                # CONDITIONAL confidence in the answer:
                response_scores = score_response_given_prompt(
                    model, tokenizer,
                    prompt_text=prompt_text if isinstance(prompt_text, str) else str(prompt_text),
                    response_text=response_text if isinstance(response_text, str) else str(response_text),
                )

                # Language detection for both fields
                input_lang_label = get_language_label(prompt_text if isinstance(prompt_text, str) else "")
                response_lang_label = get_language_label(response_text if isinstance(response_text, str) else "")

                rec = row.to_dict()
                rec.update({
                    "response_avg_logp": response_scores["avg_logp"],
                    "response_perplexity": response_scores["perplexity"],
                    "response_avg_logit_gap": response_scores["avg_logit_gap"],
                    "response_token_count": response_scores["token_count"],
                    "response_char_count": response_scores["char_count"],
                    "response_avg_logp_per_char": response_scores["avg_logp_per_char"],
                    "input_language_label": input_lang_label,
                    "response_language_label": response_lang_label,
                    "detected_lang_from_filename": lang,
                })
                lang_buckets[lang].append(rec)

        # Write ONE CSV per language for this model
        for lang, rows in lang_buckets.items():
            out_df = pd.DataFrame(rows)
            out_csv_path = os.path.join(model_output_dir, f"confidence_{lang}.csv")
            out_df.to_csv(out_csv_path, index=False)
            print(f"Saved {len(rows)} rows -> {out_csv_path}")

        # Clean up to free memory
        del model, tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Finished processing model '{model_dir}'. Memory cleared.\n")

    print(f"All models processed. Results saved under '{output_root}/'.")

if __name__ == "__main__":
    mainfolder = r'MMLU_results/spesific_results/RQ1_Results(diff_data)'
    score_and_detect(mainfolder)
