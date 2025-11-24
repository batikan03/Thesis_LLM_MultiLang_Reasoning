import os
import pandas as pd

# ----------------- CONFIG -----------------
# Point this to the RQ folder that contains MULTIPLE model folders
# e.g. r"confidence_results\new_results_200d\confidence_results_RQ2_newresults"
rq_root = r"confidence_results\new_results_200d\experiment_results\confidence_results_RQ3_newresults"

expected_total = 200  # target question count per file (for scaling)
save_csv = False       # set to False if you don't want CSV outputs
out_dir = os.path.join(rq_root, "_summaries")
# ------------------------------------------

os.makedirs(out_dir, exist_ok=True)

file_rows = []  # per-file results

for model_name in os.listdir(rq_root):
    model_path = os.path.join(rq_root, model_name)
    if not os.path.isdir(model_path):
        continue  # skip files in the RQ root

    for filename in os.listdir(model_path):
        if not filename.endswith(".csv"):
            continue

        filepath = os.path.join(model_path, filename)
        try:
            df = pd.read_csv(filepath)

            if "answer" not in df.columns or "evaluation" not in df.columns:
                print(f"Skipped {model_name}/{filename}: missing 'answer' or 'evaluation'.")
                continue

            # Clean & keep valid rows
            tmp = df.copy()
            tmp["answer"] = pd.to_numeric(tmp["answer"], errors="coerce")
            tmp["evaluation"] = pd.to_numeric(tmp["evaluation"], errors="coerce")
            valid_df = tmp.dropna(subset=["answer", "evaluation"]).head(expected_total)

            actual_count = len(valid_df)
            if actual_count == 0:
                print(f"Skipped {model_name}/{filename}: no valid rows.")
                continue

            # Adjust to 1-based answers; predictions already expected 1..4
            answer_adjusted = valid_df["answer"].astype(int) + 1
            prediction = valid_df["evaluation"].astype(int)

            correct = (prediction == answer_adjusted).sum()

            # Scale to expected_total
            scaled_correct = (correct / actual_count) * expected_total
            scaled_accuracy_pct = round((scaled_correct / expected_total) * 100, 2)

            file_rows.append({
                "model": model_name,
                "filename": filename,
                "valid_rows": int(actual_count),
                "correct": int(correct),
                "scaled_correct": int(round(scaled_correct)),
                "accuracy_pct": scaled_accuracy_pct
            })

        except Exception as e:
            print(f"Error in {model_name}/{filename}: {e}")

# ----------------- BUILD DATAFRAMES -----------------
if not file_rows:
    print("\nNo valid results found in the RQ folder.")
else:
    files_df = pd.DataFrame(file_rows).sort_values(["model", "filename"]).reset_index(drop=True)

    # Per-model summary (sum over files, then percentage over 200 * num_files)
    model_summary = (
        files_df
        .groupby("model", as_index=False)
        .agg(
            files=("filename", "count"),
            total_valid_rows=("valid_rows", "sum"),
            total_correct=("correct", "sum"),
            total_scaled_correct=("scaled_correct", "sum")
        )
    )
    model_summary["max_scaled_total"] = model_summary["files"] * expected_total
    model_summary["model_accuracy_pct"] = (
        (model_summary["total_scaled_correct"] / model_summary["max_scaled_total"]) * 100
    ).round(2)

    # Overall RQ summary
    total_scaled_correct = model_summary["total_scaled_correct"].sum()
    total_all = model_summary["max_scaled_total"].sum()
    overall_accuracy_pct = round((total_scaled_correct / total_all) * 100, 2) if total_all else 0.0

    # ----------------- PRINT -----------------
    print("\n=== File-wise Scaled Accuracy (each file scaled to 200) ===")
    print(files_df.to_string(index=False))

    print("\n=== Per-Model Summary (aggregated over files) ===")
    print(model_summary[[
        "model", "files", "total_valid_rows", "total_correct",
        "total_scaled_correct", "max_scaled_total", "model_accuracy_pct"
    ]].to_string(index=False))

    print(f"\n✅ Overall RQ Scaled Accuracy: {overall_accuracy_pct}% "
          f"({int(total_scaled_correct)}/{int(total_all)})")

    # ----------------- SAVE (optional) -----------------
    if save_csv:
        files_csv = os.path.join(out_dir, "files_accuracy.csv")
        models_csv = os.path.join(out_dir, "models_summary.csv")
        files_df.to_csv(files_csv, index=False)
        model_summary.to_csv(models_csv, index=False)
        print(f"\nSaved:\n - {files_csv}\n - {models_csv}")
