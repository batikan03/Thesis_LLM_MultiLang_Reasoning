import os
import pandas as pd

# Full source directory up to model folders
source_root = r"Most_recent_results_copy"

# Destination base where wrong predictions will be saved
target_root = r"MMLU_Most_recent_results_wrongpreds"

# Walk through all CSV files in all subdirectories
for root, _, files in os.walk(source_root):
    for file in files:
        if file.endswith(".csv"):
            source_path = os.path.join(root, file)

            # Compute path relative to source_root to retain full structure including model names
            relative_path = os.path.relpath(source_path, source_root)

            # Create corresponding path in the target root
            target_path = os.path.join(target_root, relative_path)

            try:
                df = pd.read_csv(source_path)

                if 'answer' in df.columns and 'evaluation' in df.columns:
                    valid_df = df.dropna(subset=['answer', 'evaluation'])

                    if not valid_df.empty:
                        answer_adjusted = valid_df['answer'].astype(int) + 1
                        prediction = valid_df['evaluation'].astype(int)

                        incorrect_mask = prediction != answer_adjusted
                        wrong_df = valid_df[incorrect_mask].copy()

                        if not wrong_df.empty:
                            wrong_df['expected_answer'] = answer_adjusted[incorrect_mask]
                            wrong_df['model_prediction'] = prediction[incorrect_mask]

                            # Create full folder path in target location
                            target_dir = os.path.dirname(target_path)
                            os.makedirs(target_dir, exist_ok=True)

                            # Save wrong predictions CSV
                            wrong_df.to_csv(target_path, index=False)
                            print(f"❌ Saved {len(wrong_df)} wrong predictions to: {target_path}")
                        else:
                            print(f"✅ No wrong predictions in: {relative_path}")
                    else:
                        print(f"Skipped {relative_path}: no valid rows.")
                else:
                    print(f"Skipped {relative_path}: missing 'answer' or 'evaluation' column.")
            except Exception as e:
                print(f"⚠️ Error processing {relative_path}: {e}")
