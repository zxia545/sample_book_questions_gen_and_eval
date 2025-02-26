import os
import json
import argparse
import pandas as pd

def process_file(file_path):
    """
    Process a JSONL file and aggregate evaluation metrics by question type.
    
    For rating types (fill/open), calculates the average rating.
    For binary types (judge, single-choice, multi-choice), calculates accuracy (percentage correct).
    
    Returns a dictionary mapping type -> aggregated score.
    """
    # Dictionaries to accumulate totals and counts per type.
    totals = {}   # For sum of ratings or count of correct answers.
    counts = {}   # For number of items per type.
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Determine the question type.
            # Prefer the explicit "type" field if available.
            q_type = item.get("type", "").lower().strip()
            # If no type is provided, infer based on which evaluation field exists.
            if not q_type:
                if "eval_rating" in item:
                    q_type = "fill/open"
                elif "eval_result" in item:
                    q_type = "judge"
                else:
                    q_type = "unknown"
            
            # Process rating evaluations (fill/open).
            if "eval_rating" in item:
                rating = item.get("eval_rating")
                if rating is not None:
                    totals[q_type] = totals.get(q_type, 0) + rating
                    counts[q_type] = counts.get(q_type, 0) + 1
            # Process binary evaluations (judge, single-choice, multi-choice, etc.)
            elif "eval_result" in item:
                # eval_result is expected to be boolean.
                result = item.get("eval_result")
                totals[q_type] = totals.get(q_type, 0) + (1 if result else 0)
                counts[q_type] = counts.get(q_type, 0) + 1

    # Compute aggregated score per type.
    agg = {}
    for t in totals:
        if counts[t] > 0:
            # For rating types, average the rating.
            if t in ["fill", "open", "fill/open", "rating"]:
                agg[t] = totals[t] / counts[t]
            else:
                # For binary types, calculate accuracy as a percentage.
                agg[t] = totals[t] / counts[t] * 100
    return agg

def main(folder_path, output_excel):
    # Get all JSONL files in the folder.
    jsonl_files = [os.path.join(folder_path, f)
                   for f in os.listdir(folder_path)
                   if f.lower().endswith(".jsonl")]

    results = []
    all_types = set()
    
    # Process each file.
    for file_path in jsonl_files:
        agg = process_file(file_path)
        row = {"FileName": os.path.basename(file_path)}
        row.update(agg)
        results.append(row)
        all_types.update(agg.keys())
    
    # Ensure consistent columns: first column is FileName, then one column per type.
    columns = ["FileName"] + sorted(all_types)
    df = pd.DataFrame(results, columns=columns)
    
    # Write results to an Excel file.
    df.to_excel(output_excel, index=False)
    print(f"Results written to {output_excel}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate evaluation metrics for each JSONL file in a folder and export to Excel. "
                    "For rating types (fill/open), average the score; for binary types, compute accuracy (%)."
    )
    parser.add_argument("folder", type=str, help="Folder containing JSONL files.")
    parser.add_argument("--output", type=str, default="results.xlsx", help="Output Excel file name.")
    args = parser.parse_args()
    
    main(args.folder, args.output)
