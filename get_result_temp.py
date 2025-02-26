#!/usr/bin/env python
import os
import json
import argparse
import glob
import pandas as pd

def load_reference(file_path):
    """
    Loads the reference JSONL file (which contains type information) and
    returns a dictionary mapping (question, answer) tuples to their type.
    """
    ref_dict = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Assume the reference item has 'question', 'answer', and 'type'
            q = item.get("question", "").strip()
            a = item.get("answer", "").strip()
            t = item.get("type", "").lower().strip()
            if q and a and t:
                ref_dict[(q, a)] = t
    return ref_dict

def process_eval_file(eval_file_path, ref_dict):
    """
    Processes a single evaluation JSONL file. For each eval item,
    matches the corresponding reference (by question and answer) to obtain the type.
    
    For items of type 'open' or 'fill', it aggregates the numerical rating (field 'eval_rating').
    For all other types, it aggregates the binary evaluation (field 'eval_result') as accuracy.
    
    Returns a dictionary mapping type -> aggregated score.
    """
    totals = {}   # Sum of ratings (or count of correct evaluations)
    counts = {}   # Number of items per type

    with open(eval_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            # In eval items, assume the fields 'question' and 'reference_answer' match those in the reference file.
            # (Alternatively, if the original 'answer' is preserved, adjust accordingly.)
            q = item.get("question", "").strip()
            a = item.get("reference_answer", "").strip()  # using 'reference_answer' from eval
            key = (q, a)
            t = ref_dict.get(key)
            if not t:
                # If matching by (question, answer) fails, try using just question (if that's acceptable)
                # t = ref_dict.get((q, ""))
                continue

            # For rating types ("open" and "fill")
            if t in ["open", "fill"]:
                rating = item.get("eval_rating")
                if rating is not None:
                    totals[t] = totals.get(t, 0) + rating
                    counts[t] = counts.get(t, 0) + 1
            else:
                # For binary evaluations (assumed to use 'eval_result')
                if "eval_result" in item:
                    result = item.get("eval_result")
                    totals[t] = totals.get(t, 0) + (1 if result else 0)
                    counts[t] = counts.get(t, 0) + 1

    # Compute aggregated scores.
    agg = {}
    for typ in totals:
        if counts[typ] > 0:
            if typ in ["open", "fill"]:
                # Average rating (expected between 1 and 5)
                agg[typ] = totals[typ] / counts[typ]
            else:
                # Accuracy percentage
                agg[typ] = totals[typ] / counts[typ] * 100
    return agg

def main(eval_folder, ref_folder, output_excel):
    """
    For each evaluation JSONL file in 'eval_folder', finds the corresponding reference JSONL file
    (with the same filename) in 'ref_folder'. Matches evaluation items with reference items
    to recover type information, then aggregates metrics by type.
    
    The results are saved into an Excel file with one row per file and one column per type.
    """
    eval_files = glob.glob(os.path.join(eval_folder, "*.jsonl"))
    results = []
    all_types = set()

    for eval_file in eval_files:
        filename = os.path.basename(eval_file)
        ref_file = os.path.join(ref_folder, filename)
        if not os.path.exists(ref_file):
            print(f"[WARN] Reference file not found for {filename}. Skipping.")
            continue

        ref_dict = load_reference(ref_file)
        agg = process_eval_file(eval_file, ref_dict)
        row = {"FileName": filename}
        row.update(agg)
        results.append(row)
        all_types.update(agg.keys())

    # Create a DataFrame with consistent columns: FileName plus one column per evaluation type.
    columns = ["FileName"] + sorted(all_types)
    df = pd.DataFrame(results, columns=columns)
    df.to_excel(output_excel, index=False)
    print(f"Results saved to {output_excel}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate evaluation metrics from eval JSONL files by using reference JSONL files (with type info). "
                    "For types 'open' and 'fill', the metric is average rating; for all other types, the metric is accuracy (%)."
    )
    parser.add_argument("eval_folder", type=str, help="Folder containing evaluation JSONL files")
    parser.add_argument("ref_folder", type=str, help="Folder containing reference JSONL files (with type info)")
    parser.add_argument("--output", type=str, default="results.xlsx", help="Output Excel file name")
    args = parser.parse_args()
    
    main(args.eval_folder, args.ref_folder, args.output)
