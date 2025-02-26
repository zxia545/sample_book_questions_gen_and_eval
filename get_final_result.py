#!/usr/bin/env python
import os
import json
import argparse
import glob
import pandas as pd

def process_file(file_path):
    """
    Process a single JSONL file to aggregate evaluation metrics by question type.
    
    For 'open' and 'fill' types (rating types), it calculates the average rating.
    For other types (binary True/False types), it calculates the accuracy percentage.
    
    Returns a dictionary mapping each type to its aggregated score.
    """
    totals = {}  # Sum of ratings or number of correct answers per type.
    counts = {}  # Count of items per type.
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue

            # Get the type (assumed to be present and lowercased)
            q_type = item.get("type", "").lower().strip()
            if not q_type:
                continue

            # For rating types (open, fill), use eval_rating.
            if q_type in ["open", "fill"]:
                rating = item.get("eval_rating")
                if rating is not None:
                    totals[q_type] = totals.get(q_type, 0) + rating
                    counts[q_type] = counts.get(q_type, 0) + 1
            # For all other types, use eval_result (assumed boolean).
            else:
                if "eval_result" in item:
                    result = item.get("eval_result")
                    totals[q_type] = totals.get(q_type, 0) + (1 if result else 0)
                    counts[q_type] = counts.get(q_type, 0) + 1

    # Calculate aggregated scores.
    agg = {}
    for t in totals:
        if counts[t] > 0:
            if t in ["open", "fill"]:
                # Average rating.
                agg[t] = totals[t] / counts[t]
            else:
                # Accuracy percentage.
                agg[t] = totals[t] / counts[t] * 100
    return agg

def main(folder_path, output_excel):
    """
    Searches for all JSONL files in the given folder, processes each file,
    and writes a summary Excel file where each row represents a file and each
    column corresponds to an evaluation type.
    """
    jsonl_files = glob.glob(os.path.join(folder_path, "*.jsonl"))
    results = []
    all_types = set()

    for file_path in jsonl_files:
        agg = process_file(file_path)
        row = {"FileName": os.path.basename(file_path)}
        row.update(agg)
        results.append(row)
        all_types.update(agg.keys())
    
    # Create a DataFrame with a consistent column order.
    columns = ["FileName"] + sorted(all_types)
    df = pd.DataFrame(results, columns=columns)
    
    # Write to Excel.
    df.to_excel(output_excel, index=False)
    print(f"Results saved to {output_excel}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate evaluation metrics for each JSONL file in a folder and export results to Excel. "
                    "For types 'open' and 'fill', the metric is the average rating. For all other types, the metric is the accuracy percentage."
    )
    parser.add_argument("folder", type=str, help="Folder containing JSONL files")
    parser.add_argument("--output", type=str, default="results.xlsx", help="Output Excel file name")
    args = parser.parse_args()
    
    main(args.folder, args.output)
