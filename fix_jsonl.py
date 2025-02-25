import os
import json
import glob
import codecs

def fix_jsonl_encoding(input_folder):
    """
    Reads all JSONL files in the given folder, fixes Unicode escape sequences, and rewrites the files.
    """
    jsonl_files = glob.glob(os.path.join(input_folder, "*.jsonl"))

    for file_path in jsonl_files:
        fixed_data = []
        
        # Read the file and decode any escaped Unicode sequences
        with codecs.open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        # Load JSON while ensuring Unicode characters are interpreted correctly
                        data = json.loads(line)
                        fixed_data.append(data)
                    except json.JSONDecodeError as e:
                        print(f"[ERROR] Invalid JSON in {file_path}: {line.strip()} - {e}")

        # Write back to the file with proper UTF-8 encoding, ensuring no escape sequences
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in fixed_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')  # No escaping

        print(f"[INFO] Fixed encoding for {file_path}")

# Example usage:
fix_jsonl_encoding("./dataset")  # Replace with your actual folder path
