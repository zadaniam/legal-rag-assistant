"""Text cleaning utilities for extracted legal PDF content.

Removes page numbers, catchwords, hyphenation artifacts, and normalizes
whitespace across extracted text files.
"""

import os
import re


def run_cleaner(input_dir: str, output_file_path: str) -> None:
    """Clean extracted text files and write combined output to a single file."""
    print("Starting text cleaning process using Regex...")

    all_files = sorted([f for f in os.listdir(input_dir) if f.endswith(".txt")])

    if not all_files:
        print(f"Error: No .txt files found in folder {input_dir}")
        return

    combined_text_list = []

    for file_name in all_files:
        file_path = os.path.join(input_dir, file_name)
        with open(file_path, encoding="utf-8") as f:
            page_text = f.read()
            combined_text_list.append(page_text)

    full_text = "\n".join(combined_text_list)

    full_text = re.sub(r"^\s*-\s*\d+\s*-\s*$", "", full_text, flags=re.MULTILINE)

    full_text = re.sub(r"^.*(?:\.{2,}|…)\s*$", "", full_text, flags=re.MULTILINE)

    full_text = re.sub(r"(\w+)-\s*\n\s*(\w+)", r"\1\2", full_text)

    full_text = re.sub(r"\n{3,}", "\n\n", full_text)

    full_text = re.sub(r"(?<=[a-zA-Z,])\n(?=[a-z])", " ", full_text)

    full_text = re.sub(r"[ \t]+", " ", full_text)

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(full_text)

    print(f"Cleaning complete. Combined clean text file created at: {output_file_path}")
