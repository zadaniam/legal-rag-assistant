"""Parse cleaned legal text into Markdown format with legal hierarchy."""

import os
import re


def run_parser(input_file_path: str, output_file_path: str) -> None:
    """Convert cleaned legal text into a single Markdown file.

    Applies regex transformations to format document title, BAB (chapter),
    and Pasal (article) headers into Markdown heading syntax.
    """
    print("Starting text parsing to single Markdown format...")

    if not os.path.exists(input_file_path):
        print(f"Error: Input file not found at {input_file_path}")
        return

    with open(input_file_path, encoding="utf-8") as f:
        text = f.read()

    text = re.sub(
        r"^(UNDANG-UNDANG REPUBLIK INDONESIA)$", r"# \1", text, flags=re.MULTILINE
    )

    text = re.sub(
        r"^BAB\s+([I|V|X|L]+)\s*\n\s*(.*?)$", r"## BAB \1 \2", text, flags=re.MULTILINE
    )

    text = re.sub(r"^(Pasal\s+\d+.*?)$", r"### \1", text, flags=re.MULTILINE)

    text = re.sub(r"^(\(\d+\))\s+", r"\1 ", text, flags=re.MULTILINE)
    text = re.sub(r"^(\d+\.)\s+", r"\1 ", text, flags=re.MULTILINE)

    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Parsing complete. Combined Markdown file created at: {output_file_path}")
