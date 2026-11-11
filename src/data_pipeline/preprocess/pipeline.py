"""Data preprocessing pipeline for legal documents.

Orchestrates the extraction, cleaning, and parsing of raw legal PDF
documents into Markdown format ready for ingestion.
"""

from pathlib import Path
from typing import cast

from src.config import pipeline_settings
from src.data_pipeline.preprocess import cleaner, extractor, parser


def run_preprocessing_pipeline() -> None:
    """Execute the full preprocessing pipeline: extract -> clean -> parse."""
    input_pdf: Path = cast(Path, pipeline_settings.input_pdf)
    extracted_dir = pipeline_settings.extracted_dir
    cleaned_file = pipeline_settings.cleaned_file
    parsed_md = pipeline_settings.parsed_md

    if not input_pdf.exists():
        print(f"Error: Source PDF file not found at: {input_pdf}")
        return

    print("\n--- [STEP 1/3] Extracting PDF Content ---")
    extractor.run_extractor(str(input_pdf), str(extracted_dir))

    print("\n--- [STEP 2/3] Cleaning Extracted Text ---")
    cleaner.run_cleaner(str(extracted_dir), str(cleaned_file))

    print("\n--- [STEP 3/3] Parsing Text to Markdown ---")
    parser.run_parser(str(cleaned_file), str(parsed_md))

    print("PREPROCESSING PIPELINE SUCCESSFULLY COMPLETED!")


if __name__ == "__main__":
    run_preprocessing_pipeline()
