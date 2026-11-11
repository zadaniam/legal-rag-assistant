"""Extract raw text from legal PDF documents using pdfplumber."""

import os

import pdfplumber


def run_extractor(pdf_path: str, output_dir: str) -> None:
    """Extract text from each page of a PDF and save as individual text files.

    Crops header and footer regions from each page before extraction to
    remove recurring page furniture.
    """
    os.makedirs(output_dir, exist_ok=True)

    print("Starting coordinate-based PDF extraction...")
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            width, height = page.width, page.height

            bbox = (0, 170, width, height - 80)
            cropped_page = page.within_bbox(bbox)

            text = cropped_page.extract_text(
                layout=False, x_tolerance=1.5, y_tolerance=3
            )

            if not text:
                continue

            file_name = f"page_{str(page_num).zfill(3)}.txt"
            file_path = os.path.join(output_dir, file_name)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)

    print(f"Extraction complete. Check raw text per page in folder: {output_dir}")
