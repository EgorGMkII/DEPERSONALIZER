"""
Helper script to clean PDF annotations from any target PDF.
"""

import os
import sys
import fitz


def clean_pdf(input_path: str, output_path: str) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File '{input_path}' not found.")

    doc = fitz.open(input_path)
    print(f"Cleaning annotations from {len(doc)} page(s) in '{input_path}'...")

    for page_idx, page in enumerate(doc, start=1):
        annots = list(page.annots())
        if annots:
            for annot in annots:
                page.delete_annot(annot)
            print(f"Removed {len(annots)} annotation(s) on page {page_idx}")

    doc.save(output_path)
    doc.close()
    print(f"Saved cleaned PDF to '{output_path}'.")


if __name__ == "__main__":
    inp = sys.argv[1] if len(sys.argv) > 1 else "1.pdf"
    out = sys.argv[2] if len(sys.argv) > 2 else "clean_full.pdf"
    clean_pdf(inp, out)
