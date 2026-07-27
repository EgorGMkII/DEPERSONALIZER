"""
Local PDF/Scan Anonymizer (PII Redactor) - Main CLI Entry Point.
Detects personal data on scanned PDFs using PaddleOCR + Natasha + RegEx,
masking them with black polygons directly on page images.
"""

import argparse
import os
import sys
from typing import List, Optional
from PIL import Image
from pdf2image import convert_from_path

from src.ocr import OCRProcessor
from src.pii import PIIDetector
from src.masker import PageMasker


def anonymize_pdf(
    input_path: str,
    output_path: str,
    dpi: int = 170,
    padding_px: int = 2,
    poppler_path: Optional[str] = None,
    verbose: bool = False,
    max_pages: Optional[int] = None,
    debug_dir: Optional[str] = None
) -> None:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not poppler_path:
        default_poppler_bin = r"C:\Users\egorg\Documents\poppler\poppler-24.02.0\Library\bin"
        if os.path.exists(os.path.join(default_poppler_bin, "pdfinfo.exe")):
            poppler_path = default_poppler_bin

    print(f"Converting PDF '{input_path}' to images (DPI={dpi})...")
    kwargs = {}
    if poppler_path:
        kwargs["poppler_path"] = poppler_path
    if max_pages and max_pages > 0:
        kwargs["first_page"] = 1
        kwargs["last_page"] = max_pages

    page_images = convert_from_path(input_path, dpi=dpi, **kwargs)
    print(f"Loaded {len(page_images)} page(s).")

    ocr_processor = OCRProcessor()
    pii_detector = PIIDetector()
    page_masker = PageMasker()

    redacted_images: List[Image.Image] = []

    for page_idx, page_img in enumerate(page_images, start=1):
        print(f"\n--- Processing Page {page_idx}/{len(page_images)} ---")
        
        # Stage 1: OCR Extraction
        tokens = ocr_processor.process_ocr_page(page_img, page_num=page_idx, verbose=verbose)
        if debug_dir:
            ocr_processor.save_stage1_debug(page_img, tokens, page_num=page_idx, debug_dir=debug_dir)

        # Stage 2: PII Classification
        pii_detector.detect_pii(tokens)
        if verbose:
            pii_texts = [f"{t.text} ({t.pii_reason})" for t in tokens if t.is_pii]
            print(f"Page {page_idx} detected PII tokens: {pii_texts}")
        if debug_dir:
            pii_detector.save_stage2_debug(page_img, tokens, page_num=page_idx, debug_dir=debug_dir)

        # Stage 3: Masking & Export
        masked_img = page_masker.mask_page(page_img, tokens, padding_px=padding_px)
        if debug_dir:
            page_masker.save_stage3_debug(masked_img, page_num=page_idx, debug_dir=debug_dir)

        redacted_images.append(masked_img)

    print(f"\nSaving anonymized PDF to '{output_path}'...")
    if redacted_images:
        redacted_images[0].save(
            output_path,
            save_all=True,
            append_images=redacted_images[1:],
            resolution=dpi
        )
    print("Anonymization complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Local PDF/Scan PII Redactor using PaddleOCR, Natasha, and RegEx"
    )
    parser.add_argument("--input", "-i", required=True, help="Path to input PDF file")
    parser.add_argument("--output", "-o", required=True, help="Path to save output anonymized PDF file")
    parser.add_argument("--dpi", type=int, default=170, help="DPI for PDF page rasterization (default: 170)")
    parser.add_argument("--padding", type=int, default=2, help="Padding in pixels for polygon masking (default: 2)")
    parser.add_argument("--max-pages", "-p", type=int, help="Limit processing to first N pages")
    parser.add_argument("--debug-dir", help="Directory to save intermediate 3-stage debug artifacts")
    parser.add_argument("--poppler-path", help="Path to Poppler bin directory (if not in PATH)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Print detailed PII detection debug info")

    args = parser.parse_args()

    try:
        anonymize_pdf(
            input_path=args.input,
            output_path=args.output,
            dpi=args.dpi,
            padding_px=args.padding,
            poppler_path=args.poppler_path,
            verbose=args.verbose,
            max_pages=args.max_pages,
            debug_dir=args.debug_dir
        )
    except Exception as e:
        print(f"Error during PDF anonymization: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
