"""
Job Management & Background Task Services for Depersonalizer Web Module.
"""

import os
import shutil
from typing import Dict, Any, List
from pdf2image import convert_from_path
from PIL import Image

from src.ocr import OCRProcessor
from src.pii import PIIDetector
from src.masker import PageMasker
from clean_pdf import clean_pdf

# Storage directory for web jobs
JOBS_DIR = "jobs_data"
os.makedirs(JOBS_DIR, exist_ok=True)

# In-memory database of active anonymization jobs
jobs_db: Dict[str, Dict[str, Any]] = {}


def execute_anonymization_job(
    job_id: str,
    input_pdf_path: str,
    output_pdf_path: str,
    ocr_processor: OCRProcessor,
    pii_detector: PIIDetector,
    page_masker: PageMasker,
    dpi: int = 170
) -> None:
    """Executes full PDF anonymization pipeline in a background task."""
    try:
        jobs_db[job_id]["status"] = "processing"

        # 1. Clean PDF annotations
        clean_pdf_path = os.path.join(JOBS_DIR, f"{job_id}_clean.pdf")
        clean_pdf(input_pdf_path, clean_pdf_path)

        # 2. Convert PDF to PIL Images
        images = convert_from_path(clean_pdf_path, dpi=dpi)
        if not images:
            raise ValueError(f"No images extracted from PDF '{input_pdf_path}'.")

        masked_images: List[Image.Image] = []
        total_masked_tokens = 0

        for page_idx, img in enumerate(images, start=1):
            # Stage 1: OCR
            lines = ocr_processor.process_ocr_page(img, page_idx, verbose=False)

            # Stage 2: PII Detection
            pii_detector.detect_pii(lines)

            # Count masked tokens
            page_masked = sum(1 for line in lines for w in line.words if w.is_pii)
            total_masked_tokens += page_masked

            # Stage 3: Masking
            masked_img = page_masker.mask_page(img, lines)
            masked_images.append(masked_img)

        # 3. Save anonymized multi-page PDF
        if masked_images:
            masked_images[0].save(
                output_pdf_path,
                save_all=True,
                append_images=masked_images[1:]
            )

        jobs_db[job_id]["status"] = "completed"
        jobs_db[job_id]["masked_tokens_count"] = total_masked_tokens
        print(f"Web Job '{job_id}' completed successfully. Masked tokens: {total_masked_tokens}")

    except Exception as e:
        print(f"Error processing Web Job '{job_id}': {e}")
        jobs_db[job_id]["status"] = "failed"
        jobs_db[job_id]["error"] = str(e)
