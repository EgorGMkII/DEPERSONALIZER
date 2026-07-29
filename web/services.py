"""
Job Management & Background Task Services for Depersonalizer Web Module.
"""

import os
import json
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
JOBS_JSON = os.path.join(JOBS_DIR, "jobs.json")


def load_jobs_db() -> Dict[str, Dict[str, Any]]:
    """Loads jobs database from persistent JSON file."""
    if os.path.exists(JOBS_JSON):
        try:
            with open(JOBS_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_jobs_db() -> None:
    """Saves jobs database to persistent JSON file."""
    try:
        with open(JOBS_JSON, "w", encoding="utf-8") as f:
            json.dump(jobs_db, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


# Persistent database of anonymization jobs
jobs_db: Dict[str, Dict[str, Any]] = load_jobs_db()


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
        save_jobs_db()

        # 1. Clean PDF annotations
        clean_pdf_path = os.path.join(JOBS_DIR, f"{job_id}_clean.pdf")
        clean_pdf(input_pdf_path, clean_pdf_path)

        # 2. Convert PDF to PIL Images
        images = convert_from_path(clean_pdf_path, dpi=dpi)
        if not images:
            raise ValueError(f"No images extracted from PDF '{input_pdf_path}'.")

        print(f"Starting Web Job '{job_id}' ({len(images)} page(s))...")
        masked_images: List[Image.Image] = []
        total_masked_tokens = 0

        for page_idx, img in enumerate(images, start=1):
            print(f"[{job_id[:8]}] Page {page_idx}/{len(images)}: Running OCR (PaddleOCR)...")
            lines = ocr_processor.process_ocr_page(img, page_idx, verbose=False)

            print(f"[{job_id[:8]}] Page {page_idx}/{len(images)}: Running PII Detection (Natasha + RegEx)...")
            pii_detector.detect_pii(lines)

            # Count masked tokens
            page_masked = sum(1 for line in lines for w in line.words if w.is_pii)
            total_masked_tokens += page_masked

            # Stage 3: Masking
            masked_img = page_masker.mask_page(img, lines)
            masked_images.append(masked_img)
            print(f"[{job_id[:8]}] Page {page_idx}/{len(images)} complete ({page_masked} PII tokens masked).")

        # 3. Save anonymized multi-page PDF
        if masked_images:
            masked_images[0].save(
                output_pdf_path,
                save_all=True,
                append_images=masked_images[1:]
            )

        jobs_db[job_id]["status"] = "completed"
        jobs_db[job_id]["masked_tokens_count"] = total_masked_tokens
        save_jobs_db()
        print(f"Web Job '{job_id}' completed successfully! Total masked tokens: {total_masked_tokens}")

    except Exception as e:
        print(f"Error processing Web Job '{job_id}': {e}")
        jobs_db[job_id]["status"] = "failed"
        jobs_db[job_id]["error"] = str(e)
        save_jobs_db()
