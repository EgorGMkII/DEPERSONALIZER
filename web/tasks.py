"""Memory-bounded Celery task for PDF anonymization."""

import os
import resource
import shutil
from pathlib import Path
from typing import Any

import fitz
from pdf2image import convert_from_path
from PIL import Image

from clean_pdf import clean_pdf
from src.masker import PageMasker
from src.ocr import OCRProcessor
from src.pii import PIIDetector

from .celery_app import celery_app
from .services import get_job_paths, save_job


_ocr_processor: OCRProcessor | None = None
_pii_detector: PIIDetector | None = None
_page_masker: PageMasker | None = None


def _log_stage(job_id: str, message: str) -> None:
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[Job {job_id[:8]}] {message}; peak_rss={peak_mb:.0f} MiB", flush=True)


def get_ocr_processor() -> OCRProcessor:
    global _ocr_processor
    if _ocr_processor is None:
        _ocr_processor = OCRProcessor()
    return _ocr_processor


def get_pii_detector() -> PIIDetector:
    global _pii_detector
    if _pii_detector is None:
        _pii_detector = PIIDetector()
    return _pii_detector


def get_page_masker() -> PageMasker:
    global _page_masker
    if _page_masker is None:
        _page_masker = PageMasker(padding_px=2)
    return _page_masker


def _page_count(pdf_path: Path) -> int:
    with fitz.open(pdf_path) as document:
        return document.page_count


def _render_page(pdf_path: Path, work_dir: Path, page_num: int, dpi: int) -> Path:
    rendered = convert_from_path(
        pdf_path,
        dpi=dpi,
        first_page=page_num,
        last_page=page_num,
        fmt="png",
        output_folder=work_dir,
        paths_only=True,
        thread_count=1,
    )
    if len(rendered) != 1:
        raise RuntimeError(f"Failed to render PDF page {page_num}")
    return Path(rendered[0])


def _save_pdf(page_paths: list[Path], output_path: Path, dpi: int) -> None:
    if not page_paths:
        raise RuntimeError("PDF contains no rendered pages")

    images: list[Image.Image] = []
    try:
        for page_path in page_paths:
            images.append(Image.open(page_path))
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            resolution=dpi,
        )
    finally:
        for image in images:
            image.close()


@celery_app.task(name="web.tasks.anonymize_pdf_task", bind=True)
def anonymize_pdf_task(self: Any, job_id: str, dpi: int | None = None) -> dict[str, Any]:
    """Process one job; all filesystem paths are derived from its validated UUID."""
    dpi = dpi or int(os.getenv("OCR_DPI", "144"))
    paths = get_job_paths(job_id)

    if paths.output_pdf.is_file():
        job = save_job(job_id, status="completed", percent=100, error=None)
        return {
            "status": "completed",
            "masked_tokens_count": job.get("masked_tokens_count", 0),
        }

    if not paths.input_pdf.is_file():
        save_job(job_id, status="failed", error="Input PDF is missing")
        raise FileNotFoundError("Input PDF is missing")

    paths.output_tmp_pdf.unlink(missing_ok=True)
    if paths.work_dir.exists():
        shutil.rmtree(paths.work_dir)
    paths.work_dir.mkdir(parents=True)

    try:
        clean_tmp = paths.root / "clean.tmp.pdf"
        clean_tmp.unlink(missing_ok=True)
        clean_pdf(str(paths.input_pdf), str(clean_tmp))
        os.replace(clean_tmp, paths.clean_pdf)

        total_pages = _page_count(paths.clean_pdf)
        if total_pages < 1:
            raise ValueError("PDF contains no pages")

        save_job(
            job_id,
            status="processing",
            current_page=0,
            total_pages=total_pages,
            percent=0,
            error=None,
        )

        _log_stage(job_id, f"initializing models for {total_pages} page(s) at {dpi} DPI")
        ocr = get_ocr_processor()
        pii = get_pii_detector()
        masker = get_page_masker()
        _log_stage(job_id, "models initialized")
        masked_page_paths: list[Path] = []
        total_masked_tokens = 0

        for page_num in range(1, total_pages + 1):
            _log_stage(job_id, f"page {page_num}/{total_pages}: rendering")
            source_path = _render_page(paths.clean_pdf, paths.work_dir, page_num, dpi)
            masked_path = paths.work_dir / f"masked_{page_num:06d}.png"

            with Image.open(source_path) as page_image:
                page_image.load()
                _log_stage(job_id, f"page {page_num}/{total_pages}: OCR started")
                lines = ocr.process_ocr_page(page_image, page_num, verbose=False)
                _log_stage(job_id, f"page {page_num}/{total_pages}: OCR finished")
                pii.detect_pii(lines)
                _log_stage(job_id, f"page {page_num}/{total_pages}: PII detection finished")
                total_masked_tokens += sum(
                    1 for line in lines for token in line.words if token.is_pii
                )
                masked_image = masker.mask_page(page_image, lines)

            try:
                masked_image.save(masked_path, format="PNG")
            finally:
                masked_image.close()

            source_path.unlink(missing_ok=True)
            masked_page_paths.append(masked_path)
            percent = int(page_num * 100 / total_pages)
            progress = {
                "current_page": page_num,
                "total_pages": total_pages,
                "percent": percent,
                "masked_tokens_count": total_masked_tokens,
            }
            save_job(job_id, status="processing", **progress)
            self.update_state(state="PROGRESS", meta=progress)
            _log_stage(job_id, f"page {page_num}/{total_pages}: saved")

        _log_stage(job_id, "assembling output PDF")
        _save_pdf(masked_page_paths, paths.output_tmp_pdf, dpi)
        os.replace(paths.output_tmp_pdf, paths.output_pdf)
        _log_stage(job_id, "output PDF saved")
        save_job(
            job_id,
            status="completed",
            current_page=total_pages,
            total_pages=total_pages,
            percent=100,
            masked_tokens_count=total_masked_tokens,
            error=None,
        )
        return {"status": "completed", "masked_tokens_count": total_masked_tokens}
    except Exception as exc:
        save_job(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        paths.output_tmp_pdf.unlink(missing_ok=True)
        (paths.root / "clean.tmp.pdf").unlink(missing_ok=True)
        if paths.work_dir.exists():
            shutil.rmtree(paths.work_dir, ignore_errors=True)
