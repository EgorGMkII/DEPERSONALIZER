"""FastAPI endpoints. This module intentionally never imports OCR models."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .celery_app import celery_app
from .schemas import AnonymizeResponse, HealthResponse, JobStatusResponse
from .services import get_job, get_job_paths, redis_is_ready, save_job


STATIC_DIR = Path(os.getenv("STATIC_DIR", "static"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))
UPLOAD_CHUNK_BYTES = 1024 * 1024


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield


app = FastAPI(
    title="Depersonalizer PII Redactor Web API",
    description="Background PDF anonymization service",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse, summary="Health Check")
def health_check() -> HealthResponse:
    ready = redis_is_ready()
    return HealthResponse(status="healthy" if ready else "degraded", redis_ready=ready)


async def _save_upload(file: UploadFile, destination: Path) -> None:
    total = 0
    first_chunk = True

    try:
        with destination.open("xb") as output:
            while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                if first_chunk and not chunk.startswith(b"%PDF-"):
                    raise HTTPException(status_code=400, detail="Uploaded file is not a PDF.")
                first_chunk = False
                total += len(chunk)
                if total > MAX_UPLOAD_BYTES:
                    raise HTTPException(status_code=413, detail="PDF exceeds upload size limit.")
                output.write(chunk)
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        await file.close()

    if total == 0:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Uploaded PDF is empty.")


@app.post(
    "/api/v1/anonymize",
    response_model=AnonymizeResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Submit PDF for PII anonymization",
)
async def submit_anonymization(file: UploadFile = File(...)) -> AnonymizeResponse:
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    job_id = str(uuid4())
    paths = get_job_paths(job_id, create=True)
    await _save_upload(file, paths.input_pdf)

    save_job(
        job_id,
        status="queued",
        current_page=0,
        total_pages=0,
        percent=0,
        masked_tokens_count=0,
        error=None,
    )

    try:
        celery_app.send_task("web.tasks.anonymize_pdf_task", args=[job_id])
    except Exception as exc:
        save_job(job_id, status="failed", error="Task queue is unavailable")
        raise HTTPException(status_code=503, detail="Task queue is unavailable.") from exc

    return AnonymizeResponse(
        job_id=job_id,
        status="queued",
        message="PDF queued for background anonymization.",
    )


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse, summary="Get Job Status")
def get_job_status(job_id: str) -> JobStatusResponse:
    try:
        job = get_job(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobStatusResponse(**job)


@app.get("/api/v1/jobs/{job_id}/download", summary="Download Anonymized PDF")
def download_anonymized_pdf(job_id: str) -> FileResponse:
    try:
        job = get_job(job_id)
        paths = get_job_paths(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.get("status") != "completed":
        raise HTTPException(status_code=409, detail="Job is not completed.")
    if not paths.output_pdf.is_file():
        raise HTTPException(status_code=500, detail="Anonymized output file is missing.")

    return FileResponse(
        path=paths.output_pdf,
        media_type="application/pdf",
        filename=f"anonymized_{job_id[:8]}.pdf",
    )


@app.get("/api/v1/jobs/{job_id}/preview", summary="Preview Anonymized PDF")
def preview_anonymized_pdf(job_id: str) -> FileResponse:
    try:
        job = get_job(job_id)
        paths = get_job_paths(job_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc

    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job.get("status") != "completed":
        raise HTTPException(status_code=409, detail="Job is not completed.")
    if not paths.output_pdf.is_file():
        raise HTTPException(status_code=500, detail="Anonymized output file is missing.")

    return FileResponse(
        path=paths.output_pdf,
        media_type="application/pdf",
        filename=f"anonymized_{job_id[:8]}.pdf",
        content_disposition_type="inline",
    )


@app.get("/", include_in_schema=False)
def serve_index() -> FileResponse:
    index_path = STATIC_DIR / "index.html"
    if index_path.is_file():
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="UI index.html missing.")


if STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
