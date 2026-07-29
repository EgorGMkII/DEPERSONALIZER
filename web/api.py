"""
FastAPI Web Module Entry Point & API Endpoints for Depersonalizer.
Serves static UI frontend at root '/' and REST API endpoints under '/api/v1/'.
"""

import os
import uuid
import shutil
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.ocr import OCRProcessor
from src.pii import PIIDetector
from src.masker import PageMasker

from .schemas import AnonymizeResponse, JobStatusResponse, HealthResponse
from .services import JOBS_DIR, jobs_db, save_jobs_db, execute_anonymization_job

STATIC_DIR = "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager: Preloads ML models once globally on web server startup."""
    print("Preloading ML models globally for Web Module (PaddleOCR & Natasha)...")
    app.state.ocr_processor = OCRProcessor()
    app.state.pii_detector = PIIDetector()
    app.state.page_masker = PageMasker(padding_px=2)
    print("ML models successfully initialized. Web Service ready!")
    yield
    print("Shutting down Web Service...")


app = FastAPI(
    title="Depersonalizer PII Redactor Web API",
    description="Decoupled Web Module for local PDF PII Anonymization",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health", response_model=HealthResponse, summary="Health Check")
def health_check():
    """Returns service health status and model initialization readiness."""
    ready = hasattr(app.state, "ocr_processor") and hasattr(app.state, "pii_detector")
    return HealthResponse(
        status="healthy" if ready else "initializing",
        models_loaded=ready
    )


@app.post("/api/v1/anonymize", response_model=AnonymizeResponse, summary="Submit PDF for PII Anonymization")
async def submit_anonymization(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Uploads a PDF file and starts background anonymization job."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    job_id = str(uuid.uuid4())
    job_dir = os.path.join(JOBS_DIR, job_id)
    os.makedirs(job_dir, exist_ok=True)

    input_pdf_path = os.path.join(job_dir, "input.pdf")
    output_pdf_path = os.path.join(job_dir, "anonymized.pdf")

    # Save uploaded PDF to job directory
    with open(input_pdf_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Register job state
    jobs_db[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "input_path": input_pdf_path,
        "output_path": output_pdf_path,
        "masked_tokens_count": 0,
        "error": None
    }
    save_jobs_db()

    # Dispatch background task
    background_tasks.add_task(
        execute_anonymization_job,
        job_id=job_id,
        input_pdf_path=input_pdf_path,
        output_pdf_path=output_pdf_path,
        ocr_processor=app.state.ocr_processor,
        pii_detector=app.state.pii_detector,
        page_masker=app.state.page_masker
    )

    return AnonymizeResponse(
        job_id=job_id,
        status="pending",
        message="PDF submitted successfully for background anonymization."
    )


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatusResponse, summary="Get Job Status")
def get_job_status(job_id: str):
    """Returns current status and token count of an anonymization job."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    job = jobs_db[job_id]
    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        masked_tokens_count=job.get("masked_tokens_count", 0),
        error=job.get("error")
    )


@app.get("/api/v1/jobs/{job_id}/download", summary="Download Anonymized PDF")
def download_anonymized_pdf(job_id: str):
    """Downloads anonymized PDF file if processing is completed."""
    if job_id not in jobs_db:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    job = jobs_db[job_id]
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job '{job_id}' is not completed yet (current status: '{job['status']}')."
        )

    output_path = job["output_path"]
    if not os.path.exists(output_path):
        raise HTTPException(status_code=500, detail="Anonymized output file missing.")

    return FileResponse(
        path=output_path,
        media_type="application/pdf",
        filename=f"anonymized_{job_id[:8]}.pdf"
    )


# Serve root index.html and static frontend assets
@app.get("/", include_in_schema=False)
def serve_index():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    raise HTTPException(status_code=404, detail="UI index.html missing.")


if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
