# Depersonalizer — Local PDF/Scan Anonymizer (PII Redactor)

Local Python service and CLI tool for detecting and masking Personal Identifiable Information (PII) on Russian scanned PDF documents using **PaddleOCR**, **Natasha NER**, **RegEx**, and **Pillow**.

Maintains **100% original page layout and graphics** by drawing targeted black mask polygons over PII text directly on page images.

---

## Features
- **Hierarchical 2-Level Pipeline (`LineContainer` -> `Token`)**: Preserves full line/sentence context for NER while applying exact word-level bounding box masks.
- **Natasha NER & RegEx Detection**: Detects Surnames, Names, Passports, SNILS, INN/OGRN, Emails, Dates, IPv4 Addresses, Local Phone Numbers, and Structured Postal Addresses.
- **Institution Header Safeguards**: Protects official state organ headers from false positive redaction.
- **POS Tagging Protection**: Protects verbs, prepositions, and conjunctions from accidental RegEx over-matching.
- **OCR Low-Confidence Thresholding (`confidence < 0.68`)**: Automatically masks handwritten text blocks and signatures without complex table parsing.
- **FastAPI Service**: Async REST API with background task execution (`fastapi.BackgroundTasks`) and job status polling.
- **Interactive Web UI**: Modern dark glassmorphism interface with drag & drop upload, progress spinner, live polling, scrollable PDF preview, and one-click download.
- **Docker & Docker Compose**: Full containerization support with system dependencies (`poppler-utils`, OpenGL) and model caching.

---

## Docker Deployment (Recommended)

### Using Docker Compose
```bash
docker-compose up --build -d
```
Open `http://localhost:8000/` in your browser.

### Using Docker Directly
```bash
docker build -t depersonalizer .
docker run -d -p 8000:8000 --name depersonalizer-container depersonalizer
```

---

## Local Installation & Launch

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Launch FastAPI Server & Web UI
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

---

## REST API Endpoints
- **`GET /health`**: Health check & ML model initialization status.
- **`POST /api/v1/anonymize`**: Upload a PDF file (`multipart/form-data`) -> Returns `{"job_id": "...", "status": "pending"}`.
- **`GET /api/v1/jobs/{job_id}`**: Poll job status (`pending`, `processing`, `completed`, `failed`) and masked token count.
- **`GET /api/v1/jobs/{job_id}/download`**: Download anonymized PDF file.

---

## CLI Usage

### Basic Anonymization Command
```bash
python redact.py --input input.pdf --output anonymized.pdf
```

### Advanced Multi-Page Command
```bash
python redact.py --input input.pdf --output anonymized.pdf --dpi 170 --debug-dir debug_output --verbose
```
