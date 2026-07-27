# Depersonalizer (PII Redactor)

Local Python tool for automated PII (Personally Identifiable Information) detection and redaction on scanned PDF documents and image scans.

## Features
- **Layout Preservation**: Redacts text using direct image polygon masking, preserving 100% of original document layout and graphics.
- **Word-Level Precision**: Proportional word-level bounding box splitting ensures only sensitive target words are masked, leaving clean text untouched.
- **Multi-Stage Detection**: Combines PaddleOCR text extraction, Natasha NER (PER/LOC), and specialized RegEx patterns for passports, phones, INN, emails, dates, and Russian names.
- **3-Stage Debug Output**: Generates stage-by-stage visual artifacts (`debug_output/`) for complete transparency and verification.

## Requirements
- Python >= 3.10
- Conda environment recommended
- Dependencies: `paddleocr`, `natasha`, `pdf2image`, `Pillow`, `numpy`, `PyMuPDF`

## Usage

Run PDF anonymization from the command line:

```bash
python redact.py --input input.pdf --output anonymized.pdf --dpi 170 --debug-dir debug_output --verbose
```

### Command Line Options
- `--input`, `-i`: Path to input PDF file (required).
- `--output`, `-o`: Path to save output anonymized PDF file (required).
- `--dpi`: DPI for PDF page rasterization (default: `170`).
- `--padding`: Pixel padding for black polygon mask expansion (default: `2`).
- `--max-pages`, `-p`: Limit processing to first N pages.
- `--debug-dir`: Directory path to save 3-stage intermediate debug artifacts (`01_ocr_raw.png`, `02_pii_highlight.png`, `02_pii_tokens.json`, `03_masked_page.png`).
- `--verbose`, `-v`: Print detailed PII detection log output.

## Architecture

```text
Depersonalizer/
├── redact.py                  # Main CLI entry point
└── src/
    ├── config.py              # Data structures (Token) and RegEx patterns
    ├── ocr.py                 # Stage 1: PaddleOCR extraction + Word BBox splitting
    ├── pii.py                 # Stage 2: Natasha NER + RegEx PII classification
    └── masker.py              # Stage 3: Pillow polygon masking & PDF export
```
