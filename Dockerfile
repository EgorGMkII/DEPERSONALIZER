# Use lightweight official Python 3.10 slim image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV OPENBLAS_NUM_THREADS=1
ENV NUMEXPR_NUM_THREADS=1
ENV PADDLE_CPU_THREADS=4
ENV PADDLE_ENABLE_MKLDNN=0
ENV MALLOC_ARENA_MAX=2

# Install required Linux system packages (poppler for pdf2image, libgl1/libglx-mesa0/libgomp1 for OpenCV & PaddleOCR)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libgl1 \
    libglx-mesa0 \
    libglib2.0-0 \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies (with multi-threaded Intel MKL PaddlePaddle CPU build)
COPY requirements.txt .
RUN pip install --no-cache-dir paddlepaddle -i https://www.paddlepaddle.org.cn/packages/stable/cpu/ && \
    pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY src/ ./src/
COPY web/ ./web/
COPY static/ ./static/
COPY app.py redact.py clean_pdf.py README.md ./

# Create jobs storage directory
RUN mkdir -p /app/jobs_data /app/debug_output

# Expose FastAPI port
EXPOSE 8000

# Health check endpoint (generous timeouts for heavy CPU OCR operations)
HEALTHCHECK --interval=60s --timeout=30s --start-period=60s --retries=10 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Run FastAPI app with Uvicorn
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
