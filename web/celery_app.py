"""Celery configuration kept free of OCR/model imports."""

import os

from celery import Celery


REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
VISIBILITY_TIMEOUT_SECONDS = int(os.getenv("VISIBILITY_TIMEOUT_SECONDS", "21600"))

celery_app = Celery(
    "depersonalizer",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["web.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    worker_concurrency=1,
    result_expires=86400,
    broker_connection_retry_on_startup=True,
    broker_transport_options={"visibility_timeout": VISIBILITY_TIMEOUT_SECONDS},
    result_backend_transport_options={"visibility_timeout": VISIBILITY_TIMEOUT_SECONDS},
)
