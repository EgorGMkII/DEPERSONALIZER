"""Lightweight job storage shared by the API and the Celery worker."""

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from redis import Redis


JOBS_DIR = Path(os.getenv("JOB_ROOT", "jobs_data")).resolve()
JOBS_DIR.mkdir(parents=True, exist_ok=True)
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379/0")
JOB_TTL_SECONDS = int(os.getenv("JOB_TTL_SECONDS", "86400"))

_redis = Redis.from_url(REDIS_URL, decode_responses=True)


@dataclass(frozen=True)
class JobPaths:
    root: Path
    input_pdf: Path
    clean_pdf: Path
    output_pdf: Path
    output_tmp_pdf: Path
    work_dir: Path


def normalize_job_id(job_id: str) -> str:
    """Accept only canonical UUIDs before constructing filesystem paths."""
    try:
        return str(UUID(job_id))
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError("Invalid job id") from exc


def get_job_paths(job_id: str, create: bool = False) -> JobPaths:
    job_id = normalize_job_id(job_id)
    root = JOBS_DIR / job_id
    if create:
        root.mkdir(parents=True, exist_ok=True)

    return JobPaths(
        root=root,
        input_pdf=root / "input.pdf",
        clean_pdf=root / "clean.pdf",
        output_pdf=root / "anonymized.pdf",
        output_tmp_pdf=root / "anonymized.tmp.pdf",
        work_dir=root / "work",
    )


def _job_key(job_id: str) -> str:
    return f"depersonalizer:job:{normalize_job_id(job_id)}"


def save_job(job_id: str, **fields: Any) -> dict[str, Any]:
    """Merge and atomically replace a compact JSON job record in Redis."""
    current = get_job(job_id) or {"job_id": normalize_job_id(job_id)}
    current.update(fields)
    current["updated_at"] = datetime.now(timezone.utc).isoformat()

    key = _job_key(job_id)
    _redis.set(key, json.dumps(current, ensure_ascii=False), ex=JOB_TTL_SECONDS)
    return current


def get_job(job_id: str) -> dict[str, Any] | None:
    raw = _redis.get(_job_key(job_id))
    if raw is None:
        return None
    return json.loads(raw)


def redis_is_ready() -> bool:
    try:
        return bool(_redis.ping())
    except Exception:
        return False
