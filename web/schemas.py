"""Pydantic response schemas for the web API."""

from typing import Optional

from pydantic import BaseModel


class AnonymizeResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    current_page: int = 0
    total_pages: int = 0
    percent: int = 0
    masked_tokens_count: int = 0
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    redis_ready: bool
