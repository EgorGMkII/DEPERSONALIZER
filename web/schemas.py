"""
Pydantic Schemas for Depersonalizer Web Module.
"""

from typing import Optional
from pydantic import BaseModel


class AnonymizeResponse(BaseModel):
    job_id: str
    status: str
    message: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    masked_tokens_count: Optional[int] = 0
    error: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
