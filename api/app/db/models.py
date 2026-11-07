"""Entidades de domínio persistidas no banco de dados (doc 02)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


@dataclass(frozen=True)
class UserProfile:
    """Representa um registro de `public.profiles`."""

    id: str
    username: str
    first_name: str
    last_name: str
    birth_date: date
    email: str
    created_at: datetime
    updated_at: datetime

    @property
    def full_name(self) -> str:
        return f"{self.first_name} {self.last_name}".strip()


@dataclass(frozen=True)
class ApiKeyRecord:
    """Representa um registro de `public.api_keys`."""

    id: str
    user_id: str
    name: str
    key_prefix: str
    key_hash: bytes
    status: str  # 'active' | 'revoked' | 'expired'
    created_at: datetime
    last_used_at: datetime | None = None
    expires_at: datetime | None = None
    revoked_at: datetime | None = None


@dataclass(frozen=True)
class DomainAnalysisRecord:
    """Representa um registro de `public.domain_analyses`."""

    id: str
    domain: str
    registered_domain: str
    label: str  # 'malicious' | 'benign'
    score: float
    threshold: float
    model_version: str
    model_scores: dict[str, float]
    features: dict[str, Any]
    structured_prompt: str
    evidence: dict[str, Any]
    partial: bool
    latency_ms: int
    created_at: datetime
    expires_at: datetime
    meta_weights: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ApiRequestRecord:
    """Representa um registro de auditoria e uso em `public.api_requests`."""

    id: str
    api_key_id: str
    user_id: str
    analysis_id: str | None
    channel: str  # 'web' | 'api'
    domain: str
    verdict: str | None
    score: float | None
    cached: bool
    status_code: int
    latency_ms: int
    created_at: datetime


@dataclass(frozen=True)
class OtpAuditRecord:
    """Representa um evento de auditoria em `public.otp_audit`."""

    id: str
    user_id: str | None
    email_hash: bytes
    event_type: str  # 'signup_sent' | 'login_sent' | 'reset_sent' | 'verified' | 'failed'
    ip_address: str | None
    created_at: datetime
