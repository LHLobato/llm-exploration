"""Repositório de persistência (PostgreSQL asyncpg e implementação In-Memory para testes/dev)."""

from __future__ import annotations

import json
import uuid
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from typing import Any, Protocol

from app.core.exceptions import ConflictError
from app.db.models import (
    ApiKeyRecord,
    ApiRequestRecord,
    DomainAnalysisRecord,
    OtpAuditRecord,
    UserProfile,
)


class DatabaseRepository(Protocol):
    """Interface de persistência de dados para perfis, chaves, análises, uso e auditoria OTP."""

    async def create_profile(
        self,
        *,
        user_id: str,
        username: str,
        first_name: str,
        last_name: str,
        birth_date: date,
        email: str,
    ) -> UserProfile: ...

    async def get_profile_by_id(self, user_id: str) -> UserProfile | None: ...

    async def get_profile_by_email(self, email: str) -> UserProfile | None: ...

    async def get_profile_by_username(self, username: str) -> UserProfile | None: ...

    async def create_api_key(
        self,
        *,
        user_id: str,
        name: str,
        key_prefix: str,
        key_hash: bytes,
        expires_at: datetime | None = None,
    ) -> ApiKeyRecord: ...

    async def get_active_key_for_user(self, user_id: str) -> ApiKeyRecord | None: ...

    async def get_api_key_by_id(self, key_id: str) -> ApiKeyRecord | None: ...

    async def list_api_keys_for_user(self, user_id: str) -> list[ApiKeyRecord]: ...

    async def find_api_keys_by_prefix(self, key_prefix: str) -> list[ApiKeyRecord]: ...

    async def revoke_api_key(self, key_id: str, user_id: str) -> ApiKeyRecord | None: ...

    async def touch_api_key_last_used(self, key_id: str, used_at: datetime | None = None) -> None: ...

    async def save_domain_analysis(self, analysis: DomainAnalysisRecord) -> DomainAnalysisRecord: ...

    async def get_cached_domain_analysis(
        self, domain: str, model_version: str, now: datetime | None = None
    ) -> DomainAnalysisRecord | None: ...

    async def get_domain_analysis_by_id(self, analysis_id: str) -> DomainAnalysisRecord | None: ...

    async def log_api_request(self, record: ApiRequestRecord) -> ApiRequestRecord: ...

    async def list_api_requests_for_user(
        self, user_id: str, limit: int = 50, offset: int = 0
    ) -> list[ApiRequestRecord]: ...

    async def count_requests_for_key_since(self, api_key_id: str, since: datetime) -> int: ...

    async def log_otp_audit(
        self,
        *,
        user_id: str | None,
        email_hash: bytes,
        event_type: str,
        ip_address: str | None = None,
    ) -> OtpAuditRecord: ...


class InMemoryRepository:
    """Implementação em memória fiel às constraints do PostgreSQL (`001_init_schema.sql`)."""

    def __init__(self) -> None:
        self._profiles: dict[str, UserProfile] = {}
        self._api_keys: dict[str, ApiKeyRecord] = {}
        self._analyses: dict[str, DomainAnalysisRecord] = {}
        self._requests: list[ApiRequestRecord] = []
        self._otp_audits: list[OtpAuditRecord] = []

    async def create_profile(
        self,
        *,
        user_id: str,
        username: str,
        first_name: str,
        last_name: str,
        birth_date: date,
        email: str,
    ) -> UserProfile:
        norm_username = username.strip().lower()
        norm_email = email.strip().lower()
        for existing in self._profiles.values():
            if existing.username.lower() == norm_username:
                raise ConflictError("username_taken", "Este username já está em uso.")
            if existing.email.lower() == norm_email:
                raise ConflictError("email_taken", "Este email já está cadastrado.")

        now = datetime.now(timezone.utc)
        profile = UserProfile(
            id=user_id,
            username=username.strip(),
            first_name=first_name.strip(),
            last_name=last_name.strip(),
            birth_date=birth_date,
            email=norm_email,
            created_at=now,
            updated_at=now,
        )
        self._profiles[user_id] = profile
        return profile

    async def get_profile_by_id(self, user_id: str) -> UserProfile | None:
        return self._profiles.get(user_id)

    async def get_profile_by_email(self, email: str) -> UserProfile | None:
        norm_email = email.strip().lower()
        for profile in self._profiles.values():
            if profile.email.lower() == norm_email:
                return profile
        return None

    async def get_profile_by_username(self, username: str) -> UserProfile | None:
        norm_username = username.strip().lower()
        for profile in self._profiles.values():
            if profile.username.lower() == norm_username:
                return profile
        return None

    async def create_api_key(
        self,
        *,
        user_id: str,
        name: str,
        key_prefix: str,
        key_hash: bytes,
        expires_at: datetime | None = None,
    ) -> ApiKeyRecord:
        existing_active = await self.get_active_key_for_user(user_id)
        if existing_active is not None:
            raise ConflictError(
                "active_key_exists",
                "O usuário já possui uma chave de API ativa. Use rotação ou revogue a atual.",
            )
        for k in self._api_keys.values():
            if k.key_hash == key_hash:
                raise ConflictError("duplicate_key_hash", "Conflito ao gerar chave de API.")

        now = datetime.now(timezone.utc)
        key_id = str(uuid.uuid4())
        record = ApiKeyRecord(
            id=key_id,
            user_id=user_id,
            name=name,
            key_prefix=key_prefix,
            key_hash=key_hash,
            status="active",
            created_at=now,
            last_used_at=None,
            expires_at=expires_at,
            revoked_at=None,
        )
        self._api_keys[key_id] = record
        return record

    async def get_active_key_for_user(self, user_id: str) -> ApiKeyRecord | None:
        for record in self._api_keys.values():
            if record.user_id == user_id and record.status == "active":
                return record
        return None

    async def get_api_key_by_id(self, key_id: str) -> ApiKeyRecord | None:
        return self._api_keys.get(key_id)

    async def list_api_keys_for_user(self, user_id: str) -> list[ApiKeyRecord]:
        items = [k for k in self._api_keys.values() if k.user_id == user_id]
        items.sort(key=lambda r: r.created_at, reverse=True)
        return items

    async def find_api_keys_by_prefix(self, key_prefix: str) -> list[ApiKeyRecord]:
        return [k for k in self._api_keys.values() if k.key_prefix == key_prefix]

    async def revoke_api_key(self, key_id: str, user_id: str) -> ApiKeyRecord | None:
        record = self._api_keys.get(key_id)
        if record is None or record.user_id != user_id:
            return None
        if record.status == "revoked":
            return record
        updated = replace(
            record,
            status="revoked",
            revoked_at=datetime.now(timezone.utc),
        )
        self._api_keys[key_id] = updated
        return updated

    async def touch_api_key_last_used(self, key_id: str, used_at: datetime | None = None) -> None:
        record = self._api_keys.get(key_id)
        if record is not None:
            self._api_keys[key_id] = replace(
                record, last_used_at=used_at or datetime.now(timezone.utc)
            )

    async def save_domain_analysis(self, analysis: DomainAnalysisRecord) -> DomainAnalysisRecord:
        self._analyses[analysis.id] = analysis
        return analysis

    async def get_cached_domain_analysis(
        self, domain: str, model_version: str, now: datetime | None = None
    ) -> DomainAnalysisRecord | None:
        ref = now or datetime.now(timezone.utc)
        candidates = [
            a
            for a in self._analyses.values()
            if a.domain == domain and a.model_version == model_version and a.expires_at > ref
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda a: a.created_at, reverse=True)
        return candidates[0]

    async def get_domain_analysis_by_id(self, analysis_id: str) -> DomainAnalysisRecord | None:
        return self._analyses.get(analysis_id)

    async def log_api_request(self, record: ApiRequestRecord) -> ApiRequestRecord:
        self._requests.append(record)
        return record

    async def list_api_requests_for_user(
        self, user_id: str, limit: int = 50, offset: int = 0
    ) -> list[ApiRequestRecord]:
        user_reqs = [r for r in self._requests if r.user_id == user_id]
        user_reqs.sort(key=lambda r: r.created_at, reverse=True)
        return user_reqs[offset : offset + limit]

    async def count_requests_for_key_since(self, api_key_id: str, since: datetime) -> int:
        return sum(
            1
            for r in self._requests
            if r.api_key_id == api_key_id and r.created_at >= since
        )

    async def log_otp_audit(
        self,
        *,
        user_id: str | None,
        email_hash: bytes,
        event_type: str,
        ip_address: str | None = None,
    ) -> OtpAuditRecord:
        record = OtpAuditRecord(
            id=str(uuid.uuid4()),
            user_id=user_id,
            email_hash=email_hash,
            event_type=event_type,
            ip_address=ip_address,
            created_at=datetime.now(timezone.utc),
        )
        self._otp_audits.append(record)
        return record


class PostgresRepository:
    """Implementação assíncrona em PostgreSQL usando `asyncpg.Pool`."""

    def __init__(self, pool: Any) -> None:
        self._pool = pool

    @staticmethod
    def _row_to_profile(row: Any) -> UserProfile:
        return UserProfile(
            id=str(row["id"]),
            username=str(row["username"]),
            first_name=str(row["first_name"]),
            last_name=str(row["last_name"]),
            birth_date=row["birth_date"],
            email=str(row["email"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _row_to_api_key(row: Any) -> ApiKeyRecord:
        return ApiKeyRecord(
            id=str(row["id"]),
            user_id=str(row["user_id"]),
            name=str(row["name"]),
            key_prefix=str(row["key_prefix"]),
            key_hash=bytes(row["key_hash"]),
            status=str(row["status"]),
            created_at=row["created_at"],
            last_used_at=row["last_used_at"],
            expires_at=row["expires_at"],
            revoked_at=row["revoked_at"],
        )

    async def create_profile(
        self,
        *,
        user_id: str,
        username: str,
        first_name: str,
        last_name: str,
        birth_date: date,
        email: str,
    ) -> UserProfile:
        query = """
            INSERT INTO public.profiles (id, username, first_name, last_name, birth_date, email)
            VALUES ($1::uuid, $2, $3, $4, $5, $6)
            RETURNING id, username, first_name, last_name, birth_date, email, created_at, updated_at
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                query, user_id, username.strip(), first_name.strip(), last_name.strip(), birth_date, email.strip().lower()
            )
            return self._row_to_profile(row)

    async def get_profile_by_id(self, user_id: str) -> UserProfile | None:
        query = """
            SELECT id, username, first_name, last_name, birth_date, email, created_at, updated_at
            FROM public.profiles WHERE id = $1::uuid
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, user_id)
            return self._row_to_profile(row) if row else None

    async def get_profile_by_email(self, email: str) -> UserProfile | None:
        query = """
            SELECT id, username, first_name, last_name, birth_date, email, created_at, updated_at
            FROM public.profiles WHERE email = $1
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, email.strip().lower())
            return self._row_to_profile(row) if row else None

    async def get_profile_by_username(self, username: str) -> UserProfile | None:
        query = """
            SELECT id, username, first_name, last_name, birth_date, email, created_at, updated_at
            FROM public.profiles WHERE username = $1
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, username.strip())
            return self._row_to_profile(row) if row else None

    async def create_api_key(
        self,
        *,
        user_id: str,
        name: str,
        key_prefix: str,
        key_hash: bytes,
        expires_at: datetime | None = None,
    ) -> ApiKeyRecord:
        query = """
            INSERT INTO public.api_keys (user_id, name, key_prefix, key_hash, status, expires_at)
            VALUES ($1::uuid, $2, $3, $4, 'active', $5)
            RETURNING id, user_id, name, key_prefix, key_hash, status, created_at, last_used_at, expires_at, revoked_at
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, user_id, name, key_prefix, key_hash, expires_at)
            return self._row_to_api_key(row)

    async def get_active_key_for_user(self, user_id: str) -> ApiKeyRecord | None:
        query = """
            SELECT id, user_id, name, key_prefix, key_hash, status, created_at, last_used_at, expires_at, revoked_at
            FROM public.api_keys
            WHERE user_id = $1::uuid AND status = 'active'
            LIMIT 1
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, user_id)
            return self._row_to_api_key(row) if row else None

    async def get_api_key_by_id(self, key_id: str) -> ApiKeyRecord | None:
        query = """
            SELECT id, user_id, name, key_prefix, key_hash, status, created_at, last_used_at, expires_at, revoked_at
            FROM public.api_keys WHERE id = $1::uuid
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, key_id)
            return self._row_to_api_key(row) if row else None

    async def list_api_keys_for_user(self, user_id: str) -> list[ApiKeyRecord]:
        query = """
            SELECT id, user_id, name, key_prefix, key_hash, status, created_at, last_used_at, expires_at, revoked_at
            FROM public.api_keys WHERE user_id = $1::uuid
            ORDER BY created_at DESC
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, user_id)
            return [self._row_to_api_key(r) for r in rows]

    async def find_api_keys_by_prefix(self, key_prefix: str) -> list[ApiKeyRecord]:
        query = """
            SELECT id, user_id, name, key_prefix, key_hash, status, created_at, last_used_at, expires_at, revoked_at
            FROM public.api_keys WHERE key_prefix = $1
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, key_prefix)
            return [self._row_to_api_key(r) for r in rows]

    async def revoke_api_key(self, key_id: str, user_id: str) -> ApiKeyRecord | None:
        query = """
            UPDATE public.api_keys
            SET status = 'revoked', revoked_at = COALESCE(revoked_at, now())
            WHERE id = $1::uuid AND user_id = $2::uuid
            RETURNING id, user_id, name, key_prefix, key_hash, status, created_at, last_used_at, expires_at, revoked_at
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, key_id, user_id)
            return self._row_to_api_key(row) if row else None

    async def touch_api_key_last_used(self, key_id: str, used_at: datetime | None = None) -> None:
        query = "UPDATE public.api_keys SET last_used_at = $2 WHERE id = $1::uuid"
        async with self._pool.acquire() as conn:
            await conn.execute(query, key_id, used_at or datetime.now(timezone.utc))

    async def save_domain_analysis(self, analysis: DomainAnalysisRecord) -> DomainAnalysisRecord:
        query = """
            INSERT INTO public.domain_analyses (
                id, domain, registered_domain, label, score, threshold, model_version,
                model_scores, features, structured_prompt, evidence, partial, latency_ms,
                created_at, expires_at
            )
            VALUES (
                $1::uuid, $2, $3, $4, $5, $6, $7,
                $8::jsonb, $9::jsonb, $10, $11::jsonb, $12, $13,
                $14, $15
            )
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                query,
                analysis.id,
                analysis.domain,
                analysis.registered_domain,
                analysis.label,
                analysis.score,
                analysis.threshold,
                analysis.model_version,
                json.dumps(analysis.model_scores),
                json.dumps(analysis.features),
                analysis.structured_prompt,
                json.dumps(analysis.evidence),
                analysis.partial,
                analysis.latency_ms,
                analysis.created_at,
                analysis.expires_at,
            )
        return analysis

    async def get_cached_domain_analysis(
        self, domain: str, model_version: str, now: datetime | None = None
    ) -> DomainAnalysisRecord | None:
        ref = now or datetime.now(timezone.utc)
        query = """
            SELECT id, domain, registered_domain, label, score, threshold, model_version,
                   model_scores, features, structured_prompt, evidence, partial, latency_ms,
                   created_at, expires_at
            FROM public.domain_analyses
            WHERE domain = $1 AND model_version = $2 AND expires_at > $3
            ORDER BY created_at DESC
            LIMIT 1
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, domain, model_version, ref)
            if not row:
                return None
            return DomainAnalysisRecord(
                id=str(row["id"]),
                domain=row["domain"],
                registered_domain=row["registered_domain"],
                label=row["label"],
                score=float(row["score"]),
                threshold=float(row["threshold"]),
                model_version=row["model_version"],
                model_scores=json.loads(row["model_scores"]) if isinstance(row["model_scores"], str) else dict(row["model_scores"]),
                features=json.loads(row["features"]) if isinstance(row["features"], str) else dict(row["features"]),
                structured_prompt=row["structured_prompt"],
                evidence=json.loads(row["evidence"]) if isinstance(row["evidence"], str) else dict(row["evidence"]),
                partial=bool(row["partial"]),
                latency_ms=int(row["latency_ms"] or 0),
                created_at=row["created_at"],
                expires_at=row["expires_at"],
            )

    async def get_domain_analysis_by_id(self, analysis_id: str) -> DomainAnalysisRecord | None:
        query = """
            SELECT id, domain, registered_domain, label, score, threshold, model_version,
                   model_scores, features, structured_prompt, evidence, partial, latency_ms,
                   created_at, expires_at
            FROM public.domain_analyses
            WHERE id = $1::uuid
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, analysis_id)
            if not row:
                return None
            return DomainAnalysisRecord(
                id=str(row["id"]),
                domain=row["domain"],
                registered_domain=row["registered_domain"],
                label=row["label"],
                score=float(row["score"]),
                threshold=float(row["threshold"]),
                model_version=row["model_version"],
                model_scores=json.loads(row["model_scores"]) if isinstance(row["model_scores"], str) else dict(row["model_scores"]),
                features=json.loads(row["features"]) if isinstance(row["features"], str) else dict(row["features"]),
                structured_prompt=row["structured_prompt"],
                evidence=json.loads(row["evidence"]) if isinstance(row["evidence"], str) else dict(row["evidence"]),
                partial=bool(row["partial"]),
                latency_ms=int(row["latency_ms"] or 0),
                created_at=row["created_at"],
                expires_at=row["expires_at"],
            )

    async def log_api_request(self, record: ApiRequestRecord) -> ApiRequestRecord:
        query = """
            INSERT INTO public.api_requests (
                id, api_key_id, user_id, analysis_id, channel, domain,
                verdict, score, cached, status_code, latency_ms, created_at
            )
            VALUES ($1::uuid, $2::uuid, $3::uuid, $4::uuid, $5, $6, $7, $8, $9, $10, $11, $12)
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                query,
                record.id,
                record.api_key_id,
                record.user_id,
                record.analysis_id,
                record.channel,
                record.domain,
                record.verdict,
                record.score,
                record.cached,
                record.status_code,
                record.latency_ms,
                record.created_at,
            )
        return record

    async def list_api_requests_for_user(
        self, user_id: str, limit: int = 50, offset: int = 0
    ) -> list[ApiRequestRecord]:
        query = """
            SELECT id, api_key_id, user_id, analysis_id, channel, domain,
                   verdict, score, cached, status_code, latency_ms, created_at
            FROM public.api_requests
            WHERE user_id = $1::uuid
            ORDER BY created_at DESC
            LIMIT $2 OFFSET $3
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(query, user_id, limit, offset)
            return [
                ApiRequestRecord(
                    id=str(r["id"]),
                    api_key_id=str(r["api_key_id"]),
                    user_id=str(r["user_id"]),
                    analysis_id=str(r["analysis_id"]) if r["analysis_id"] else None,
                    channel=str(r["channel"]),
                    domain=str(r["domain"]),
                    verdict=str(r["verdict"]) if r["verdict"] else None,
                    score=float(r["score"]) if r["score"] is not None else None,
                    cached=bool(r["cached"]),
                    status_code=int(r["status_code"]),
                    latency_ms=int(r["latency_ms"] or 0),
                    created_at=r["created_at"],
                )
                for r in rows
            ]

    async def count_requests_for_key_since(self, api_key_id: str, since: datetime) -> int:
        query = """
            SELECT count(*)::int AS cnt
            FROM public.api_requests
            WHERE api_key_id = $1::uuid AND created_at >= $2
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(query, api_key_id, since)
            return int(row["cnt"]) if row else 0

    async def log_otp_audit(
        self,
        *,
        user_id: str | None,
        email_hash: bytes,
        event_type: str,
        ip_address: str | None = None,
    ) -> OtpAuditRecord:
        record_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc)
        query = """
            INSERT INTO public.otp_audit (id, user_id, email_hash, event_type, ip_address, created_at)
            VALUES ($1::uuid, $2::uuid, $3, $4, $5::inet, $6)
        """
        async with self._pool.acquire() as conn:
            await conn.execute(query, record_id, user_id, email_hash, event_type, ip_address, now)
        return OtpAuditRecord(
            id=record_id,
            user_id=user_id,
            email_hash=email_hash,
            event_type=event_type,
            ip_address=ip_address,
            created_at=now,
        )
