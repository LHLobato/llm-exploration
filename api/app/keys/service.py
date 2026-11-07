"""Serviço de chaves de API (`HMAC-SHA256`), perfil `/v1/me` e estatísticas `/v1/usage`."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from app.core.config import Settings, get_settings
from app.core.exceptions import (
    AgeRestrictedError,
    ConflictError,
    ForbiddenError,
    NotFoundError,
    UnauthorizedError,
)
from app.core.security import (
    extract_api_key_prefix,
    generate_api_key,
    is_adult,
    verify_api_key_hash,
)
from app.db.models import ApiKeyRecord, UserProfile
from app.db.repository import DatabaseRepository


@dataclass(frozen=True)
class ValidatedKeyContext:
    """Contexto resolvido após validar uma chave de API (via X-API-Key ou sessão Web JWT)."""

    user: UserProfile
    api_key: ApiKeyRecord
    channel: str  # 'api' | 'web'


class ApiKeyService:
    """
    Gerencia o ciclo de vida das chaves de API conforme doc 04 e doc 05:
    - Exige maioridade (`>= 18` anos) na criação, rotação e validação.
    - Armazena apenas `key_hash = HMAC-SHA256(pepper, key)` e `key_prefix`.
    - Exibe o segredo completo `dmd_...` **uma única vez** na resposta `201` de criação/rotação.
    - Resolve e valida automaticamente a chave ativa quando o acesso vem da Web (JWT).
    """

    def __init__(
        self,
        repo: DatabaseRepository,
        settings: Settings | None = None,
    ) -> None:
        self._repo = repo
        self._settings = settings or get_settings()

    @staticmethod
    def _serialize_key_metadata(record: ApiKeyRecord) -> dict[str, Any]:
        """Serializa metadados públicos da chave (nunca inclui `key_hash` nem a chave em texto)."""
        return {
            "id": record.id,
            "name": record.name,
            "key_prefix": record.key_prefix,
            "status": record.status,
            "created_at": record.created_at.isoformat(),
            "last_used_at": record.last_used_at.isoformat() if record.last_used_at else None,
            "expires_at": record.expires_at.isoformat() if record.expires_at else None,
            "revoked_at": record.revoked_at.isoformat() if record.revoked_at else None,
        }

    async def get_me(self, profile: UserProfile) -> dict[str, Any]:
        """
        Retorna os dados do perfil e o estado atual da chave de API (`GET /v1/me` — doc 04/05/07).
        Estados possíveis de `key_status`: `active`, `none`, `revoked`, `age_restricted`.
        """
        adult = is_adult(profile.birth_date)
        keys = await self._repo.list_api_keys_for_user(profile.id)
        active_key = next((k for k in keys if k.status == "active"), None)

        if not adult:
            key_status = "age_restricted"
        elif active_key is not None:
            key_status = "active"
        elif any(k.status == "revoked" for k in keys):
            key_status = "revoked"
        else:
            key_status = "none"

        return {
            "id": profile.id,
            "username": profile.username,
            "first_name": profile.first_name,
            "last_name": profile.last_name,
            "full_name": profile.full_name,
            "birth_date": profile.birth_date.isoformat(),
            "email": profile.email,
            "is_adult": adult,
            "has_active_key": active_key is not None and adult,
            "can_generate_key": adult and active_key is None,
            "key_status": key_status,
            "key_id": active_key.id if active_key else None,
            "key_prefix": active_key.key_prefix if active_key else None,
            "last_used_at": (
                active_key.last_used_at.isoformat()
                if active_key and active_key.last_used_at
                else None
            ),
            "api_key": self._serialize_key_metadata(active_key) if active_key else None,
        }

    async def create_key(
        self,
        profile: UserProfile,
        name: str = "default",
    ) -> dict[str, Any]:
        """
        Gera uma nova chave de API para um usuário maior de 18 anos e retorna a chave em texto puro
        **uma única vez** (`shown_once: True`).
        """
        if not is_adult(profile.birth_date):
            raise AgeRestrictedError()

        existing_active = await self._repo.get_active_key_for_user(profile.id)
        if existing_active is not None:
            raise ConflictError(
                "active_key_exists",
                "Você já possui uma chave de API ativa. Use a opção de rotação ou revogue a chave atual.",
            )

        generated = generate_api_key(
            pepper=self._settings.api_key_pepper,
            prefix_tag=self._settings.api_key_prefix_tag,
        )
        record = await self._repo.create_api_key(
            user_id=profile.id,
            name=(name or "default").strip() or "default",
            key_prefix=generated.key_prefix,
            key_hash=generated.key_hash,
        )
        return {
            "id": record.id,
            "name": record.name,
            "key": generated.plaintext_key,
            "key_prefix": record.key_prefix,
            "status": record.status,
            "created_at": record.created_at.isoformat(),
            "shown_once": True,
        }

    async def list_keys(self, profile: UserProfile) -> list[dict[str, Any]]:
        """Lista o histórico de chaves do usuário (apenas metadados seguros, sem `key_hash`)."""
        records = await self._repo.list_api_keys_for_user(profile.id)
        return [self._serialize_key_metadata(r) for r in records]

    async def revoke_key(self, profile: UserProfile, key_id: str) -> dict[str, Any]:
        """Revoga imediatamente uma chave pertencente ao usuário."""
        revoked = await self._repo.revoke_api_key(key_id=key_id, user_id=profile.id)
        if revoked is None:
            raise NotFoundError("Chave de API não encontrada.")
        return self._serialize_key_metadata(revoked)

    async def rotate_key(
        self,
        profile: UserProfile,
        key_id: str,
        name: str = "default",
    ) -> dict[str, Any]:
        """Revoga a chave especificada e emite uma nova chave exibida uma única vez."""
        if not is_adult(profile.birth_date):
            raise AgeRestrictedError()

        target = await self._repo.get_api_key_by_id(key_id)
        if target is None or target.user_id != profile.id:
            raise NotFoundError("Chave de API não encontrada.")

        if target.status == "active":
            await self._repo.revoke_api_key(key_id=target.id, user_id=profile.id)

        # Se houver outra chave ativa residual, revoga para permitir a nova
        other_active = await self._repo.get_active_key_for_user(profile.id)
        if other_active is not None:
            await self._repo.revoke_api_key(key_id=other_active.id, user_id=profile.id)

        return await self.create_key(profile=profile, name=name or target.name)

    async def validate_api_key_secret(self, plaintext_key: str) -> ValidatedKeyContext:
        """
        Valida uma chave enviada via cabeçalho `X-API-Key` (`channel='api'`):
        1. Extrai `key_prefix` e valida formato `dmd_<8chars>_<secret>`.
        2. Busca candidatos pelo `key_prefix`.
        3. Compara `HMAC-SHA256` em tempo constante (`hmac.compare_digest`).
        4. Verifica `status == 'active'`, não expirada e dono `is_adult`.
        """
        prefix = extract_api_key_prefix(plaintext_key)
        if prefix is None:
            raise UnauthorizedError("Chave de API ausente ou em formato inválido.")

        candidates = await self._repo.find_api_keys_by_prefix(prefix)
        matched_key: ApiKeyRecord | None = None
        for candidate in candidates:
            if verify_api_key_hash(
                plaintext_key,
                candidate.key_hash,
                pepper=self._settings.api_key_pepper,
            ):
                matched_key = candidate
                break

        if matched_key is None:
            raise UnauthorizedError("Chave de API inválida.")

        now = datetime.now(timezone.utc)
        if matched_key.status != "active":
            raise ForbiddenError("api_key_revoked", "Esta chave de API foi revogada ou inativada.")
        if matched_key.expires_at is not None and matched_key.expires_at <= now:
            raise ForbiddenError("api_key_expired", "Esta chave de API expirou.")

        owner = await self._repo.get_profile_by_id(matched_key.user_id)
        if owner is None:
            raise UnauthorizedError("Proprietário da chave de API não encontrado.")
        if not is_adult(owner.birth_date):
            raise AgeRestrictedError()

        await self._repo.touch_api_key_last_used(matched_key.id, used_at=now)
        return ValidatedKeyContext(user=owner, api_key=matched_key, channel="api")

    async def resolve_active_key_for_web_user(self, profile: UserProfile) -> ValidatedKeyContext:
        """
        Resolve e valida automaticamente a chave ativa do usuário autenticado via JWT na Web
        (`channel='web'`, conforme Decisão D2 do doc 04 e doc 10).
        """
        if not is_adult(profile.birth_date):
            raise AgeRestrictedError()

        active_key = await self._repo.get_active_key_for_user(profile.id)
        if active_key is None:
            raise ForbiddenError(
                "no_active_api_key",
                "Você precisa gerar uma chave de API ativa no painel antes de realizar consultas.",
            )

        now = datetime.now(timezone.utc)
        if active_key.expires_at is not None and active_key.expires_at <= now:
            raise ForbiddenError("api_key_expired", "Sua chave de API ativa expirou.")

        await self._repo.touch_api_key_last_used(active_key.id, used_at=now)
        return ValidatedKeyContext(user=profile, api_key=active_key, channel="web")

    async def get_usage_summary(
        self, profile: UserProfile, limit: int = 50, offset: int = 0
    ) -> dict[str, Any]:
        """Retorna estatísticas de uso da chave ativa e o histórico paginado de consultas (`/v1/usage`)."""
        now = datetime.now(timezone.utc)
        active_key = await self._repo.get_active_key_for_user(profile.id)
        requests_last_minute = 0
        requests_today = 0

        if active_key is not None:
            requests_last_minute = await self._repo.count_requests_for_key_since(
                active_key.id, now - timedelta(minutes=1)
            )
            start_of_day = datetime(now.year, now.month, now.day, tzinfo=timezone.utc)
            requests_today = await self._repo.count_requests_for_key_since(
                active_key.id, start_of_day
            )

        history_records = await self._repo.list_api_requests_for_user(
            profile.id, limit=limit, offset=offset
        )
        latencies = [r.latency_ms for r in history_records if r.latency_ms is not None]
        avg_latency_ms = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

        return {
            "active_key_prefix": active_key.key_prefix if active_key else None,
            "quota": {
                "limit_per_minute": self._settings.rate_limit_per_minute,
                "used_last_minute": requests_last_minute,
                "remaining_minute": max(
                    0, self._settings.rate_limit_per_minute - requests_last_minute
                ),
                "limit_per_day": self._settings.rate_limit_per_day,
                "used_today": requests_today,
                "remaining_day": max(0, self._settings.rate_limit_per_day - requests_today),
            },
            "metrics": {
                "total_recent_requests": len(history_records),
                "malicious_count": sum(1 for r in history_records if r.verdict == "malicious"),
                "benign_count": sum(1 for r in history_records if r.verdict == "benign"),
                "avg_latency_ms": avg_latency_ms,
            },
            "history": [
                {
                    "id": r.id,
                    "analysis_id": r.analysis_id,
                    "domain": r.domain,
                    "verdict": r.verdict,
                    "score": r.score,
                    "channel": r.channel,
                    "cached": r.cached,
                    "status_code": r.status_code,
                    "latency_ms": r.latency_ms,
                    "created_at": r.created_at.isoformat(),
                }
                for r in history_records
            ],
        }
