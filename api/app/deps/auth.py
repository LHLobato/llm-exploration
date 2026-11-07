"""Dependências FastAPI de autenticação JWT, validação dual (`X-API-Key` / Web JWT) e Rate Limit."""

from __future__ import annotations

import time
from collections import defaultdict, deque

from fastapi import Depends, Header, Request

from app.auth.service import AuthService
from app.core.config import get_settings
from app.core.exceptions import RateLimitExceededError, UnauthorizedError
from app.db.models import UserProfile
from app.keys.service import ApiKeyService, ValidatedKeyContext


class SlidingWindowRateLimiter:
    """Rate limiter em memória por janela deslizante para chaves de API e IPs."""

    def __init__(self) -> None:
        self._minute_buckets: dict[str, deque[float]] = defaultdict(deque)
        self._day_buckets: dict[str, deque[float]] = defaultdict(deque)

    def check_and_record(
        self,
        bucket_key: str,
        limit_per_minute: int,
        limit_per_day: int | None = None,
    ) -> dict[str, str]:
        now = time.time()
        min_q = self._minute_buckets[bucket_key]
        while min_q and now - min_q[0] >= 60.0:
            min_q.popleft()

        if len(min_q) >= limit_per_minute:
            retry_after = max(1, int(60.0 - (now - min_q[0])))
            raise RateLimitExceededError(
                "Limite de requisições por minuto excedido para esta chave.",
                retry_after=retry_after,
            )

        if limit_per_day is not None:
            day_q = self._day_buckets[bucket_key]
            while day_q and now - day_q[0] >= 86400.0:
                day_q.popleft()
            if len(day_q) >= limit_per_day:
                retry_after = max(1, int(86400.0 - (now - day_q[0])))
                raise RateLimitExceededError(
                    "Cota diária de requisições excedida para esta chave.",
                    retry_after=retry_after,
                )
            day_q.append(now)

        min_q.append(now)
        remaining = max(0, limit_per_minute - len(min_q))
        reset_epoch = int(now + 60)
        return {
            "X-RateLimit-Limit": str(limit_per_minute),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(reset_epoch),
        }


def get_auth_service(request: Request) -> AuthService:
    return request.app.state.auth_service


def get_api_key_service(request: Request) -> ApiKeyService:
    return request.app.state.api_key_service


def get_rate_limiter(request: Request) -> SlidingWindowRateLimiter:
    return request.app.state.rate_limiter


def _extract_bearer_token(authorization: str | None) -> str | None:
    if not authorization:
        return None
    parts = authorization.strip().split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip()
    return None


async def get_current_user(
    authorization: str | None = Header(default=None),
    auth_service: AuthService = Depends(get_auth_service),
) -> UserProfile:
    """Exige um JWT de sessão válido no header `Authorization: Bearer <jwt>`."""
    token = _extract_bearer_token(authorization)
    if not token:
        raise UnauthorizedError("Cabeçalho Authorization Bearer ausente.")
    return await auth_service.authenticate_jwt(token)


async def require_api_key_or_web_jwt(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    authorization: str | None = Header(default=None),
    auth_service: AuthService = Depends(get_auth_service),
    key_service: ApiKeyService = Depends(get_api_key_service),
    rate_limiter: SlidingWindowRateLimiter = Depends(get_rate_limiter),
) -> ValidatedKeyContext:
    """
    Dependência obrigatória para `/v1/domains/*` (doc 04 e doc 05):
    1. Se `X-API-Key` for enviado (ou `Authorization: Bearer dmd_...`), valida o segredo via
       `HMAC-SHA256` (`channel='api'`).
    2. Caso contrário, se `Authorization: Bearer <jwt>` for enviado pela SPA Web, valida a sessão
       e resolve/valida automaticamente a chave ativa do usuário (`channel='web'`).
    3. Aplica o rate limit por `api_key.id`.
    """
    settings = get_settings()
    bearer = _extract_bearer_token(authorization)

    api_key_secret = x_api_key.strip() if x_api_key else None
    if not api_key_secret and bearer and bearer.startswith(f"{settings.api_key_prefix_tag}_"):
        api_key_secret = bearer

    if api_key_secret:
        ctx = await key_service.validate_api_key_secret(api_key_secret)
    elif bearer:
        profile = await auth_service.authenticate_jwt(bearer)
        ctx = await key_service.resolve_active_key_for_web_user(profile)
    else:
        raise UnauthorizedError(
            "Informe o cabeçalho X-API-Key (para chamadas via API/curl) ou Authorization Bearer (na web)."
        )

    rate_limiter.check_and_record(
        bucket_key=f"key:{ctx.api_key.id}",
        limit_per_minute=settings.rate_limit_per_minute,
        limit_per_day=settings.rate_limit_per_day,
    )
    return ctx
