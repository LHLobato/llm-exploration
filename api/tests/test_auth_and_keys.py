"""Testes do Incremento 2: Autenticação em 2 etapas com OTP, Maioridade (18+) e Chaves HMAC-SHA256."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import uuid

import pytest

from app.auth.service import AuthService
from app.core.exceptions import (
    AgeRestrictedError,
    ConflictError,
    ForbiddenError,
    RateLimitExceededError,
    UnauthorizedError,
)
from app.db.models import ApiRequestRecord
from app.db.repository import InMemoryRepository
from app.keys.service import ApiKeyService


def _run(coro):
    return asyncio.run(coro)


def test_signup_otp_and_two_step_login_flow() -> None:
    async def scenario() -> None:
        repo = InMemoryRepository()
        auth = AuthService(repo=repo)

        # 1. Cadastro -> retorna 202 / otp_required
        signup_res = await auth.signup(
            {
                "username": "alice_sec",
                "first_name": "Alice",
                "last_name": "Silva",
                "birth_date": "1998-04-15",
                "email": "alice@exemplo.com",
                "password": "StrongPassword!123",
            }
        )
        assert signup_res["otp_required"] is True
        assert signup_res["email"] == "a***@exemplo.com"

        # Antes de confirmar OTP de cadastro, o perfil não existe e login falha
        assert await repo.get_profile_by_email("alice@exemplo.com") is None
        with pytest.raises(UnauthorizedError):
            await auth.login_step1("alice@exemplo.com", "StrongPassword!123")

        # 2. Verificação de OTP de cadastro -> cria perfil e emite sessão
        otp_code = auth.get_last_otp_for_testing("alice@exemplo.com")
        assert otp_code is not None and len(otp_code) == 6

        session_res = await auth.verify_signup_otp("alice@exemplo.com", otp_code)
        assert "access_token" in session_res
        assert session_res["user"]["username"] == "alice_sec"
        assert session_res["user"]["full_name"] == "Alice Silva"
        assert session_res["user"]["is_adult"] is True

        # Reuso do mesmo OTP deve falhar (uso único)
        with pytest.raises(UnauthorizedError):
            await auth.verify_signup_otp("alice@exemplo.com", otp_code)

        # 3. Login Etapa 1 (senha) -> não retorna access_token, exige OTP
        login_step1 = await auth.login_step1("alice@exemplo.com", "StrongPassword!123")
        assert login_step1 == {"otp_required": True, "expires_in_seconds": 600}

        # 4. Login Etapa 2 (OTP) -> retorna sessão JWT válida
        login_otp = auth.get_last_otp_for_testing("alice@exemplo.com")
        assert login_otp is not None
        login_session = await auth.verify_login_otp("alice@exemplo.com", login_otp)
        profile = await auth.authenticate_jwt(login_session["access_token"])
        assert profile.email == "alice@exemplo.com"

        # 5. Logout invalida o token JWT imediatamente
        auth.logout(login_session["access_token"])
        with pytest.raises(UnauthorizedError):
            await auth.authenticate_jwt(login_session["access_token"])

    _run(scenario())


def test_under_18_user_can_create_account_but_cannot_generate_api_key() -> None:
    async def scenario() -> None:
        repo = InMemoryRepository()
        auth = AuthService(repo=repo)
        keys = ApiKeyService(repo=repo)

        await auth.signup(
            {
                "username": "junior_dev",
                "first_name": "Junior",
                "last_name": "Santos",
                "birth_date": "2012-08-20",  # Menor de 18 anos
                "email": "junior@exemplo.com",
                "password": "StrongPassword!123",
            }
        )
        otp = auth.get_last_otp_for_testing("junior@exemplo.com")
        assert otp is not None
        session = await auth.verify_signup_otp("junior@exemplo.com", otp)
        assert session["user"]["is_adult"] is False

        profile = await auth.authenticate_jwt(session["access_token"])
        me_info = await keys.get_me(profile)
        assert me_info["is_adult"] is False
        assert me_info["key_status"] == "age_restricted"
        assert me_info["can_generate_key"] is False
        assert me_info["has_active_key"] is False

        with pytest.raises(AgeRestrictedError):
            await keys.create_key(profile)

        with pytest.raises(AgeRestrictedError):
            await keys.resolve_active_key_for_web_user(profile)

    _run(scenario())


def test_adult_api_key_lifecycle_one_time_display_rotation_and_revocation() -> None:
    async def scenario() -> None:
        repo = InMemoryRepository()
        auth = AuthService(repo=repo)
        keys = ApiKeyService(repo=repo)

        await auth.signup(
            {
                "username": "carlos_sec",
                "first_name": "Carlos",
                "last_name": "Mendes",
                "birth_date": "1990-01-10",
                "email": "carlos@exemplo.com",
                "password": "StrongPassword!123",
            }
        )
        otp = auth.get_last_otp_for_testing("carlos@exemplo.com")
        assert otp is not None
        session = await auth.verify_signup_otp("carlos@exemplo.com", otp)
        profile = await auth.authenticate_jwt(session["access_token"])

        # Estado inicial: sem chave ("none")
        me_before = await keys.get_me(profile)
        assert me_before["key_status"] == "none"
        assert me_before["can_generate_key"] is True

        # Web sem chave ativa deve bloquear consulta com erro 403 no_active_api_key
        with pytest.raises(ForbiddenError) as exc_info:
            await keys.resolve_active_key_for_web_user(profile)
        assert exc_info.value.code == "no_active_api_key"

        # 1. Gera chave -> exibida uma única vez
        created = await keys.create_key(profile, name="cli-prod")
        assert created["shown_once"] is True
        plaintext_key = created["key"]
        key_id = created["id"]
        assert plaintext_key.startswith(created["key_prefix"] + "_")

        # O banco guarda apenas key_hash (bytes), nunca a chave em texto puro
        stored_record = await repo.get_api_key_by_id(key_id)
        assert stored_record is not None
        assert stored_record.key_hash != plaintext_key.encode("utf-8")

        # 2. Segunda criação simultânea é bloqueada (1 chave ativa por usuário)
        with pytest.raises(ConflictError):
            await keys.create_key(profile)

        # 3. GET /v1/me e GET /v1/keys nunca retornam o segredo completo
        me_active = await keys.get_me(profile)
        assert me_active["key_status"] == "active"
        assert me_active["has_active_key"] is True
        assert me_active["key_prefix"] == created["key_prefix"]
        assert "key" not in (me_active["api_key"] or {})

        listed = await keys.list_keys(profile)
        assert len(listed) == 1
        assert "key" not in listed[0]
        assert "key_hash" not in listed[0]

        # 4. Validação via X-API-Key (curl) e resolução automática via Web JWT
        ctx_api = await keys.validate_api_key_secret(plaintext_key)
        assert ctx_api.channel == "api"
        assert ctx_api.api_key.id == key_id

        ctx_web = await keys.resolve_active_key_for_web_user(profile)
        assert ctx_web.channel == "web"
        assert ctx_web.api_key.id == key_id

        # 5. Rotação revoga a chave antiga e retorna uma nova chave uma única vez
        rotated = await keys.rotate_key(profile, key_id=key_id, name="rotated-key")
        new_plaintext_key = rotated["key"]
        new_key_id = rotated["id"]
        assert new_key_id != key_id
        assert rotated["shown_once"] is True

        # A chave antiga passa a ser rejeitada imediatamente (403 api_key_revoked)
        with pytest.raises(ForbiddenError) as revoked_exc:
            await keys.validate_api_key_secret(plaintext_key)
        assert revoked_exc.value.code == "api_key_revoked"

        # A chave nova funciona normalmente
        ctx_new = await keys.validate_api_key_secret(new_plaintext_key)
        assert ctx_new.api_key.id == new_key_id

        # 6. Revogação explícita + estatísticas de uso (/v1/usage)
        await repo.log_api_request(
            ApiRequestRecord(
                id=str(uuid.uuid4()),
                api_key_id=new_key_id,
                user_id=profile.id,
                analysis_id=None,
                channel="api",
                domain="exemplo.com.br",
                verdict="benign",
                score=0.05,
                cached=False,
                status_code=200,
                latency_ms=145,
                created_at=datetime.now(timezone.utc),
            )
        )
        usage = await keys.get_usage_summary(profile)
        assert usage["quota"]["used_today"] == 1
        assert usage["metrics"]["benign_count"] == 1
        assert len(usage["history"]) == 1

        await keys.revoke_key(profile, key_id=new_key_id)
        me_after_revoke = await keys.get_me(profile)
        assert me_after_revoke["key_status"] == "revoked"
        assert me_after_revoke["has_active_key"] is False

    _run(scenario())


def test_otp_resend_cooldown_is_enforced() -> None:
    async def scenario() -> None:
        repo = InMemoryRepository()
        auth = AuthService(repo=repo)
        await auth.signup(
            {
                "username": "cooldown_user",
                "first_name": "Ana",
                "last_name": "Costa",
                "birth_date": "1995-03-10",
                "email": "ana@exemplo.com",
                "password": "StrongPassword!123",
            }
        )
        # Reenvio imediato dentro dos 60s deve retornar 429 RateLimitExceededError
        with pytest.raises(RateLimitExceededError):
            await auth.resend_otp("ana@exemplo.com", flow="signup")

    _run(scenario())
