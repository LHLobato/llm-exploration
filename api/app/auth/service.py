"""Serviço de autenticação em 2 etapas (Senha + OTP por email) e emissão/validação de JWT."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any

from app.auth.schemas import SignupData, mask_email, validate_otp_token, validate_signup_payload
from app.core.config import Settings, get_settings
from app.core.exceptions import (
    AppError,
    ConflictError,
    RateLimitExceededError,
    UnauthorizedError,
)
from app.core.security import is_adult
from app.db.models import UserProfile
from app.db.repository import DatabaseRepository


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64url_decode(encoded: str) -> bytes:
    padding = "=" * (-len(encoded) % 4)
    return base64.urlsafe_b64decode((encoded + padding).encode("ascii"))


def create_jwt_token(
    user_id: str,
    email: str,
    secret: str,
    expires_in_seconds: int = 3600,
    token_type: str = "access",
) -> str:
    """Emite um JWT HS256 compatível com o payload do Supabase Auth (`sub`, `email`, `exp`)."""
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": user_id,
        "email": email,
        "role": "authenticated",
        "type": token_type,
        "iat": now,
        "exp": now + expires_in_seconds,
        "jti": str(uuid.uuid4()),
    }
    signing_input = (
        f"{_b64url_encode(json.dumps(header, separators=(',', ':')).encode('utf-8'))}."
        f"{_b64url_encode(json.dumps(payload, separators=(',', ':')).encode('utf-8'))}"
    )
    signature = hmac.new(
        secret.encode("utf-8"),
        signing_input.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return f"{signing_input}.{_b64url_encode(signature)}"


def decode_and_verify_jwt(token: str, secret: str) -> dict[str, Any]:
    """Valida a assinatura HS256 e expiração de um JWT e retorna suas claims."""
    if not token or token.count(".") != 2:
        raise UnauthorizedError("Token de sessão inválido.")
    header_b64, payload_b64, sig_b64 = token.split(".")
    signing_input = f"{header_b64}.{payload_b64}"
    expected_sig = hmac.new(
        secret.encode("utf-8"),
        signing_input.encode("ascii"),
        hashlib.sha256,
    ).digest()
    try:
        actual_sig = _b64url_decode(sig_b64)
        payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
    except Exception as exc:
        raise UnauthorizedError("Token de sessão malformado.") from exc

    if not hmac.compare_digest(actual_sig, expected_sig):
        raise UnauthorizedError("Assinatura do token inválida.")

    exp = int(payload.get("exp", 0))
    if exp <= int(time.time()):
        raise UnauthorizedError("Sessão expirada. Faça login novamente.")

    return payload


def _hash_password(password: str, salt: bytes | None = None) -> tuple[bytes, bytes]:
    salt_bytes = salt or secrets.token_bytes(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt_bytes, 120_000)
    return salt_bytes, derived


def _email_sha256(email: str, pepper: str) -> bytes:
    return hmac.new(
        pepper.encode("utf-8"),
        email.strip().lower().encode("utf-8"),
        hashlib.sha256,
    ).digest()


@dataclass
class PendingAuthAccount:
    """Credenciais de usuário em memória/auth store vinculadas ao perfil."""

    user_id: str
    username: str
    first_name: str
    last_name: str
    birth_date: date
    email: str
    password_salt: bytes
    password_hash: bytes
    confirmed: bool = False


@dataclass
class OtpChallenge:
    """Desafio OTP de 6 dígitos com validade curta, uso único e limite de tentativas."""

    email: str
    flow: str  # 'signup' | 'login' | 'reset'
    code_hash: bytes
    expires_at: float
    created_at: float
    attempts: int = 0
    max_attempts: int = 5
    consumed: bool = False


class AuthService:
    """
    Implementa os fluxos de autenticação do doc 03 e doc 05:
    - Cadastro + envio de OTP de 6 dígitos
    - Verificação de OTP de cadastro -> criação do `profiles` + emissão de sessão JWT
    - Login Etapa 1 (validação de email + senha **sem emitir sessão** + envio de OTP)
    - Login Etapa 2 (verificação de OTP -> emissão de sessão JWT)
    - Reenvio de OTP com cooldown de 60s
    - Recuperação de senha via OTP
    """

    OTP_TTL_SECONDS = 600
    OTP_RESEND_COOLDOWN_SECONDS = 60
    OTP_MAX_ATTEMPTS = 5

    def __init__(
        self,
        repo: DatabaseRepository,
        settings: Settings | None = None,
    ) -> None:
        self._repo = repo
        self._settings = settings or get_settings()
        self._accounts_by_email: dict[str, PendingAuthAccount] = {}
        self._challenges: dict[tuple[str, str], OtpChallenge] = {}
        self._revoked_tokens: set[str] = set()
        # Outbox de desenvolvimento/teste para inspecionar o último OTP emitido por email
        self._dev_otp_outbox: dict[str, str] = {}

    def get_last_otp_for_testing(self, email: str) -> str | None:
        """Apenas para testes automatizados e Mailpit local."""
        return self._dev_otp_outbox.get(email.strip().lower())

    def _generate_and_store_otp(self, email: str, flow: str) -> str:
        norm_email = email.strip().lower()
        code = f"{secrets.randbelow(1_000_000):06d}"
        now_ts = time.time()
        code_hash = _email_sha256(code, self._settings.api_key_pepper)
        self._challenges[(norm_email, flow)] = OtpChallenge(
            email=norm_email,
            flow=flow,
            code_hash=code_hash,
            expires_at=now_ts + self.OTP_TTL_SECONDS,
            created_at=now_ts,
            attempts=0,
            max_attempts=self.OTP_MAX_ATTEMPTS,
            consumed=False,
        )
        self._dev_otp_outbox[norm_email] = code
        return code

    async def signup(
        self, payload: dict[str, Any], ip_address: str | None = None
    ) -> dict[str, Any]:
        """Inicia cadastro, valida unicidade e dispara OTP de 6 dígitos (`202 Accepted`)."""
        data: SignupData = validate_signup_payload(payload)

        if await self._repo.get_profile_by_username(data.username) is not None:
            raise ConflictError("username_taken", "Este username já está em uso.")
        if await self._repo.get_profile_by_email(data.email) is not None:
            raise ConflictError("email_taken", "Este email já está cadastrado.")

        for acc in self._accounts_by_email.values():
            if acc.confirmed and acc.username.lower() == data.username.lower():
                raise ConflictError("username_taken", "Este username já está em uso.")

        salt, pwd_hash = _hash_password(data.password)
        user_id = str(uuid.uuid4())
        self._accounts_by_email[data.email] = PendingAuthAccount(
            user_id=user_id,
            username=data.username,
            first_name=data.first_name,
            last_name=data.last_name,
            birth_date=data.birth_date,
            email=data.email,
            password_salt=salt,
            password_hash=pwd_hash,
            confirmed=False,
        )

        self._generate_and_store_otp(data.email, flow="signup")
        await self._repo.log_otp_audit(
            user_id=user_id,
            email_hash=_email_sha256(data.email, self._settings.api_key_pepper),
            event_type="signup_sent",
            ip_address=ip_address,
        )
        return {
            "otp_required": True,
            "email": mask_email(data.email),
            "expires_in_seconds": self.OTP_TTL_SECONDS,
        }

    def _verify_challenge(self, email: str, token: str, flow: str) -> None:
        norm_email = email.strip().lower()
        clean_token = validate_otp_token(token)
        challenge = self._challenges.get((norm_email, flow))
        if challenge is None or challenge.consumed:
            raise UnauthorizedError("Código OTP inválido ou expirado.")

        if time.time() > challenge.expires_at:
            raise UnauthorizedError("Código OTP expirado. Solicite um novo código.")

        if challenge.attempts >= challenge.max_attempts:
            raise RateLimitExceededError(
                "Número máximo de tentativas de OTP excedido. Solicite um novo código.",
                retry_after=self.OTP_RESEND_COOLDOWN_SECONDS,
            )

        challenge.attempts += 1
        candidate_hash = _email_sha256(clean_token, self._settings.api_key_pepper)
        if not hmac.compare_digest(candidate_hash, challenge.code_hash):
            raise UnauthorizedError("Código OTP inválido.")

        challenge.consumed = True

    def _build_session_response(self, profile: UserProfile) -> dict[str, Any]:
        access_token = create_jwt_token(
            user_id=profile.id,
            email=profile.email,
            secret=self._settings.supabase_jwt_secret,
            expires_in_seconds=3600,
            token_type="access",
        )
        refresh_token = create_jwt_token(
            user_id=profile.id,
            email=profile.email,
            secret=self._settings.supabase_jwt_secret,
            expires_in_seconds=86400 * 7,
            token_type="refresh",
        )
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "user": {
                "id": profile.id,
                "username": profile.username,
                "first_name": profile.first_name,
                "last_name": profile.last_name,
                "full_name": profile.full_name,
                "birth_date": profile.birth_date.isoformat(),
                "email": profile.email,
                "is_adult": is_adult(profile.birth_date),
            },
        }

    async def verify_signup_otp(
        self, email: str, token: str, ip_address: str | None = None
    ) -> dict[str, Any]:
        """Valida o OTP de cadastro, cria `public.profiles` e emite a sessão inicial."""
        norm_email = str(email or "").strip().lower()
        account = self._accounts_by_email.get(norm_email)
        if account is None:
            raise UnauthorizedError("Código OTP inválido ou expirado.")

        try:
            self._verify_challenge(norm_email, token, flow="signup")
        except AppError:
            await self._repo.log_otp_audit(
                user_id=account.user_id,
                email_hash=_email_sha256(norm_email, self._settings.api_key_pepper),
                event_type="failed",
                ip_address=ip_address,
            )
            raise

        account.confirmed = True
        profile = await self._repo.get_profile_by_id(account.user_id)
        if profile is None:
            profile = await self._repo.create_profile(
                user_id=account.user_id,
                username=account.username,
                first_name=account.first_name,
                last_name=account.last_name,
                birth_date=account.birth_date,
                email=account.email,
            )

        await self._repo.log_otp_audit(
            user_id=profile.id,
            email_hash=_email_sha256(norm_email, self._settings.api_key_pepper),
            event_type="verified",
            ip_address=ip_address,
        )
        return self._build_session_response(profile)

    async def login_step1(
        self, email: str, password: str, ip_address: str | None = None
    ) -> dict[str, Any]:
        """
        Etapa 1 do login (doc 03): valida email + senha **sem emitir sessão** e envia OTP.
        Retorna mensagem genérica em falha para evitar enumeração de contas.
        """
        norm_email = str(email or "").strip().lower()
        account = self._accounts_by_email.get(norm_email)
        if account is None or not account.confirmed:
            # Simula derivação para equalizar tempo de resposta
            _hash_password(password or "dummy")
            raise UnauthorizedError("Credenciais inválidas.")

        _, candidate_hash = _hash_password(password or "", salt=account.password_salt)
        if not hmac.compare_digest(candidate_hash, account.password_hash):
            raise UnauthorizedError("Credenciais inválidas.")

        self._generate_and_store_otp(norm_email, flow="login")
        await self._repo.log_otp_audit(
            user_id=account.user_id,
            email_hash=_email_sha256(norm_email, self._settings.api_key_pepper),
            event_type="login_sent",
            ip_address=ip_address,
        )
        return {
            "otp_required": True,
            "expires_in_seconds": self.OTP_TTL_SECONDS,
        }

    async def verify_login_otp(
        self, email: str, token: str, ip_address: str | None = None
    ) -> dict[str, Any]:
        """Etapa 2 do login: valida o OTP de 6 dígitos e devolve a sessão (`access_token`)."""
        norm_email = str(email or "").strip().lower()
        account = self._accounts_by_email.get(norm_email)
        if account is None or not account.confirmed:
            raise UnauthorizedError("Código OTP inválido ou expirado.")

        try:
            self._verify_challenge(norm_email, token, flow="login")
        except AppError:
            await self._repo.log_otp_audit(
                user_id=account.user_id,
                email_hash=_email_sha256(norm_email, self._settings.api_key_pepper),
                event_type="failed",
                ip_address=ip_address,
            )
            raise

        profile = await self._repo.get_profile_by_id(account.user_id)
        if profile is None:
            raise UnauthorizedError("Perfil de usuário não encontrado.")

        await self._repo.log_otp_audit(
            user_id=profile.id,
            email_hash=_email_sha256(norm_email, self._settings.api_key_pepper),
            event_type="verified",
            ip_address=ip_address,
        )
        return self._build_session_response(profile)

    async def resend_otp(
        self, email: str, flow: str = "login", ip_address: str | None = None
    ) -> dict[str, Any]:
        """Reenvia código OTP respeitando o cooldown de 60 segundos."""
        norm_email = str(email or "").strip().lower()
        if flow not in {"signup", "login", "reset"}:
            raise AppError("invalid_flow", "Fluxo de OTP inválido.", 400)

        existing = self._challenges.get((norm_email, flow))
        now_ts = time.time()
        if existing is not None:
            elapsed = now_ts - existing.created_at
            if elapsed < self.OTP_RESEND_COOLDOWN_SECONDS:
                remaining = int(self.OTP_RESEND_COOLDOWN_SECONDS - elapsed) + 1
                raise RateLimitExceededError(
                    f"Aguarde {remaining}s antes de solicitar um novo código OTP.",
                    retry_after=remaining,
                )

        account = self._accounts_by_email.get(norm_email)
        if account is not None:
            self._generate_and_store_otp(norm_email, flow=flow)
            event_map = {
                "signup": "signup_sent",
                "login": "login_sent",
                "reset": "reset_sent",
            }
            await self._repo.log_otp_audit(
                user_id=account.user_id,
                email_hash=_email_sha256(norm_email, self._settings.api_key_pepper),
                event_type=event_map[flow],
                ip_address=ip_address,
            )
        return {
            "sent": True,
            "cooldown_seconds": self.OTP_RESEND_COOLDOWN_SECONDS,
        }

    async def request_password_reset(
        self, email: str, ip_address: str | None = None
    ) -> dict[str, Any]:
        """Inicia fluxo de recuperação de senha enviando OTP (sem revelar se email existe)."""
        norm_email = str(email or "").strip().lower()
        account = self._accounts_by_email.get(norm_email)
        if account is not None and account.confirmed:
            self._generate_and_store_otp(norm_email, flow="reset")
            await self._repo.log_otp_audit(
                user_id=account.user_id,
                email_hash=_email_sha256(norm_email, self._settings.api_key_pepper),
                event_type="reset_sent",
                ip_address=ip_address,
            )
        return {
            "otp_required": True,
            "expires_in_seconds": self.OTP_TTL_SECONDS,
        }

    async def verify_password_reset(
        self, email: str, token: str, new_password: str, ip_address: str | None = None
    ) -> dict[str, Any]:
        """Valida OTP de recuperação e atualiza a senha do usuário."""
        if len(new_password or "") < 12:
            raise AppError(
                "weak_password", "A nova senha deve conter no mínimo 12 caracteres.", 400
            )
        norm_email = str(email or "").strip().lower()
        account = self._accounts_by_email.get(norm_email)
        if account is None or not account.confirmed:
            raise UnauthorizedError("Código OTP inválido ou expirado.")

        self._verify_challenge(norm_email, token, flow="reset")
        salt, pwd_hash = _hash_password(new_password)
        account.password_salt = salt
        account.password_hash = pwd_hash

        await self._repo.log_otp_audit(
            user_id=account.user_id,
            email_hash=_email_sha256(norm_email, self._settings.api_key_pepper),
            event_type="verified",
            ip_address=ip_address,
        )
        return {"password_updated": True}

    def logout(self, token: str) -> None:
        """Invalida um token JWT na blocklist em memória."""
        if token:
            self._revoked_tokens.add(token)

    async def authenticate_jwt(self, token: str) -> UserProfile:
        """Valida o JWT Bearer e retorna o `UserProfile` correspondente."""
        if token in self._revoked_tokens:
            raise UnauthorizedError("Sessão encerrada. Faça login novamente.")
        claims = decode_and_verify_jwt(token, self._settings.supabase_jwt_secret)
        user_id = str(claims.get("sub", ""))
        if not user_id:
            raise UnauthorizedError("Token de sessão inválido.")
        profile = await self._repo.get_profile_by_id(user_id)
        if profile is None:
            raise UnauthorizedError("Usuário da sessão não encontrado.")
        return profile
