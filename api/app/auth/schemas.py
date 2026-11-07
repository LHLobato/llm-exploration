"""Schemas de entrada/saída e validação do módulo de autenticação e OTP (doc 03 e doc 05)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from app.core.exceptions import AppError

_USERNAME_RE = re.compile(r"^[a-zA-Z0-9_]{3,30}$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_OTP_RE = re.compile(r"^\d{6}$")


def mask_email(email: str) -> str:
    """Mascara um endereço de email (ex.: `usuario@exemplo.com` -> `u***@exemplo.com`)."""
    clean = email.strip().lower()
    if "@" not in clean:
        return "***"
    local, domain = clean.split("@", 1)
    prefix = local[0] if local else "u"
    return f"{prefix}***@{domain}"


def validate_signup_payload(data: dict[str, Any]) -> SignupData:
    """Valida os campos obrigatórios de cadastro conforme regras do doc 03."""
    username = str(data.get("username", "")).strip()
    first_name = str(data.get("first_name", "")).strip()
    last_name = str(data.get("last_name", "")).strip()
    email = str(data.get("email", "")).strip().lower()
    password = str(data.get("password", ""))
    raw_birth = data.get("birth_date")

    if not _USERNAME_RE.match(username):
        raise AppError(
            "invalid_username",
            "O username deve ter entre 3 e 30 caracteres alfanuméricos ou underscore.",
            400,
        )
    if not first_name or not last_name:
        raise AppError("invalid_name", "Nome e sobrenome são obrigatórios.", 400)
    if not _EMAIL_RE.match(email):
        raise AppError("invalid_email", "Formato de email inválido.", 400)
    if len(password) < 12:
        raise AppError(
            "weak_password",
            "A senha deve conter no mínimo 12 caracteres.",
            400,
        )

    if isinstance(raw_birth, date):
        birth_date = raw_birth
    elif isinstance(raw_birth, str):
        try:
            birth_date = date.fromisoformat(raw_birth)
        except ValueError as exc:
            raise AppError(
                "invalid_birth_date",
                "Data de nascimento inválida (use o formato YYYY-MM-DD).",
                400,
            ) from exc
    else:
        raise AppError("invalid_birth_date", "Data de nascimento é obrigatória.", 400)

    today = datetime.now(timezone.utc).date()
    if birth_date < date(1900, 1, 1) or birth_date > today:
        raise AppError(
            "invalid_birth_date",
            "A data de nascimento informada está fora do intervalo permitido.",
            400,
        )

    return SignupData(
        username=username,
        first_name=first_name,
        last_name=last_name,
        birth_date=birth_date,
        email=email,
        password=password,
    )


def validate_otp_token(token: str) -> str:
    """Valida que o código OTP possui exatamente 6 dígitos numéricos."""
    clean = str(token or "").strip()
    if not _OTP_RE.match(clean):
        raise AppError("invalid_otp", "O código OTP deve conter exatamente 6 dígitos.", 400)
    return clean


@dataclass(frozen=True)
class SignupData:
    username: str
    first_name: str
    last_name: str
    birth_date: date
    email: str
    password: str
