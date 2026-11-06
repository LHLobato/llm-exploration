"""Utilitários criptográficos e de segurança: chaves de API (HMAC-SHA256) e maioridade (doc 03/04)."""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import string
from dataclasses import dataclass
from datetime import date, datetime, timezone

from app.core.config import get_settings

_ALPHABET_ALNUM = string.ascii_letters + string.digits
_API_KEY_PATTERN = re.compile(r"^([a-z]{2,8}_[A-Za-z0-9]{8})_([A-Za-z0-9_-]{32,64})$")


@dataclass(frozen=True)
class GeneratedApiKey:
    """Representa uma chave recém-gerada antes do descarte do segredo em texto puro."""

    plaintext_key: str
    key_prefix: str
    key_hash: bytes


def generate_api_key(
    pepper: str | None = None,
    prefix_tag: str | None = None,
) -> GeneratedApiKey:
    """
    Gera uma nova chave de API de alta entropia no formato do doc 04:
    `dmd_<prefixo 8 chars>_<segredo 32 bytes base64url>`
    e calcula seu `HMAC-SHA256(pepper, chave_completa)`.
    """
    settings = get_settings()
    tag = prefix_tag or settings.api_key_prefix_tag
    random_prefix = "".join(secrets.choice(_ALPHABET_ALNUM) for _ in range(8))
    key_prefix = f"{tag}_{random_prefix}"
    secret_part = secrets.token_urlsafe(32)
    plaintext_key = f"{key_prefix}_{secret_part}"
    key_hash = hash_api_key(plaintext_key, pepper=pepper)
    return GeneratedApiKey(
        plaintext_key=plaintext_key,
        key_prefix=key_prefix,
        key_hash=key_hash,
    )


def hash_api_key(plaintext_key: str, pepper: str | None = None) -> bytes:
    """Calcula `HMAC-SHA256(API_KEY_PEPPER, chave_completa)` retornando bytes brutos (bytea)."""
    effective_pepper = pepper if pepper is not None else get_settings().api_key_pepper
    return hmac.new(
        effective_pepper.encode("utf-8"),
        plaintext_key.encode("utf-8"),
        hashlib.sha256,
    ).digest()


def verify_api_key_hash(
    plaintext_key: str,
    expected_hash: bytes,
    pepper: str | None = None,
) -> bool:
    """Compara a chave em texto puro contra o `expected_hash` em tempo constante."""
    computed = hash_api_key(plaintext_key, pepper=pepper)
    return hmac.compare_digest(computed, expected_hash)


def extract_api_key_prefix(plaintext_key: str) -> str | None:
    """
    Valida o formato estrutural da chave (`dmd_<8chars>_<secret>`) e retorna `key_prefix`
    (`dmd_<8chars>`), ou `None` se o formato for inválido.
    """
    if not plaintext_key or not isinstance(plaintext_key, str):
        return None
    match = _API_KEY_PATTERN.match(plaintext_key.strip())
    if not match:
        return None
    return match.group(1)


def calculate_age_years(birth_date: date, reference_date: date | None = None) -> int:
    """Calcula a idade completa em anos na data de referência (UTC por padrão)."""
    today = reference_date or datetime.now(timezone.utc).date()
    years = today.year - birth_date.year
    if (today.month, today.day) < (birth_date.month, birth_date.day):
        years -= 1
    return years


def is_adult(birth_date: date, reference_date: date | None = None, min_age: int | None = None) -> bool:
    """
    Verifica se o usuário possui pelo menos 18 anos completos (paridade com a função SQL
    `public.is_adult(birth_date)` do doc 02).
    """
    threshold = min_age if min_age is not None else get_settings().min_adult_age_years
    return calculate_age_years(birth_date, reference_date=reference_date) >= threshold
