"""Testes unitários do Incremento 1: Fundação da API, Segurança HMAC-SHA256 e Maioridade."""

from __future__ import annotations

from datetime import date

import pytest

from app.core.config import get_settings
from app.core.exceptions import (
    AgeRestrictedError,
    InvalidDomainError,
    RateLimitExceededError,
    build_error_payload,
)
from app.core.logging import get_request_id, request_id_ctx
from app.core.security import (
    calculate_age_years,
    extract_api_key_prefix,
    generate_api_key,
    hash_api_key,
    is_adult,
    verify_api_key_hash,
)


def test_settings_defaults_match_architecture_spec() -> None:
    settings = get_settings()
    assert settings.api_v1_prefix == "/v1"
    assert settings.api_key_prefix_tag == "dmd"
    assert settings.min_adult_age_years == 18
    assert settings.decision_threshold == 0.5
    assert settings.model_version == "ensemble-v1-cnn42-mlp42-llama1b"


def test_error_classes_and_canonical_payload_match_doc_05() -> None:
    token = request_id_ctx.set("550e8400-e29b-41d4-a716-446655440000")
    try:
        err_domain = InvalidDomainError()
        assert err_domain.status_code == 400
        assert err_domain.code == "invalid_domain"

        payload = build_error_payload(err_domain.code, err_domain.message)
        assert payload == {
            "error": {
                "code": "invalid_domain",
                "message": (
                    "Informe apenas o nome do domínio (sem protocolo http/https, caminho ou porta)."
                ),
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
            }
        }

        err_age = AgeRestrictedError()
        assert err_age.status_code == 403
        assert err_age.code == "age_restricted"

        err_rl = RateLimitExceededError(retry_after=30)
        assert err_rl.status_code == 429
        assert err_rl.headers["Retry-After"] == "30"
    finally:
        request_id_ctx.reset(token)


def test_api_key_generation_and_hmac_sha256_verification() -> None:
    pepper = "test-pepper-secret-32-bytes-long!"
    generated = generate_api_key(pepper=pepper)

    assert generated.plaintext_key.startswith("dmd_")
    assert generated.key_prefix.startswith("dmd_")
    assert len(generated.key_prefix) == 12  # "dmd_" + 8 chars
    assert len(generated.key_hash) == 32  # SHA-256 digest bytes
    assert generated.plaintext_key.encode("utf-8") != generated.key_hash

    extracted_prefix = extract_api_key_prefix(generated.plaintext_key)
    assert extracted_prefix == generated.key_prefix

    assert verify_api_key_hash(generated.plaintext_key, generated.key_hash, pepper=pepper)
    assert not verify_api_key_hash(
        generated.plaintext_key + "x", generated.key_hash, pepper=pepper
    )
    assert not verify_api_key_hash(
        generated.plaintext_key, generated.key_hash, pepper="wrong-pepper"
    )
    assert hash_api_key(generated.plaintext_key, pepper=pepper) == generated.key_hash


def test_extract_api_key_prefix_rejects_malformed_keys() -> None:
    assert extract_api_key_prefix("") is None
    assert extract_api_key_prefix("invalid_key") is None
    assert extract_api_key_prefix("dmd_short_secret") is None


def test_is_adult_age_calculation_boundary_cases() -> None:
    ref = date(2026, 11, 6)

    # Exatamente 18 anos hoje -> adulto
    born_18_today = date(2008, 11, 6)
    assert calculate_age_years(born_18_today, reference_date=ref) == 18
    assert is_adult(born_18_today, reference_date=ref) is True

    # Faz 18 anos amanhã -> menor (17 anos)
    born_18_tomorrow = date(2008, 11, 7)
    assert calculate_age_years(born_18_tomorrow, reference_date=ref) == 17
    assert is_adult(born_18_tomorrow, reference_date=ref) is False

    # Maior de 18 anos
    assert is_adult(date(1995, 5, 20), reference_date=ref) is True


def test_fastapi_routes_when_fastapi_available() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from app.main import create_app

    app = create_app()
    with TestClient(app) as client:
        res_health = client.get("/healthz")
        assert res_health.status_code == 200
        assert res_health.json() == {"status": "ok"}
        assert "X-Request-Id" in res_health.headers

        res_ready = client.get("/readyz")
        assert res_ready.status_code == 200
        assert res_ready.json()["status"] == "ready"
