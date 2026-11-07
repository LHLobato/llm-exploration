"""Testes do Incremento 3: Validação de Domínio, 42 Atributos Canônicos, GADF 42×42 e Prompt de 7 Seções."""

from __future__ import annotations

import base64
import struct

import pytest

from app.core.exceptions import InvalidDomainError
from app.domains.validator import normalize_and_validate_domain
from app.pipeline.collectors.dns_collector import default_empty_dns_features
from app.pipeline.constants import CANONICAL_42_FEATURES, DNS_FEATURE_ORDER, LEXICAL_FEATURE_ORDER
from app.pipeline.features.builder import FeatureVectorBuilder
from app.pipeline.features.lexical import extract_lexical_features
from app.pipeline.renderers.gadf import render_gadf_42x42
from app.pipeline.renderers.prompt import build_structured_prompt, format_llm_chat_input


def test_domain_validator_normalizes_and_strips_www() -> None:
    res = normalize_and_validate_domain("  WWW.MoneysavingCentral.co.uk. ")
    assert res.domain == "moneysavingcentral.co.uk"
    assert res.registered_domain == "moneysavingcentral.co.uk"

    sub = normalize_and_validate_domain("fr7ehgd.duckdns.org")
    assert sub.domain == "fr7ehgd.duckdns.org"
    assert sub.registered_domain == "duckdns.org"


@pytest.mark.parametrize(
    "invalid_input",
    [
        "",
        "   ",
        "https://exemplo.com.br",
        "http://exemplo.com.br/login",
        "exemplo.com.br/caminho",
        "exemplo.com.br:8080",
        "user@exemplo.com.br",
        "192.168.1.1",
        "2001:db8::1",
        "*.exemplo.com",
        "localhost",
        "servidor.local",
        "rede.internal",
        "apenasumlabel",
        "-invalido.com",
    ],
)
def test_domain_validator_rejects_urls_ips_ports_and_reserved_names(invalid_input: str) -> None:
    with pytest.raises(InvalidDomainError):
        normalize_and_validate_domain(invalid_input)


def test_lexical_features_and_42_canonical_vector_assembly() -> None:
    assert len(CANONICAL_42_FEATURES) == 42
    assert len(LEXICAL_FEATURE_ORDER) == 13
    assert len(DNS_FEATURE_ORDER) == 23

    lex = extract_lexical_features("fr7ehgd.duckdns.org")
    assert lex["lex_length"] == 19.0
    assert lex["lex_digit_count"] == 1.0
    assert lex["lex_starts_with_digit"] == 0.0
    assert lex["lex_hyphen_count"] == 0.0
    assert lex["lex_max_consonant_seq"] == 4.0
    assert lex["lex_max_digit_seq"] == 1.0
    # Soma de vowel_ratio e consonant_ratio sobre letras deve ser 1.0 (advanced_analysis.py linha 26)
    assert round(lex["lex_vowel_ratio"] + lex["lex_consonant_ratio"], 5) == 1.0

    dns_feats = default_empty_dns_features()
    dns_feats.update(
        {
            "A Count": 1.0,
            "IP Count": 1.0,
            "NS Count": 2.0,
            "A Missing": 0.0,
            "ASN Count": 1.0,
            "MX Missing": 1.0,
            "A TTL Count": 1.0,
            "A Medium TTL": 60.0,
            "SOA Missing": 0.0,
            "SOA Refresh": 43200.0,
            "SOA Retry": 3600.0,
            "SOA Minimum": 3601.0,
            "SOA Expire Length": 6.0,
            "SOA Serial Length": 10.0,
            "SOA TTL Count": 1.0,
            "SOA Medium TTL": 3601.0,
            "SOA TTL Standard Deviation": 3601.0,
        }
    )
    whois_risk = {
        "has_whois": 0.0,
        "lifetime": 0.0,
        "active_time": 0.0,
        "has_country": 1.0,
        "CCR": 0.863,
        "CCA": 0.863,
    }

    builder = FeatureVectorBuilder()
    assembled = builder.assemble(lex, dns_feats, whois_risk)

    assert len(assembled.raw_vector_42) == 42
    assert len(assembled.scaled_vector_42) == 42
    assert all(0.0 <= v <= 1.0 for v in assembled.scaled_vector_42)

    # Verifica grupos na resposta JSON
    assert set(assembled.grouped_features.keys()) == {
        "lexical",
        "dns_counts_and_flags",
        "ttl_stats",
        "soa_params",
        "whois_and_risk",
    }
    total_grouped_keys = sum(len(group) for group in assembled.grouped_features.values())
    assert total_grouped_keys == 42


def test_structured_prompt_matches_7_sections_from_test_prompt_py() -> None:
    lex = extract_lexical_features("fr7ehgd.duckdns.org")
    dns_feats = default_empty_dns_features()
    dns_feats.update(
        {
            "A Count": 1.0,
            "IP Count": 1.0,
            "NS Count": 2.0,
            "A Missing": 0.0,
            "ASN Count": 1.0,
            "MX Missing": 1.0,
            "A TTL Count": 1.0,
            "A Medium TTL": 60.0,
            "SOA Missing": 0.0,
            "SOA Refresh": 43200.0,
            "SOA Retry": 3600.0,
            "SOA Minimum": 3601.0,
            "SOA Expire Length": 6.0,
            "SOA Serial Length": 10.0,
            "SOA TTL Count": 1.0,
            "SOA Medium TTL": 3601.0,
            "SOA TTL Standard Deviation": 0.0,
        }
    )
    whois_risk = {
        "has_whois": 0.0,
        "lifetime": 0.0,
        "active_time": 0.0,
        "has_country": 1.0,
        "CCR": 0.863,
        "CCA": 0.863,
    }
    builder = FeatureVectorBuilder()
    assembled = builder.assemble(lex, dns_feats, whois_risk)

    prompt = build_structured_prompt("fr7ehgd.duckdns.org", assembled.flat_features)
    lines = prompt.splitlines()
    assert len(lines) == 7
    assert lines[0] == "[name]: fr7ehgd.duckdns.org"
    assert lines[1].startswith("[entropy]: 3.6819")
    assert lines[2] == "[whois]: present=no"
    assert lines[3] == "[risk]: ccr=0.863 cca=0.863 country=yes"
    assert (
        lines[4]
        == "[dns]: A=1 AAAA=0 MX=0 NS=2 CNAME=0 IPs=1 ASNs=1 a_missing=no mx_missing=yes"
    )
    assert lines[5] == "[ttl]: a_med=60.0 a_std=0.0 a_distinct=1 mx_med=0.0 aaaa_med=0.0"
    assert (
        lines[6]
        == "[soa]: missing=no refresh=43200 retry=3600 min=3601 expire_len=6 serial_len=10 med=3601.0 std=0.0 distinct=1"
    )

    chat_input = format_llm_chat_input(prompt)
    assert "You are a cybersecurity classifier." in chat_input
    assert "[name]: fr7ehgd.duckdns.org" in chat_input


def test_gadf_42x42_renderer_produces_valid_3x42x42_tensor_and_png() -> None:
    values_42 = [float(i) / 41.0 for i in range(42)]
    artifact = render_gadf_42x42(values_42)

    assert len(artifact.matrix_42x42) == 42
    assert all(len(row) == 42 for row in artifact.matrix_42x42)
    # Na matriz GADF (sin(phi_i - phi_j)), a diagonal principal deve ser zero
    for idx in range(42):
        assert abs(artifact.matrix_42x42[idx][idx]) < 1e-6

    # Tensor RGB (3, 42, 42) normalizado em [0.0, 1.0]
    assert len(artifact.rgb_tensor_chw) == 3
    assert len(artifact.rgb_tensor_chw[0]) == 42
    assert len(artifact.rgb_tensor_chw[0][0]) == 42

    # Verifica PNG Base64 (cabeçalho PNG + dimensões 42x42 no chunk IHDR)
    assert artifact.png_base64_data_uri.startswith("data:image/png;base64,")
    raw_png = base64.b64decode(artifact.png_base64_data_uri.split(",", 1)[1])
    assert raw_png[:8] == b"\x89PNG\r\n\x1a\n"
    width, height = struct.unpack("!II", raw_png[16:24])
    assert (width, height) == (42, 42)
