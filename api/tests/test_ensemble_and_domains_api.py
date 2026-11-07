"""Testes do Incremento 4: Motor `Domain_Ensemble` (`esemble/`) e Serviço `/v1/domains/analyze`."""

from __future__ import annotations

import asyncio

import pytest

from app.auth.service import AuthService
from app.core.exceptions import InvalidDomainError, NotFoundError
from app.db.repository import InMemoryRepository
from app.domains.service import DomainAnalysisService
from app.keys.service import ApiKeyService
from app.pipeline.collectors.dns_collector import DnsCollectionResult, default_empty_dns_features
from app.pipeline.collectors.whois_collector import WhoisAndRiskResult
from app.pipeline.features.builder import FeatureVectorBuilder
from app.pipeline.features.lexical import extract_lexical_features
from app.pipeline.inference.ensemble import Domain_Ensemble
from app.pipeline.renderers.gadf import render_gadf_42x42
from app.pipeline.renderers.prompt import build_structured_prompt


class _MockDnsCollector:
    def __init__(self, features: dict[str, float], ips: list[str]) -> None:
        self._features = features
        self._ips = ips

    async def collect(self, domain: str) -> DnsCollectionResult:
        return DnsCollectionResult(
            features=dict(self._features),
            evidence={
                "a_records": list(self._ips),
                "aaaa_records": [],
                "mx_records": [],
                "ns_records": ["ns1.duckdns.org", "ns2.duckdns.org"],
                "cname_records": [],
                "soa": {
                    "mname": "ns1.duckdns.org",
                    "serial": 2026110601,
                    "refresh": 43200,
                    "retry": 3600,
                    "expire": 604800,
                    "minimum": 3601,
                    "ttl": 3601,
                },
            },
            resolved_ips=list(self._ips),
            partial=False,
            notes=[],
        )


class _MockWhoisCollector:
    def __init__(self, features: dict[str, float]) -> None:
        self._features = features

    async def collect(
        self, registered_domain: str, resolved_ips: list[str] | None = None
    ) -> WhoisAndRiskResult:
        return WhoisAndRiskResult(
            features=dict(self._features),
            whois_evidence={
                "present": bool(self._features.get("has_whois", 0.0)),
                "creation_date": None,
                "updated_date": None,
                "expiration_date": None,
                "registrar": None,
            },
            geo_asn_evidence={
                "countries": ["US"] if self._features.get("has_country") else [],
                "asns": ["AS16509"],
                "ccr": self._features.get("CCR", 0.0),
                "cca": self._features.get("CCA", 0.0),
            },
            resolved_asn_count=1,
            partial=False,
            notes=[],
        )


def test_domain_ensemble_combines_three_modalities_and_exposes_meta_model() -> None:
    ensemble = Domain_Ensemble(eager_load=False)
    # Verifica compatibilidade com esemble/test.py (linha 73: ensemble.meta_model)
    assert ensemble.meta_model is ensemble.clf

    weights = ensemble.get_meta_weights()
    assert set(weights.keys()) == {"vision_cnn", "tabular_mlp", "text_llm", "intercept"}

    # Verifica monotonicidade do meta-classificador LogisticRegression
    low_prob = ensemble.combine_probabilities(0.05, 0.05, 0.05)
    high_prob = ensemble.combine_probabilities(0.95, 0.95, 0.95)
    assert low_prob < 0.25
    assert high_prob > 0.85

    # Testa predict_single completo em domínio suspeito/malicioso de alta entropia sem WHOIS
    domain = "fr7ehgd99x8z7q1.duckdns.org"
    lex = extract_lexical_features(domain)
    dns_feats = default_empty_dns_features()
    dns_feats.update({"A Missing": 1.0, "MX Missing": 1.0, "SOA Missing": 1.0})
    whois_risk = {
        "has_whois": 0.0,
        "lifetime": 0.0,
        "active_time": 0.0,
        "has_country": 1.0,
        "CCR": 0.92,
        "CCA": 0.92,
    }
    builder = FeatureVectorBuilder()
    assembled = builder.assemble(lex, dns_feats, whois_risk)
    prompt = build_structured_prompt(domain, assembled.flat_features)
    gadf = render_gadf_42x42(assembled.scaled_vector_42)

    pred = ensemble.predict_single(
        rgb_tensor_chw=gadf.rgb_tensor_chw,
        scaled_vector_42=assembled.scaled_vector_42,
        structured_prompt=prompt,
        flat_features=assembled.flat_features,
    )
    assert pred.verdict == "malicious"
    assert pred.label_id == 1
    assert pred.score >= 0.5
    assert set(pred.model_scores.keys()) == {"vision_cnn", "tabular_mlp", "text_llm"}


def test_end_to_end_domain_analysis_service_caching_and_audit() -> None:
    async def scenario() -> None:
        repo = InMemoryRepository()
        auth = AuthService(repo=repo)
        keys = ApiKeyService(repo=repo)

        # Cria usuário maior de 18 anos e chave de API
        await auth.signup(
            {
                "username": "analyst_01",
                "first_name": "Marina",
                "last_name": "Rocha",
                "birth_date": "1992-07-14",
                "email": "marina@exemplo.com",
                "password": "StrongPassword!123",
            }
        )
        otp = auth.get_last_otp_for_testing("marina@exemplo.com")
        assert otp is not None
        session = await auth.verify_signup_otp("marina@exemplo.com", otp)
        profile = await auth.authenticate_jwt(session["access_token"])
        created_key = await keys.create_key(profile, name="prod-key")
        auth_ctx = await keys.validate_api_key_secret(created_key["key"])

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
        whois_feats = {
            "has_whois": 0.0,
            "lifetime": 0.0,
            "active_time": 0.0,
            "has_country": 1.0,
            "CCR": 0.863,
            "CCA": 0.863,
        }

        service = DomainAnalysisService(
            repo=repo,
            ensemble=Domain_Ensemble(eager_load=False),
            dns_collector=_MockDnsCollector(dns_feats, ["203.0.113.45"]),  # type: ignore[arg-type]
            whois_collector=_MockWhoisCollector(whois_feats),  # type: ignore[arg-type]
        )

        # 1ª chamada -> cache miss, executa pipeline completo
        res1 = await service.analyze_domain("WWW.fr7ehgd.duckdns.org", auth_ctx=auth_ctx)
        assert res1["domain"] == "fr7ehgd.duckdns.org"
        assert res1["registered_domain"] == "duckdns.org"
        assert res1["cached"] is False
        assert res1["verdict"] in {"malicious", "benign"}
        assert "vision_cnn" in res1["model_scores"]
        assert "tabular_mlp" in res1["model_scores"]
        assert "text_llm" in res1["model_scores"]
        assert res1["artifacts"]["gadf_image_base64"].startswith("data:image/png;base64,")
        assert len(res1["artifacts"]["structured_prompt"].splitlines()) == 7
        assert sum(len(g) for g in res1["features"].values()) == 42

        # 2ª chamada para o mesmo domínio -> cache hit
        res2 = await service.analyze_domain("fr7ehgd.duckdns.org", auth_ctx=auth_ctx)
        assert res2["id"] == res1["id"]
        assert res2["cached"] is True

        # Consulta por ID
        fetched = await service.get_analysis_by_id(res1["id"], auth_ctx=auth_ctx)
        assert fetched["domain"] == "fr7ehgd.duckdns.org"

        # Consulta de ID inexistente -> 404 NotFoundError
        with pytest.raises(NotFoundError):
            await service.get_analysis_by_id("00000000-0000-0000-0000-000000000000", auth_ctx=auth_ctx)

        # Domínio inválido (URL com protocolo) -> 400 InvalidDomainError
        with pytest.raises(InvalidDomainError):
            await service.analyze_domain("https://fr7ehgd.duckdns.org/login", auth_ctx=auth_ctx)

        # Verifica auditoria em /v1/usage
        usage = await keys.get_usage_summary(profile)
        assert usage["quota"]["used_today"] == 2
        assert len(usage["history"]) == 2

    asyncio.run(scenario())
