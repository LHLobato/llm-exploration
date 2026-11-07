"""Serviço de análise de domínios orquestrando coleta, 42 atributos, GADF, prompt e `Domain_Ensemble`."""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from app.core.config import Settings, get_settings
from app.core.exceptions import InvalidDomainError, NotFoundError
from app.db.models import ApiRequestRecord, DomainAnalysisRecord
from app.db.repository import DatabaseRepository
from app.domains.validator import normalize_and_validate_domain
from app.keys.service import ValidatedKeyContext
from app.pipeline.collectors.dns_collector import DnsCollectionResult, DnsCollector
from app.pipeline.collectors.whois_collector import WhoisAndRiskCollector, WhoisAndRiskResult
from app.pipeline.features.builder import FeatureVectorBuilder
from app.pipeline.features.lexical import extract_lexical_features
from app.pipeline.inference.ensemble import Domain_Ensemble
from app.pipeline.renderers.gadf import render_gadf_42x42
from app.pipeline.renderers.prompt import build_structured_prompt


class DomainAnalysisService:
    """
    Executa o pipeline completo de detecção especificado em `planejamento/06-pipeline-de-deteccao.md`:
    1. Valida e normaliza o domínio de entrada (`clean_domain` + `remove_www_prefix` + `idna`).
    2. Consulta cache em `public.domain_analyses` para `(domain, model_version)`.
    3. Em cache miss:
       - Extrai os 13 atributos lexicais;
       - Executa `DnsCollector` (23 atributos DNS/TTL/SOA) e `WhoisAndRiskCollector` (6 atributos WHOIS/CCR/CCA);
       - Monta e escalona o vetor canônico de 42 atributos (`FeatureVectorBuilder`);
       - Gera a imagem GADF 42×42 RGB em memória e o prompt estruturado de 7 seções;
       - Executa `Domain_Ensemble.predict_single` (`CustomCNN` + `MLP` + `Llama-3.2-1B` + `LogisticRegression`);
       - Persiste em `public.domain_analyses`.
    4. Registra a requisição em `public.api_requests` vinculada à chave validada (`api_key_id`, `user_id`, `channel`).
    """

    CACHE_TTL_HOURS = 12

    def __init__(
        self,
        repo: DatabaseRepository,
        ensemble: Domain_Ensemble | None = None,
        dns_collector: DnsCollector | None = None,
        whois_collector: WhoisAndRiskCollector | None = None,
        feature_builder: FeatureVectorBuilder | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._repo = repo
        self._settings = settings or get_settings()
        self._ensemble = ensemble or Domain_Ensemble.from_settings(self._settings)
        self._dns_collector = dns_collector or DnsCollector(
            timeout_s=self._settings.dns_timeout_s,
            lifetime_s=self._settings.dns_timeout_s,
        )
        self._whois_collector = whois_collector or WhoisAndRiskCollector(
            timeout_s=self._settings.whois_timeout_s,
            lookup_path=self._settings.ccr_cca_lookup_path,
        )
        self._feature_builder = feature_builder or FeatureVectorBuilder(
            scaler_path=self._settings.tabular_scaler_path
        )

    @staticmethod
    def _format_analysis_response(
        record: DomainAnalysisRecord,
        *,
        cached: bool,
        latency_ms: int,
    ) -> dict[str, Any]:
        """Formata `DomainAnalysisRecord` no contrato exato de `POST /v1/domains/analyze` (doc 05)."""
        gadf_b64 = record.evidence.get("gadf_image_base64", "")
        clean_evidence = {
            k: v for k, v in record.evidence.items() if k != "gadf_image_base64"
        }
        return {
            "id": record.id,
            "domain": record.domain,
            "registered_domain": record.registered_domain,
            "verdict": record.label,
            "label_id": 1 if record.label == "malicious" else 0,
            "score": round(float(record.score), 4),
            "threshold": round(float(record.threshold), 4),
            "model_version": record.model_version,
            "model_scores": record.model_scores,
            "meta_weights": record.meta_weights,
            "partial": record.partial,
            "cached": cached,
            "latency_ms": latency_ms,
            "analyzed_at": record.created_at.isoformat(),
            "artifacts": {
                "structured_prompt": record.structured_prompt,
                "gadf_image_base64": gadf_b64,
            },
            "features": record.features,
            "evidence": clean_evidence,
        }

    async def analyze_domain(
        self,
        raw_domain: str,
        auth_ctx: ValidatedKeyContext,
        bypass_cache: bool = False,
    ) -> dict[str, Any]:
        """Analisa um único domínio em tempo real e registra auditoria na chave do usuário."""
        t0 = time.perf_counter()

        normalized = normalize_and_validate_domain(raw_domain)
        domain = normalized.domain
        registered_domain = normalized.registered_domain

        now = datetime.now(timezone.utc)
        if not bypass_cache:
            cached_record = await self._repo.get_cached_domain_analysis(
                domain=domain,
                model_version=self._ensemble.model_version,
                now=now,
            )
            if cached_record is not None:
                latency_ms = max(1, int(round((time.perf_counter() - t0) * 1000)))
                await self._repo.log_api_request(
                    ApiRequestRecord(
                        id=str(uuid.uuid4()),
                        api_key_id=auth_ctx.api_key.id,
                        user_id=auth_ctx.user.id,
                        analysis_id=cached_record.id,
                        channel=auth_ctx.channel,
                        domain=domain,
                        verdict=cached_record.label,
                        score=cached_record.score,
                        cached=True,
                        status_code=200,
                        latency_ms=latency_ms,
                        created_at=now,
                    )
                )
                return self._format_analysis_response(
                    cached_record,
                    cached=True,
                    latency_ms=latency_ms,
                )

        # 1. Extração Lexicográfica (13 atributos)
        lexical_features = extract_lexical_features(domain)

        # 2. Coleta DNS (23 atributos) + Coleta WHOIS/Risco (6 atributos)
        dns_res: DnsCollectionResult = await self._dns_collector.collect(domain)
        whois_res: WhoisAndRiskResult = await self._whois_collector.collect(
            registered_domain=registered_domain,
            resolved_ips=dns_res.resolved_ips,
        )

        # Atualiza ASN Count caso o enriquecimento de ASN tenha identificado múltiplos ASNs
        if whois_res.resolved_asn_count > 0 and dns_res.features.get("ASN Count", 0.0) == 0.0:
            dns_res.features["ASN Count"] = float(whois_res.resolved_asn_count)

        # 3. Montagem do vetor canônico de 42 atributos + MinMaxScaler
        assembled = self._feature_builder.assemble(
            lexical=lexical_features,
            dns_features=dns_res.features,
            whois_risk=whois_res.features,
        )

        # 4. Geração dos Artefatos Multimodais (Prompt de 7 seções + Imagem GADF 42×42 RGB)
        structured_prompt = build_structured_prompt(domain, assembled.flat_features)
        gadf_artifact = render_gadf_42x42(assembled.scaled_vector_42)

        # 5. Inferência no Domain_Ensemble (esemble/: CustomCNN + MLP + Llama-3.2-1B + LogisticRegression)
        prediction = self._ensemble.predict_single(
            rgb_tensor_chw=gadf_artifact.rgb_tensor_chw,
            scaled_vector_42=assembled.scaled_vector_42,
            structured_prompt=structured_prompt,
            flat_features=assembled.flat_features,
        )

        latency_ms = max(1, int(round((time.perf_counter() - t0) * 1000)))
        notes = [*dns_res.notes, *whois_res.notes]
        partial = bool(dns_res.partial or whois_res.partial)

        evidence_payload: dict[str, Any] = {
            "dns": dns_res.evidence,
            "whois": whois_res.whois_evidence,
            "geo_asn": whois_res.geo_asn_evidence,
            "notes": notes,
            "gadf_image_base64": gadf_artifact.png_base64_data_uri,
        }

        analysis_record = DomainAnalysisRecord(
            id=str(uuid.uuid4()),
            domain=domain,
            registered_domain=registered_domain,
            label=prediction.verdict,
            score=prediction.score,
            threshold=prediction.threshold,
            model_version=prediction.model_version,
            model_scores=prediction.model_scores,
            meta_weights=prediction.meta_weights,
            features=assembled.grouped_features,
            structured_prompt=structured_prompt,
            evidence=evidence_payload,
            partial=partial,
            latency_ms=latency_ms,
            created_at=now,
            expires_at=now + timedelta(hours=self.CACHE_TTL_HOURS),
        )
        await self._repo.save_domain_analysis(analysis_record)

        await self._repo.log_api_request(
            ApiRequestRecord(
                id=str(uuid.uuid4()),
                api_key_id=auth_ctx.api_key.id,
                user_id=auth_ctx.user.id,
                analysis_id=analysis_record.id,
                channel=auth_ctx.channel,
                domain=domain,
                verdict=analysis_record.label,
                score=analysis_record.score,
                cached=False,
                status_code=200,
                latency_ms=latency_ms,
                created_at=now,
            )
        )

        return self._format_analysis_response(
            analysis_record,
            cached=False,
            latency_ms=latency_ms,
        )

    async def get_analysis_by_id(
        self,
        analysis_id: str,
        auth_ctx: ValidatedKeyContext,
    ) -> dict[str, Any]:
        """Recupera uma análise específica garantindo que o usuário tenha acesso a ela."""
        record = await self._repo.get_domain_analysis_by_id(analysis_id)
        if record is None:
            raise NotFoundError("Análise de domínio não encontrada.")

        user_requests = await self._repo.list_api_requests_for_user(auth_ctx.user.id, limit=500)
        if not any(r.analysis_id == analysis_id for r in user_requests):
            raise NotFoundError("Análise de domínio não encontrada no histórico deste usuário.")

        return self._format_analysis_response(
            record,
            cached=True,
            latency_ms=record.latency_ms,
        )
