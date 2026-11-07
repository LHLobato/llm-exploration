"""Rotas HTTP `/v1/domains/*` protegidas por `X-API-Key` ou sessão Web com chave ativa (doc 05)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Request, status

from app.core.exceptions import InvalidDomainError
from app.deps.auth import require_api_key_or_web_jwt
from app.domains.service import DomainAnalysisService
from app.keys.service import ValidatedKeyContext

router = APIRouter(prefix="/domains", tags=["Domains"])


def get_domain_analysis_service(request: Request) -> DomainAnalysisService:
    return request.app.state.domain_analysis_service


@router.post("/analyze", status_code=status.HTTP_200_OK)
async def analyze_domain(
    payload: dict[str, Any],
    auth_ctx: ValidatedKeyContext = Depends(require_api_key_or_web_jwt),
    service: DomainAnalysisService = Depends(get_domain_analysis_service),
) -> dict[str, Any]:
    """
    Analisa um nome de domínio através do `Domain_Ensemble` multimodal (`esemble/`):
    - Recebe exclusivamente `{"domain": "exemplo.com.br"}`.
    - Exige `X-API-Key: dmd_...` (`curl`/clientes externos) ou `Authorization: Bearer <jwt>` (interface web).
    """
    if not isinstance(payload, dict) or "domain" not in payload:
        raise InvalidDomainError("O corpo da requisição deve conter apenas o campo 'domain'.")
    extra_keys = set(payload.keys()) - {"domain"}
    if extra_keys:
        raise InvalidDomainError(
            "A requisição aceita exclusivamente o nome do domínio no campo 'domain'."
        )
    return await service.analyze_domain(
        raw_domain=str(payload.get("domain", "")),
        auth_ctx=auth_ctx,
    )


@router.get("/analyses/{analysis_id}", status_code=status.HTTP_200_OK)
async def get_analysis_by_id(
    analysis_id: str,
    auth_ctx: ValidatedKeyContext = Depends(require_api_key_or_web_jwt),
    service: DomainAnalysisService = Depends(get_domain_analysis_service),
) -> dict[str, Any]:
    """Recupera os detalhes completos de uma análise de domínio pelo seu UUID."""
    return await service.get_analysis_by_id(analysis_id=analysis_id, auth_ctx=auth_ctx)
