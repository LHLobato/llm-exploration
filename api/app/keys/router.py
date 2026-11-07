"""Rotas HTTP `/v1/me`, `/v1/keys/*` e `/v1/usage` (doc 04 e doc 05)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, Response, status

from app.db.models import UserProfile
from app.deps.auth import get_api_key_service, get_current_user
from app.keys.service import ApiKeyService

router = APIRouter(tags=["Keys & Profile"])


@router.get("/me", status_code=status.HTTP_200_OK)
async def get_me(
    current_user: UserProfile = Depends(get_current_user),
    key_service: ApiKeyService = Depends(get_api_key_service),
) -> dict[str, Any]:
    """Retorna o perfil do usuário autenticado e o estado atual de sua chave de API."""
    return await key_service.get_me(current_user)


@router.post("/keys", status_code=status.HTTP_201_CREATED)
async def create_key(
    response: Response,
    payload: dict[str, Any] | None = None,
    current_user: UserProfile = Depends(get_current_user),
    key_service: ApiKeyService = Depends(get_api_key_service),
) -> dict[str, Any]:
    """
    Gera uma nova chave de API (exclusivo para maiores de 18 anos) e retorna o segredo
    completo `dmd_...` **uma única vez** (`Cache-Control: no-store`).
    """
    response.headers["Cache-Control"] = "no-store"
    name = str((payload or {}).get("name", "default"))
    return await key_service.create_key(current_user, name=name)


@router.get("/keys", status_code=status.HTTP_200_OK)
async def list_keys(
    current_user: UserProfile = Depends(get_current_user),
    key_service: ApiKeyService = Depends(get_api_key_service),
) -> dict[str, Any]:
    """Lista os metadados das chaves de API do usuário (sem expor `key_hash` nem segredo)."""
    items = await key_service.list_keys(current_user)
    return {"items": items}


@router.post("/keys/{key_id}/rotate", status_code=status.HTTP_201_CREATED)
async def rotate_key(
    key_id: str,
    response: Response,
    payload: dict[str, Any] | None = None,
    current_user: UserProfile = Depends(get_current_user),
    key_service: ApiKeyService = Depends(get_api_key_service),
) -> dict[str, Any]:
    """Revoga a chave atual e emite uma nova chave exibida uma única vez."""
    response.headers["Cache-Control"] = "no-store"
    name = str((payload or {}).get("name", "default"))
    return await key_service.rotate_key(current_user, key_id=key_id, name=name)


@router.delete("/keys/{key_id}", status_code=status.HTTP_200_OK)
async def revoke_key(
    key_id: str,
    current_user: UserProfile = Depends(get_current_user),
    key_service: ApiKeyService = Depends(get_api_key_service),
) -> dict[str, Any]:
    """Revoga imediatamente a chave de API especificada."""
    return await key_service.revoke_key(current_user, key_id=key_id)


@router.get("/usage", status_code=status.HTTP_200_OK)
async def get_usage(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    current_user: UserProfile = Depends(get_current_user),
    key_service: ApiKeyService = Depends(get_api_key_service),
) -> dict[str, Any]:
    """Retorna consumo de cota e histórico paginado de análises do usuário."""
    return await key_service.get_usage_summary(current_user, limit=limit, offset=offset)
