"""Rotas HTTP `/v1/auth/*` para cadastro, login em 2 etapas, OTP e reset de senha (doc 05)."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Header, Request, Response, status

from app.auth.service import AuthService

router = APIRouter(prefix="/auth", tags=["Auth"])


def get_auth_service(request: Request) -> AuthService:
    return request.app.state.auth_service


def _client_ip(request: Request) -> str | None:
    return request.client.host if request.client else None


@router.post("/signup", status_code=status.HTTP_202_ACCEPTED)
async def signup(
    payload: dict[str, Any],
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
) -> dict[str, Any]:
    """Inicia o cadastro e envia o código OTP de 6 dígitos por e-mail."""
    return await auth_service.signup(payload, ip_address=_client_ip(request))


@router.post("/signup/verify", status_code=status.HTTP_200_OK)
async def verify_signup(
    payload: dict[str, Any],
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
) -> dict[str, Any]:
    """Confirma o OTP de cadastro, cria o perfil e retorna a sessão autenticada."""
    return await auth_service.verify_signup_otp(
        email=str(payload.get("email", "")),
        token=str(payload.get("token", "")),
        ip_address=_client_ip(request),
    )


@router.post("/login", status_code=status.HTTP_200_OK)
async def login(
    payload: dict[str, Any],
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
) -> dict[str, Any]:
    """Etapa 1 do login: valida email e senha sem reter sessão e envia OTP por e-mail."""
    return await auth_service.login_step1(
        email=str(payload.get("email", "")),
        password=str(payload.get("password", "")),
        ip_address=_client_ip(request),
    )


@router.post("/login/verify", status_code=status.HTTP_200_OK)
async def verify_login(
    payload: dict[str, Any],
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
) -> dict[str, Any]:
    """Etapa 2 do login: valida o OTP de 6 dígitos e retorna `access_token` e `refresh_token`."""
    return await auth_service.verify_login_otp(
        email=str(payload.get("email", "")),
        token=str(payload.get("token", "")),
        ip_address=_client_ip(request),
    )


@router.post("/otp/resend", status_code=status.HTTP_200_OK)
async def resend_otp(
    payload: dict[str, Any],
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
) -> dict[str, Any]:
    """Reenvia o código OTP respeitando o intervalo mínimo de 60 segundos."""
    return await auth_service.resend_otp(
        email=str(payload.get("email", "")),
        flow=str(payload.get("flow", "login")),
        ip_address=_client_ip(request),
    )


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    authorization: str | None = Header(default=None),
    auth_service: AuthService = Depends(get_auth_service),
) -> Response:
    """Invalida a sessão atual do usuário."""
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
        auth_service.logout(token)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post("/password/reset", status_code=status.HTTP_200_OK)
async def request_password_reset(
    payload: dict[str, Any],
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
) -> dict[str, Any]:
    """Dispara código OTP para redefinição de senha."""
    return await auth_service.request_password_reset(
        email=str(payload.get("email", "")),
        ip_address=_client_ip(request),
    )


@router.post("/password/reset/verify", status_code=status.HTTP_200_OK)
async def verify_password_reset(
    payload: dict[str, Any],
    request: Request,
    auth_service: AuthService = Depends(get_auth_service),
) -> dict[str, Any]:
    """Valida o OTP de recuperação e redefine a senha do usuário."""
    return await auth_service.verify_password_reset(
        email=str(payload.get("email", "")),
        token=str(payload.get("token", "")),
        new_password=str(payload.get("new_password", "")),
        ip_address=_client_ip(request),
    )
