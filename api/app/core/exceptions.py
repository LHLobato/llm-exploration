"""Hierarquia de exceções e formato padronizado de erro da API (doc 05)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from app.core.logging import get_request_id

if TYPE_CHECKING:
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse


class AppError(Exception):
    """Exceção base da aplicação que gera o envelope de erro padronizado do doc 05."""

    def __init__(
        self,
        code: str,
        message: str,
        status_code: int = 400,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.headers = headers or {}


class InvalidDomainError(AppError):
    """Erro 400 quando o domínio informado é inválido ou contém URL/IP/caminho."""

    def __init__(
        self,
        message: str = (
            "Informe apenas o nome do domínio (sem protocolo http/https, caminho ou porta)."
        ),
    ) -> None:
        super().__init__(
            code="invalid_domain",
            message=message,
            status_code=400,
        )


class UnauthorizedError(AppError):
    """Erro 401 para ausência ou invalidade de credenciais (JWT ou X-API-Key)."""

    def __init__(self, message: str = "Credenciais de autenticação ausentes ou inválidas.") -> None:
        super().__init__(
            code="unauthorized",
            message=message,
            status_code=401,
        )


class ForbiddenError(AppError):
    """Erro 403 para acesso negado (chave revogada, expirada ou sem permissão)."""

    def __init__(
        self,
        code: str = "forbidden",
        message: str = "Acesso negado para este recurso.",
    ) -> None:
        super().__init__(
            code=code,
            message=message,
            status_code=403,
        )


class AgeRestrictedError(ForbiddenError):
    """Erro 403 específico para usuários menores de 18 anos (doc 03 e doc 04)."""

    def __init__(
        self,
        message: str = (
            "A disponibilização e o uso de chaves de API são exclusivos para maiores de 18 anos."
        ),
    ) -> None:
        super().__init__(code="age_restricted", message=message)


class NotFoundError(AppError):
    """Erro 404 para recursos não encontrados."""

    def __init__(self, message: str = "Recurso não encontrado.") -> None:
        super().__init__(
            code="not_found",
            message=message,
            status_code=404,
        )


class ConflictError(AppError):
    """Erro 409 para conflitos de estado (ex.: chave ativa já existente ou username em uso)."""

    def __init__(
        self, code: str = "conflict", message: str = "Conflito no estado do recurso."
    ) -> None:
        super().__init__(
            code=code,
            message=message,
            status_code=409,
        )


class RateLimitExceededError(AppError):
    """Erro 429 quando o limite de requisições por chave/IP/OTP é excedido."""

    def __init__(
        self,
        message: str = "Limite de requisições excedido. Tente novamente em instantes.",
        retry_after: int = 60,
    ) -> None:
        super().__init__(
            code="rate_limit_exceeded",
            message=message,
            status_code=429,
            headers={"Retry-After": str(retry_after)},
        )


def build_error_payload(code: str, message: str, request_id: str | None = None) -> dict[str, Any]:
    """Constrói o JSON de erro no formato canônico do doc 05."""
    return {
        "error": {
            "code": code,
            "message": message,
            "request_id": request_id or get_request_id(),
        }
    }


def register_exception_handlers(app: "FastAPI") -> None:
    """Registra tratadores globais de exceção na instância FastAPI."""
    from fastapi import Request, status
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse

    @app.exception_handler(AppError)
    async def handle_app_error(_request: Request, exc: AppError) -> JSONResponse:
        rid = get_request_id()
        headers = {"X-Request-Id": rid, **exc.headers}
        return JSONResponse(
            status_code=exc.status_code,
            content=build_error_payload(exc.code, exc.message, rid),
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        rid = get_request_id()
        errors = exc.errors()
        first_msg = "Dados de entrada inválidos."
        if errors:
            loc = ".".join(str(part) for part in errors[0].get("loc", []) if part != "body")
            msg = errors[0].get("msg", first_msg)
            first_msg = f"Campo '{loc}': {msg}" if loc else msg
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=build_error_payload("validation_error", first_msg, rid),
            headers={"X-Request-Id": rid},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_request: Request, _exc: Exception) -> JSONResponse:
        rid = get_request_id()
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=build_error_payload(
                "internal_error",
                "Erro interno inesperado ao processar a requisição.",
                rid,
            ),
            headers={"X-Request-Id": rid},
        )
