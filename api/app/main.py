"""Entrypoint da aplicação FastAPI, middleware de Request-ID e rotas de healthcheck."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from app.core.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, request_id_ctx

logger = logging.getLogger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Injeta `X-Request-Id` em cada requisição/resposta e registra log estruturado."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        incoming_rid = request.headers.get("X-Request-Id", "").strip()
        try:
            rid = str(uuid.UUID(incoming_rid)) if incoming_rid else str(uuid.uuid4())
        except ValueError:
            rid = str(uuid.uuid4())

        token = request_id_ctx.set(rid)
        start_perf = time.perf_counter()
        try:
            response = await call_next(request)
            latency_ms = round((time.perf_counter() - start_perf) * 1000, 2)
            response.headers["X-Request-Id"] = rid
            logger.info(
                "http_request_completed",
                extra={
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "latency_ms": latency_ms,
                },
            )
            return response
        finally:
            request_id_ctx.reset(token)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Gerencia ciclo de vida da API (conexões de banco e carregamento do Domain_Ensemble)."""
    settings = get_settings()
    configure_logging("DEBUG" if settings.debug else "INFO")
    app.state.ready = True
    app.state.model_version = settings.model_version
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    yield
    app.state.ready = False
    logger.info("Shutting down %s", settings.app_name)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Cria e configura a instância FastAPI."""
    cfg = settings or get_settings()
    app = FastAPI(
        title=cfg.app_name,
        version=cfg.app_version,
        docs_url="/docs",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    app.state.ready = True
    app.state.model_version = cfg.model_version

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=[
            "X-Request-Id",
            "X-RateLimit-Limit",
            "X-RateLimit-Remaining",
            "X-RateLimit-Reset",
            "Retry-After",
        ],
    )
    app.add_middleware(RequestContextMiddleware)

    register_exception_handlers(app)

    @app.get("/healthz", tags=["Health"])
    async def healthz() -> dict[str, str]:
        """Liveness probe para Docker/Kubernetes (`start.sh`)."""
        return {"status": "ok"}

    @app.get("/readyz", tags=["Health"])
    async def readyz(request: Request) -> dict[str, object]:
        """Readiness probe reportando prontidão do serviço e versão do ensemble."""
        return {
            "status": "ready" if getattr(request.app.state, "ready", False) else "starting",
            "model_version": getattr(request.app.state, "model_version", cfg.model_version),
        }

    return app


app = create_app()
