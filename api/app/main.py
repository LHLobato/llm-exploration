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

from app.auth.router import router as auth_router
from app.auth.service import AuthService
from app.core.config import Settings, get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, request_id_ctx
from app.db.repository import DatabaseRepository, InMemoryRepository
from app.deps.auth import SlidingWindowRateLimiter
from app.domains.router import router as domains_router
from app.domains.service import DomainAnalysisService
from app.keys.router import router as keys_router
from app.keys.service import ApiKeyService
from app.pipeline.inference.ensemble import Domain_Ensemble

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
    if settings.eager_load_models and hasattr(app.state, "ensemble"):
        app.state.ensemble.load_models()
    app.state.ready = True
    app.state.model_version = settings.model_version
    logger.info("Starting %s v%s", settings.app_name, settings.app_version)
    yield
    app.state.ready = False
    logger.info("Shutting down %s", settings.app_name)


def create_app(
    settings: Settings | None = None,
    repo: DatabaseRepository | None = None,
    ensemble: Domain_Ensemble | None = None,
    domain_analysis_service: DomainAnalysisService | None = None,
) -> FastAPI:
    """Cria e configura a instância FastAPI com serviços de Auth, Chaves, Banco e Domain_Ensemble."""
    cfg = settings or get_settings()
    db_repo = repo or InMemoryRepository()
    ensemble_engine = ensemble or Domain_Ensemble.from_settings(cfg)
    analysis_service = domain_analysis_service or DomainAnalysisService(
        repo=db_repo,
        ensemble=ensemble_engine,
        settings=cfg,
    )

    app = FastAPI(
        title=cfg.app_name,
        version=cfg.app_version,
        docs_url="/docs",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    app.state.ready = True
    app.state.model_version = cfg.model_version
    app.state.repo = db_repo
    app.state.auth_service = AuthService(repo=db_repo, settings=cfg)
    app.state.api_key_service = ApiKeyService(repo=db_repo, settings=cfg)
    app.state.rate_limiter = SlidingWindowRateLimiter()
    app.state.ensemble = ensemble_engine
    app.state.domain_analysis_service = analysis_service

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

    # Rotas /v1/auth/*, /v1/me, /v1/keys/*, /v1/usage e /v1/domains/*
    app.include_router(auth_router, prefix=cfg.api_v1_prefix)
    app.include_router(keys_router, prefix=cfg.api_v1_prefix)
    app.include_router(domains_router, prefix=cfg.api_v1_prefix)

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
