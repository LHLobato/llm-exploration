"""Logging estruturado e rastreamento de request_id via ContextVar (doc 08)."""

from __future__ import annotations

import json
import logging
import sys
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")


def get_request_id() -> str:
    """Retorna o request_id do contexto atual ou gera um novo UUID v4."""
    rid = request_id_ctx.get()
    if not rid:
        rid = str(uuid.uuid4())
        request_id_ctx.set(rid)
    return rid


class StructuredJsonFormatter(logging.Formatter):
    """Formata registros de log em JSON de linha única com request_id."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "request_id": request_id_ctx.get() or None,
        }
        for extra_field in ("user_id", "api_key_id", "channel", "latency_ms", "status_code", "path"):
            val = getattr(record, extra_field, None)
            if val is not None:
                payload[extra_field] = val
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str = "INFO") -> None:
    """Configura o logger raiz para saída estruturada em JSON."""
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredJsonFormatter())

    root_logger.handlers = [handler]
