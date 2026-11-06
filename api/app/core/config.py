"""Configurações centralizadas da API carregadas via variáveis de ambiente."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BASE_DIR.parent


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_list(name: str, default: list[str]) -> list[str]:
    val = os.getenv(name)
    if not val:
        return list(default)
    return [item.strip() for item in val.split(",") if item.strip()]


@dataclass(frozen=True)
class Settings:
    """Configurações tipadas do serviço da API (doc 01)."""

    app_name: str = field(default_factory=lambda: os.getenv("APP_NAME", "Domain Detection API"))
    app_version: str = field(default_factory=lambda: os.getenv("APP_VERSION", "0.1.0"))
    environment: str = field(default_factory=lambda: os.getenv("ENVIRONMENT", "development"))
    debug: bool = field(default_factory=lambda: _env_bool("DEBUG", False))
    api_v1_prefix: str = field(default_factory=lambda: os.getenv("API_V1_PREFIX", "/v1"))

    # CORS
    cors_origins: list[str] = field(
        default_factory=lambda: _env_list(
            "CORS_ORIGINS", ["http://localhost:3000", "http://127.0.0.1:3000"]
        )
    )

    # Supabase & PostgreSQL
    supabase_url: str = field(
        default_factory=lambda: os.getenv("SUPABASE_URL", "http://localhost:54321")
    )
    supabase_anon_key: str = field(
        default_factory=lambda: os.getenv("SUPABASE_ANON_KEY", "dev-anon-key")
    )
    supabase_service_role_key: str = field(
        default_factory=lambda: os.getenv("SUPABASE_SERVICE_ROLE_KEY", "dev-service-role-key")
    )
    supabase_jwt_secret: str = field(
        default_factory=lambda: os.getenv(
            "SUPABASE_JWT_SECRET", "dev-super-secret-jwt-token-with-at-least-32-characters"
        )
    )
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "postgresql://postgres:postgres@localhost:54322/postgres"
        )
    )
    redis_url: str = field(
        default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0")
    )

    # Segurança de Chaves de API (HMAC-SHA256 + Pepper)
    api_key_pepper: str = field(
        default_factory=lambda: os.getenv(
            "API_KEY_PEPPER", "dev-secret-pepper-change-in-production-32b"
        )
    )
    api_key_prefix_tag: str = field(default_factory=lambda: os.getenv("API_KEY_PREFIX_TAG", "dmd"))
    min_adult_age_years: int = field(
        default_factory=lambda: int(os.getenv("MIN_ADULT_AGE_YEARS", "18"))
    )

    # Rate Limiting & Quotas (doc 08)
    rate_limit_per_minute: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    )
    rate_limit_per_day: int = field(
        default_factory=lambda: int(os.getenv("RATE_LIMIT_PER_DAY", "5000"))
    )
    ip_rate_limit_per_minute: int = field(
        default_factory=lambda: int(os.getenv("IP_RATE_LIMIT_PER_MINUTE", "20"))
    )

    # Coleta de Rede (DNS / WHOIS)
    dns_timeout_s: float = field(default_factory=lambda: float(os.getenv("DNS_TIMEOUT_S", "1.0")))
    whois_timeout_s: float = field(
        default_factory=lambda: float(os.getenv("WHOIS_TIMEOUT_S", "3.0"))
    )

    # Caminho dos Modelos e Artefatos do Ensemble (esemble/)
    models_root: Path = field(
        default_factory=lambda: Path(os.getenv("MODELS_ROOT", str(REPO_ROOT / "models")))
    )
    cnn_checkpoint: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "CNN_CHECKPOINT",
                str(REPO_ROOT / "models" / "cnn" / "CustomCNN0-GADF-42_30.pth"),
            )
        )
    )
    mlp_checkpoint: Path = field(
        default_factory=lambda: Path(
            os.getenv("MLP_CHECKPOINT", str(REPO_ROOT / "models" / "mlp" / "best_mlp.pt"))
        )
    )
    llm_checkpoint: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "LLM_CHECKPOINT",
                str(REPO_ROOT / "models" / "Llama" / "domain-enriched-tokenized-1B" / "best"),
            )
        )
    )
    llm_base_model: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
    )
    tabular_scaler_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "TABULAR_SCALER_PATH",
                str(REPO_ROOT / "models" / "ensemble" / "tabular_scaler.joblib"),
            )
        )
    )
    meta_model_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "META_MODEL_PATH",
                str(REPO_ROOT / "models" / "ensemble" / "meta_logreg.joblib"),
            )
        )
    )
    ccr_cca_lookup_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "CCR_CCA_LOOKUP_PATH",
                str(REPO_ROOT / "models" / "ensemble" / "ccr_cca_lookup.json"),
            )
        )
    )
    inference_device: str = field(default_factory=lambda: os.getenv("INFERENCE_DEVICE", "cuda"))
    decision_threshold: float = field(
        default_factory=lambda: float(os.getenv("DECISION_THRESHOLD", "0.5"))
    )
    model_version: str = field(
        default_factory=lambda: os.getenv(
            "MODEL_VERSION", "ensemble-v1-cnn42-mlp42-llama1b"
        )
    )
    eager_load_models: bool = field(
        default_factory=lambda: _env_bool("EAGER_LOAD_MODELS", False)
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retorna instância singleton das configurações."""
    return Settings()
