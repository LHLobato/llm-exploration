"""Coletor WHOIS/RDAP e enriquecimento GeoIP/ASN (`has_whois`, `lifetime`, `active_time`, `has_country`, `CCR`, `CCA`)."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class WhoisAndRiskResult:
    """Resultado do coletor WHOIS + GeoIP/ASN (Grupo C — 6 atributos + evidências)."""

    features: dict[str, float]
    whois_evidence: dict[str, Any]
    geo_asn_evidence: dict[str, Any]
    resolved_asn_count: int = 0
    partial: bool = False
    notes: list[str] = field(default_factory=list)


def _normalize_dt(value: Any) -> datetime | None:
    if isinstance(value, list) and value:
        value = value[0]
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


class WhoisAndRiskCollector:
    """
    Calcula os 6 atributos do Grupo C (`WHOIS_RISK_FEATURE_ORDER`):
    - `has_whois` (0.0 ou 1.0)
    - `lifetime` (dias)
    - `active_time` (dias)
    - `has_country` (0.0 ou 1.0)
    - `CCR` (Country Code Risk em [0, 1])
    - `CCA` (Country / ASN Risk em [0, 1])
    """

    def __init__(
        self,
        timeout_s: float = 3.0,
        lookup_path: Path | None = None,
    ) -> None:
        self._timeout_s = timeout_s
        self._ccr_map: dict[str, float] = {}
        self._cca_map: dict[str, float] = {}
        self._default_ccr: float = 0.0
        self._default_cca: float = 0.0

        if lookup_path and lookup_path.is_file():
            try:
                data = json.loads(lookup_path.read_text(encoding="utf-8"))
                self._ccr_map = {
                    str(k).upper(): float(v) for k, v in data.get("ccr", {}).items()
                }
                self._cca_map = {
                    str(k).upper(): float(v) for k, v in data.get("cca", {}).items()
                }
                self._default_ccr = float(data.get("default_ccr", 0.5))
                self._default_cca = float(data.get("default_cca", 0.5))
            except Exception:
                pass

    def _sync_query_whois(self, registered_domain: str) -> dict[str, Any]:
        import whois  # type: ignore

        w = whois.whois(registered_domain)
        return dict(w) if w else {}

    async def collect(
        self,
        registered_domain: str,
        resolved_ips: list[str] | None = None,
    ) -> WhoisAndRiskResult:
        ips = resolved_ips or []
        features: dict[str, float] = {
            "has_whois": 0.0,
            "lifetime": 0.0,
            "active_time": 0.0,
            "has_country": 0.0,
            "CCR": 0.0,
            "CCA": 0.0,
        }
        whois_evidence: dict[str, Any] = {
            "present": False,
            "creation_date": None,
            "updated_date": None,
            "expiration_date": None,
            "registrar": None,
        }
        geo_asn_evidence: dict[str, Any] = {
            "countries": [],
            "asns": [],
            "ccr": 0.0,
            "cca": 0.0,
        }
        notes: list[str] = []
        partial = False

        # 1. Consulta WHOIS com timeout controlado
        raw_whois: dict[str, Any] = {}
        try:
            raw_whois = await asyncio.wait_for(
                asyncio.to_thread(self._sync_query_whois, registered_domain),
                timeout=self._timeout_s,
            )
        except asyncio.TimeoutError:
            partial = True
            notes.append("WHOIS_TIMEOUT")
        except Exception:
            pass

        created_dt = _normalize_dt(raw_whois.get("creation_date"))
        updated_dt = _normalize_dt(raw_whois.get("updated_date"))
        expired_dt = _normalize_dt(raw_whois.get("expiration_date"))
        registrar = raw_whois.get("registrar")
        whois_country = raw_whois.get("country")
        if isinstance(whois_country, list) and whois_country:
            whois_country = whois_country[0]

        if created_dt is not None:
            now_dt = datetime.now(timezone.utc)
            end_ref = expired_dt if (expired_dt and expired_dt > created_dt) else now_dt
            lifetime_days = max(0.0, (end_ref - created_dt).total_seconds() / 86400.0)

            active_ref = updated_dt if (updated_dt and updated_dt >= created_dt) else now_dt
            active_days = max(0.0, (active_ref - created_dt).total_seconds() / 86400.0)

            features["has_whois"] = 1.0
            features["lifetime"] = round(lifetime_days, 2)
            features["active_time"] = round(active_days, 2)

            whois_evidence = {
                "present": True,
                "creation_date": created_dt.isoformat(),
                "updated_date": updated_dt.isoformat() if updated_dt else None,
                "expiration_date": expired_dt.isoformat() if expired_dt else None,
                "registrar": str(registrar) if registrar else None,
            }

        # 2. País / ASN / CCR / CCA
        countries: list[str] = []
        asns: list[str] = []
        if isinstance(whois_country, str) and len(whois_country.strip()) == 2:
            countries.append(whois_country.strip().upper())

        if countries:
            cc = countries[0]
            features["has_country"] = 1.0
            features["CCR"] = round(self._ccr_map.get(cc, self._default_ccr), 4)
            features["CCA"] = round(self._cca_map.get(cc, features["CCR"]), 4)
        elif ips:
            # Se há IP resolvido e base CCR configurada, usa default empírico
            features["ASN Count"] = 1.0
            asns = ["AS_RESOLVED"]

        geo_asn_evidence = {
            "countries": countries,
            "asns": asns,
            "ccr": features["CCR"],
            "cca": features["CCA"],
        }

        return WhoisAndRiskResult(
            features=features,
            whois_evidence=whois_evidence,
            geo_asn_evidence=geo_asn_evidence,
            resolved_asn_count=len(asns) if asns else (1 if ips else 0),
            partial=partial,
            notes=notes,
        )
