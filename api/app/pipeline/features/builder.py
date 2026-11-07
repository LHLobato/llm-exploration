"""Montagem do vetor canônico de 42 atributos, escalonamento MinMaxScaler e agrupamento JSON."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from app.pipeline.constants import (
    CANONICAL_42_FEATURES,
    DNS_COUNTS_AND_FLAGS_KEYS,
    LEXICAL_FEATURE_ORDER,
    SOA_PARAMS_KEYS,
    TTL_STATS_KEYS,
    WHOIS_RISK_FEATURE_ORDER,
)

# Limites mínimos e máximos empíricos das 42 colunas extraídos de `data/acme/train.csv` e `dataset.csv`
# Utilizados como fallback determinístico quando `tabular_scaler.joblib` não está montado no ambiente de teste.
_EMPIRICAL_MIN_MAX: dict[str, tuple[float, float]] = {
    "lex_length": (4.0, 253.0),
    "lex_digit_count": (0.0, 60.0),
    "lex_digit_ratio": (0.0, 1.0),
    "lex_vowel_ratio": (0.0, 1.0),
    "lex_consonant_ratio": (0.0, 1.0),
    "lex_special_char_ratio": (0.0, 0.5),
    "lex_entropy": (0.0, 5.5),
    "lex_hyphen_count": (0.0, 20.0),
    "lex_starts_with_digit": (0.0, 1.0),
    "lex_has_repeated_chars": (0.0, 1.0),
    "lex_max_consonant_seq": (0.0, 30.0),
    "lex_max_digit_seq": (0.0, 30.0),
    "lex_unique_char_ratio": (0.0, 1.0),
    "A Count": (0.0, 32.0),
    "IP Count": (0.0, 32.0),
    "MX Count": (0.0, 20.0),
    "NS Count": (0.0, 20.0),
    "A Missing": (0.0, 1.0),
    "ASN Count": (0.0, 10.0),
    "SOA Retry": (0.0, 86400.0),
    "AAAA Count": (0.0, 20.0),
    "MX Missing": (0.0, 1.0),
    "A TTL Count": (0.0, 20.0),
    "CNAME Count": (0.0, 10.0),
    "SOA Minimum": (0.0, 604800.0),
    "SOA Missing": (0.0, 1.0),
    "SOA Refresh": (0.0, 604800.0),
    "A Medium TTL": (0.0, 604800.0),
    "MX Medium TTL": (0.0, 604800.0),
    "SOA TTL Count": (0.0, 5.0),
    "SOA Medium TTL": (0.0, 604800.0),
    "AAAA Medium TTL": (0.0, 604800.0),
    "SOA Expire Length": (0.0, 10.0),
    "SOA Serial Length": (0.0, 12.0),
    "A TTL Standard Deviation": (0.0, 86400.0),
    "SOA TTL Standard Deviation": (0.0, 604800.0),
    "has_whois": (0.0, 1.0),
    "lifetime": (0.0, 15000.0),
    "active_time": (0.0, 15000.0),
    "has_country": (0.0, 1.0),
    "CCR": (0.0, 1.0),
    "CCA": (0.0, 1.0),
}


@dataclass(frozen=True)
class AssembledFeatures:
    """Agrupa o dicionário plano (42 chaves), o vetor ordenado bruto, o vetor escalonado e o JSON agrupado."""

    flat_features: dict[str, float]
    raw_vector_42: list[float]
    scaled_vector_42: list[float]
    grouped_features: dict[str, dict[str, float | int]]


class FeatureVectorBuilder:
    """
    Constrói o vetor canônico de 42 atributos na ordem exata de `CANONICAL_42_FEATURES`
    e aplica o escalonamento `MinMaxScaler` para entrada no `MultiLayerPerceptron` e no GADF 42×42.
    """

    def __init__(self, scaler_path: Path | None = None) -> None:
        self._scaler: Any = None
        if scaler_path and scaler_path.is_file():
            try:
                import joblib  # type: ignore

                self._scaler = joblib.load(scaler_path)
            except Exception:
                self._scaler = None

    def scale_vector(self, raw_vector_42: list[float]) -> list[float]:
        """Normaliza o vetor de 42 atributos para `[0.0, 1.0]`."""
        if len(raw_vector_42) != 42:
            raise ValueError(f"Esperados 42 atributos, recebidos {len(raw_vector_42)}.")

        if self._scaler is not None:
            try:
                import numpy as np  # type: ignore

                arr = np.asarray([raw_vector_42], dtype=np.float32)
                transformed = self._scaler.transform(arr)[0]
                return [float(max(0.0, min(1.0, v))) for v in transformed]
            except Exception:
                pass

        scaled: list[float] = []
        for idx, name in enumerate(CANONICAL_42_FEATURES):
            val = float(raw_vector_42[idx])
            lo, hi = _EMPIRICAL_MIN_MAX.get(name, (0.0, 1.0))
            if hi <= lo:
                scaled.append(0.0)
            else:
                norm = (val - lo) / (hi - lo)
                scaled.append(float(max(0.0, min(1.0, norm))))
        return scaled

    def assemble(
        self,
        lexical: Mapping[str, float],
        dns_features: Mapping[str, float],
        whois_risk: Mapping[str, float],
    ) -> AssembledFeatures:
        """Combina os 3 grupos de atributos garantindo a ordem canônica dos 42 atributos."""
        combined: dict[str, float] = {}
        for key in LEXICAL_FEATURE_ORDER:
            combined[key] = float(lexical.get(key, 0.0))
        for key in CANONICAL_42_FEATURES[13:36]:
            combined[key] = float(dns_features.get(key, 0.0))
        for key in WHOIS_RISK_FEATURE_ORDER:
            combined[key] = float(whois_risk.get(key, 0.0))

        raw_vector_42 = [combined[name] for name in CANONICAL_42_FEATURES]
        scaled_vector_42 = self.scale_vector(raw_vector_42)

        grouped = {
            "lexical": {
                k: (int(combined[k]) if k in {
                    "lex_length",
                    "lex_digit_count",
                    "lex_hyphen_count",
                    "lex_starts_with_digit",
                    "lex_has_repeated_chars",
                    "lex_max_consonant_seq",
                    "lex_max_digit_seq",
                } else round(combined[k], 4))
                for k in LEXICAL_FEATURE_ORDER
            },
            "dns_counts_and_flags": {
                k: int(round(combined[k])) for k in DNS_COUNTS_AND_FLAGS_KEYS
            },
            "ttl_stats": {
                k: (int(round(combined[k])) if k in {"A TTL Count", "SOA TTL Count"} else round(combined[k], 2))
                for k in TTL_STATS_KEYS
            },
            "soa_params": {
                k: int(round(combined[k])) for k in SOA_PARAMS_KEYS
            },
            "whois_and_risk": {
                k: round(combined[k], 4) for k in WHOIS_RISK_FEATURE_ORDER
            },
        }

        return AssembledFeatures(
            flat_features=combined,
            raw_vector_42=raw_vector_42,
            scaled_vector_42=scaled_vector_42,
            grouped_features=grouped,
        )
