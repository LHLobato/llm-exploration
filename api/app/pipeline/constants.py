"""Constantes canônicas dos 42 atributos, grupos e instruções do LLM (doc 06 e pasta esemble/)."""

from __future__ import annotations

# Grupo A — 13 Atributos Lexicais (Índices 0 a 12)
LEXICAL_FEATURE_ORDER: tuple[str, ...] = (
    "lex_length",
    "lex_digit_count",
    "lex_digit_ratio",
    "lex_vowel_ratio",
    "lex_consonant_ratio",
    "lex_special_char_ratio",
    "lex_entropy",
    "lex_hyphen_count",
    "lex_starts_with_digit",
    "lex_has_repeated_chars",
    "lex_max_consonant_seq",
    "lex_max_digit_seq",
    "lex_unique_char_ratio",
)

# Grupo B — 23 Atributos DNS, TTL e SOA (Índices 13 a 35)
# Ordem idêntica a DOMAIN_FEATURE_ORDER em data/less-is-more/dns-enrich.py e data/dataset.csv
DNS_FEATURE_ORDER: tuple[str, ...] = (
    "A Count",
    "IP Count",
    "MX Count",
    "NS Count",
    "A Missing",
    "ASN Count",
    "SOA Retry",
    "AAAA Count",
    "MX Missing",
    "A TTL Count",
    "CNAME Count",
    "SOA Minimum",
    "SOA Missing",
    "SOA Refresh",
    "A Medium TTL",
    "MX Medium TTL",
    "SOA TTL Count",
    "SOA Medium TTL",
    "AAAA Medium TTL",
    "SOA Expire Length",
    "SOA Serial Length",
    "A TTL Standard Deviation",
    "SOA TTL Standard Deviation",
)

# Grupo C — 6 Atributos WHOIS e Risco (Índices 36 a 41)
WHOIS_RISK_FEATURE_ORDER: tuple[str, ...] = (
    "has_whois",
    "lifetime",
    "active_time",
    "has_country",
    "CCR",
    "CCA",
)

# Ordem Canônica dos 42 Atributos de Entrada do MLP ([42, 256, 128, 64, 2]) e do GADF 42×42
CANONICAL_42_FEATURES: tuple[str, ...] = (
    *LEXICAL_FEATURE_ORDER,
    *DNS_FEATURE_ORDER,
    *WHOIS_RISK_FEATURE_ORDER,
)

assert len(CANONICAL_42_FEATURES) == 42, "O vetor canônico deve ter exatamente 42 atributos."

# Subgrupos para formatação estruturada na resposta JSON da API (doc 05)
DNS_COUNTS_AND_FLAGS_KEYS: tuple[str, ...] = (
    "A Count",
    "IP Count",
    "MX Count",
    "NS Count",
    "A Missing",
    "ASN Count",
    "AAAA Count",
    "MX Missing",
    "CNAME Count",
    "SOA Missing",
)

TTL_STATS_KEYS: tuple[str, ...] = (
    "A TTL Count",
    "A Medium TTL",
    "A TTL Standard Deviation",
    "MX Medium TTL",
    "AAAA Medium TTL",
    "SOA TTL Count",
    "SOA Medium TTL",
    "SOA TTL Standard Deviation",
)

SOA_PARAMS_KEYS: tuple[str, ...] = (
    "SOA Refresh",
    "SOA Retry",
    "SOA Minimum",
    "SOA Expire Length",
    "SOA Serial Length",
)

# System Instruction exata de esemble/multimodal.py (linhas 36-39)
SYSTEM_INSTRUCTION: str = (
    "You are a cybersecurity classifier. "
    "Analyze the domain record below and classify it as benign or malicious."
)

LLM_MAX_LENGTH: int = 160
GADF_IMAGE_SIZE: int = 42
GADF_IN_CHANNELS: int = 3
