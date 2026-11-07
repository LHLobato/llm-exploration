"""Validação e normalização estrita de nomes de domínio (doc 05 e doc 06)."""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass

from app.core.exceptions import InvalidDomainError

_LABEL_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_RESERVED_TLDS = {"local", "internal", "arpa", "test", "example", "invalid", "localhost"}
_MULTI_PART_PUBLIC_SUFFIXES = {
    "co.uk",
    "org.uk",
    "gov.uk",
    "ac.uk",
    "com.br",
    "net.br",
    "org.br",
    "gov.br",
    "edu.br",
    "com.au",
    "co.jp",
    "co.kr",
    "co.in",
    "com.mx",
    "com.ar",
    "co.za",
}


@dataclass(frozen=True)
class NormalizedDomain:
    """Resultado da validação e normalização de um domínio de entrada."""

    domain: str
    registered_domain: str


def clean_domain(raw: str) -> str:
    """Limpa espaços, converte para minúsculas e remove ponto final de FQDN."""
    return (raw or "").strip().lower().rstrip(".")


def remove_www_prefix(domain: str) -> str:
    """Remove o prefixo `www.` mantendo paridade com `data/filterdata.py` e `data/enriching.py`."""
    if domain.startswith("www."):
        return domain[4:]
    return domain


def extract_registered_domain(domain: str) -> str:
    """Extrai o domínio registrável (eTLD+1) usando `tldextract` se instalado ou heurística PSL."""
    try:
        import tldextract  # type: ignore

        ext = tldextract.extract(domain)
        if ext.domain and ext.suffix:
            return f"{ext.domain}.{ext.suffix}"
    except Exception:
        pass

    parts = domain.split(".")
    if len(parts) >= 3:
        last_two = f"{parts[-2]}.{parts[-1]}"
        if last_two in _MULTI_PART_PUBLIC_SUFFIXES:
            return f"{parts[-3]}.{last_two}"
    if len(parts) >= 2:
        return f"{parts[-2]}.{parts[-1]}"
    return domain


def normalize_and_validate_domain(raw_input: str) -> NormalizedDomain:
    """
    Valida que a string de entrada contém **apenas um nome de domínio válido** (doc 05):
    1. Rejeita strings vazias, protocolos (`http://`, `https://`, `//`), caminhos (`/`),
       query (`?`), fragmento (`#`), credenciais (`@`), portas (`:`), espaços, wildcards (`*`)
       ou endereços IPv4/IPv6 literais.
    2. Converte para minúsculas, remove ponto final e prefixo `www.`.
    3. Converte IDN para ASCII/Punycode (`idna`).
    4. Valida comprimento total (<= 253), labels RFC 1035 e bloqueia sufixos internos/reservados.
    """
    if not isinstance(raw_input, str):
        raise InvalidDomainError()

    raw = raw_input.strip()
    if not raw:
        raise InvalidDomainError("O nome do domínio não pode ser vazio.")

    # Bloqueia URLs, caminhos, portas, credenciais, query strings, espaços ou wildcards
    forbidden_substrings = ("://", "//", "/", "?", "#", "@", ":", " ", "\t", "\n", "*", ",")
    if any(token in raw for token in forbidden_substrings):
        raise InvalidDomainError(
            "Informe apenas o nome do domínio (sem protocolo http/https, caminho, porta ou parâmetros)."
        )

    cleaned = remove_www_prefix(clean_domain(raw))
    if not cleaned:
        raise InvalidDomainError()

    # Rejeita literais IPv4 / IPv6
    try:
        ipaddress.ip_address(cleaned.strip("[]"))
        raise InvalidDomainError("Endereços IP não são aceitos; informe um nome de domínio.")
    except ValueError:
        pass

    # Converte IDN (Unicode) para ASCII/Punycode
    try:
        ascii_domain = cleaned.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise InvalidDomainError("Domínio internacionalizado (IDN) inválido.") from exc

    if len(ascii_domain) > 253 or "." not in ascii_domain:
        raise InvalidDomainError("Formato de domínio inválido.")

    labels = ascii_domain.split(".")
    if len(labels) < 2:
        raise InvalidDomainError("O domínio deve conter pelo menos um sufixo público (TLD).")

    tld = labels[-1]
    if tld in _RESERVED_TLDS or ascii_domain in _RESERVED_TLDS:
        raise InvalidDomainError("Domínios internos ou reservados não são permitidos.")

    if tld.isdigit() or len(tld) < 2:
        raise InvalidDomainError("Sufixo de domínio (TLD) inválido.")

    for label in labels:
        if not label or len(label) > 63 or not _LABEL_RE.match(label):
            raise InvalidDomainError(f"Rótulo de domínio inválido: '{label}'.")

    registered = extract_registered_domain(ascii_domain)
    return NormalizedDomain(domain=ascii_domain, registered_domain=registered)
