"""Construção do prompt estruturado de 7 seções (`data/test-prompt.py`) e template de chat (`esemble/multimodal.py`)."""

from __future__ import annotations

from typing import Any, Mapping

from app.pipeline.constants import SYSTEM_INSTRUCTION


def fmt_bool(val: Any) -> str:
    """Converte valores booleanos/numéricos/strings para `'yes'` ou `'no'` (`data/test-prompt.py`)."""
    if isinstance(val, str):
        return "yes" if val.strip().lower() in {"true", "1", "yes"} else "no"
    try:
        return "yes" if bool(float(val)) else "no"
    except (ValueError, TypeError):
        return "no"


def fmt_float(val: Any, decimals: int = 4) -> str:
    """Formata float arredondado com `decimals` casas decimais."""
    try:
        return str(round(float(val), decimals))
    except (ValueError, TypeError):
        return "0"


def fmt_int(val: Any) -> str:
    """Formata valor numérico como inteiro truncado."""
    try:
        return str(int(float(val)))
    except (ValueError, TypeError):
        return "0"


def build_structured_prompt(domain_name: str, features: Mapping[str, Any]) -> str:
    """
    Constrói o prompt estruturado de 7 seções idêntico a `build_structured_prompt`
    em `data/test-prompt.py` (linhas 39-124):
    1. `[name]`
    2. `[entropy]`
    3. `[whois]`
    4. `[risk]`
    5. `[dns]`
    6. `[ttl]`
    7. `[soa]`
    """
    lines: list[str] = []

    # 1. [name]
    lines.append(f"[name]: {domain_name}")

    # 2. [entropy]
    lines.append(f"[entropy]: {fmt_float(features.get('lex_entropy', 0.0), 4)}")

    # 3. [whois]
    raw_whois = features.get("has_whois", 0.0)
    try:
        is_whois = bool(float(raw_whois)) if raw_whois is not None else False
    except (ValueError, TypeError):
        is_whois = str(raw_whois).strip().lower() == "true"

    if is_whois:
        lifetime = fmt_float(features.get("lifetime", 0.0), 2)
        active = fmt_float(features.get("active_time", 0.0), 2)
        lines.append(f"[whois]: present=yes | lifetime={lifetime}d active={active}d")
    else:
        lines.append("[whois]: present=no")

    # 4. [risk]
    has_country = fmt_bool(features.get("has_country", 0.0))
    ccr = fmt_float(features.get("CCR", 0.0), 4)
    cca = fmt_float(features.get("CCA", 0.0), 4)
    lines.append(f"[risk]: ccr={ccr} cca={cca} country={has_country}")

    # 5. [dns]
    a_count = fmt_int(features.get("A Count", 0))
    aaaa_count = fmt_int(features.get("AAAA Count", 0))
    mx_count = fmt_int(features.get("MX Count", 0))
    ns_count = fmt_int(features.get("NS Count", 0))
    cname = fmt_int(features.get("CNAME Count", 0))
    ip_count = fmt_int(features.get("IP Count", 0))
    asn_count = fmt_int(features.get("ASN Count", 0))
    a_miss = fmt_bool(features.get("A Missing", 0))
    mx_miss = fmt_bool(features.get("MX Missing", 0))
    lines.append(
        f"[dns]: A={a_count} AAAA={aaaa_count} MX={mx_count} NS={ns_count} "
        f"CNAME={cname} IPs={ip_count} ASNs={asn_count} "
        f"a_missing={a_miss} mx_missing={mx_miss}"
    )

    # 6. [ttl]
    a_med = fmt_float(features.get("A Medium TTL", 0.0), 1)
    a_std = fmt_float(features.get("A TTL Standard Deviation", 0.0), 1)
    a_ttlc = fmt_int(features.get("A TTL Count", 0))
    mx_med = fmt_float(features.get("MX Medium TTL", 0.0), 1)
    aaaa_med = fmt_float(features.get("AAAA Medium TTL", 0.0), 1)
    lines.append(
        f"[ttl]: a_med={a_med} a_std={a_std} a_distinct={a_ttlc} "
        f"mx_med={mx_med} aaaa_med={aaaa_med}"
    )

    # 7. [soa]
    soa_miss = fmt_bool(features.get("SOA Missing", 0))
    soa_refresh = fmt_int(features.get("SOA Refresh", 0))
    soa_retry = fmt_int(features.get("SOA Retry", 0))
    soa_min = fmt_int(features.get("SOA Minimum", 0))
    soa_exp_len = fmt_int(features.get("SOA Expire Length", 0))
    soa_ser_len = fmt_int(features.get("SOA Serial Length", 0))
    soa_med = fmt_float(features.get("SOA Medium TTL", 0.0), 1)
    soa_std = fmt_float(features.get("SOA TTL Standard Deviation", 0.0), 1)
    soa_ttlc = fmt_int(features.get("SOA TTL Count", 0))
    lines.append(
        f"[soa]: missing={soa_miss} refresh={soa_refresh} retry={soa_retry} "
        f"min={soa_min} expire_len={soa_exp_len} serial_len={soa_ser_len} "
        f"med={soa_med} std={soa_std} distinct={soa_ttlc}"
    )

    return "\n".join(lines)


def format_llm_chat_input(structured_prompt: str, tokenizer: Any | None = None) -> str:
    """
    Aplica a `SYSTEM_INSTRUCTION` e o `apply_chat_template` conforme `TextInferenceDataset`
    em `esemble/multimodal.py` (linhas 60-66).
    """
    user_content = f"{SYSTEM_INSTRUCTION}\n\n{structured_prompt}"
    messages = [{"role": "user", "content": user_content}]
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return user_content
