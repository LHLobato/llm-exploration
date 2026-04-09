"""
rebuild_prompts.py
------------------
Reconstrói os prompts do CSV de forma estruturada e compacta,
reduzindo de ~500 tokens para ~80-120 tokens por amostra.

Uso:
    python rebuild_prompts.py --input seu_dataset.csv --output dataset_structured.csv
    python rebuild_prompts.py --input seu_dataset.csv --output dataset_structured.csv --preview 5
"""

import pandas as pd
import argparse


# ── Helpers ────────────────────────────────────────────────────────────────────

def fmt_bool(val) -> str:
    """Converte valores booleanos/strings para sim/não."""
    if isinstance(val, str):
        val = val.strip().lower() == "true"
    return "yes" if val else "no"

def fmt_float(val, decimals=4) -> str:
    try:
        return str(round(float(val), decimals))
    except (ValueError, TypeError):
        return "0"

def fmt_int(val) -> str:
    try:
        return str(int(float(val)))
    except (ValueError, TypeError):
        return "0"


# ── Prompt builder ─────────────────────────────────────────────────────────────

def build_structured_prompt(row: pd.Series) -> str:
    """
    Constrói prompt estruturado com tags de campo.
    
    Formato alvo (~80-120 tokens):
    
    [name]: example.com
    [entropy]: 3.24
    [whois]: missing | lifetime=13.3d active=12.3d
    [risk]: ccr=0.863 cca=0.863 country=yes
    [dns]: A=1 AAAA=0 MX=5 NS=2 CNAME=0 IPs=1 ASNs=1
    [ttl]: a_med=0 a_std=0 mx_med=1800 aaaa_med=0
    [soa]: missing=no refresh=43200 retry=3600 min=3601 expire_len=6 serial_len=10 med=3601 std=3601
    """

    lines = []

    # [name]
    lines.append(f"[name]: {row['name']}")

    # [entropy]
    lines.append(f"[entropy]: {fmt_float(row['lex_entropy'])}")

    # [whois] — só inclui datas se o whois estiver presente
    raw_whois = row.get('has_whois', False)
    whois_present = fmt_bool(raw_whois)
    # CORREÇÃO: has_whois é 1.0/0.0 (float), não string
    is_whois = bool(float(raw_whois)) if raw_whois is not None else False

    if is_whois:
        lifetime = fmt_float(row.get('lifetime', 0), 2)
        active   = fmt_float(row.get('active_time', 0), 2)
        lines.append(f"[whois]: present=yes | lifetime={lifetime}d active={active}d")
    else:
        lines.append(f"[whois]: present=no")

    # [risk] — ratios de país/ASN
    has_country = fmt_bool(row.get('has_country', False))
    ccr = fmt_float(row.get('CCR', 0))
    cca = fmt_float(row.get('CCA', 0))
    lines.append(f"[risk]: ccr={ccr} cca={cca} country={has_country}")

    # [dns] — contagens de registros
    a_count    = fmt_int(row.get('A Count', 0))
    aaaa_count = fmt_int(row.get('AAAA Count', 0))
    mx_count   = fmt_int(row.get('MX Count', 0))
    ns_count   = fmt_int(row.get('NS Count', 0))
    cname      = fmt_int(row.get('CNAME Count', 0))
    ip_count   = fmt_int(row.get('IP Count', 0))
    asn_count  = fmt_int(row.get('ASN Count', 0))
    a_miss     = fmt_bool(row.get('A Missing', 0))
    mx_miss    = fmt_bool(row.get('MX Missing', 0))
    lines.append(
        f"[dns]: A={a_count} AAAA={aaaa_count} MX={mx_count} NS={ns_count} "
        f"CNAME={cname} IPs={ip_count} ASNs={asn_count} "
        f"a_missing={a_miss} mx_missing={mx_miss}"
    )

    # [ttl] — comportamento de cache
    a_med   = fmt_float(row.get('A Medium TTL', 0), 1)
    a_std   = fmt_float(row.get('A TTL Standard Deviation', 0), 1)
    mx_med  = fmt_float(row.get('MX Medium TTL', 0), 1)
    aaaa_med= fmt_float(row.get('AAAA Medium TTL', 0), 1)
    a_ttlc  = fmt_int(row.get('A TTL Count', 0))
    lines.append(
        f"[ttl]: a_med={a_med} a_std={a_std} a_distinct={a_ttlc} "
        f"mx_med={mx_med} aaaa_med={aaaa_med}"
    )

    # [soa] — parâmetros SOA
    soa_miss    = fmt_bool(row.get('SOA Missing', 0))
    soa_refresh = fmt_int(row.get('SOA Refresh', 0))
    soa_retry   = fmt_int(row.get('SOA Retry', 0))
    soa_min     = fmt_int(row.get('SOA Minimum', 0))
    soa_exp_len = fmt_int(row.get('SOA Expire Length', 0))
    soa_ser_len = fmt_int(row.get('SOA Serial Length', 0))
    soa_med     = fmt_float(row.get('SOA Medium TTL', 0), 1)
    soa_std     = fmt_float(row.get('SOA TTL Standard Deviation', 0), 1)
    soa_ttlc    = fmt_int(row.get('SOA TTL Count', 0))
    lines.append(
        f"[soa]: missing={soa_miss} refresh={soa_refresh} retry={soa_retry} "
        f"min={soa_min} expire_len={soa_exp_len} serial_len={soa_ser_len} "
        f"med={soa_med} std={soa_std} distinct={soa_ttlc}"
    )

    return "\n".join(lines)


# ── Token counter (estimativa sem carregar modelo) ─────────────────────────────

def estimate_tokens(text: str) -> int:
    """Estimativa rápida: ~1 token por 4 caracteres (regra de bolso para BERT-like)."""
    return max(1, len(text) // 4)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Reconstrói prompts estruturados a partir do CSV.")
    parser.add_argument("--input",   type=str, required=True,  help="CSV de entrada")
    parser.add_argument("--output",  type=str, required=True,  help="CSV de saída com nova coluna 'prompt'")
    parser.add_argument("--preview", type=int, default=0,      help="Imprime N exemplos antes de salvar")
    args = parser.parse_args()

    print(f"Lendo {args.input}...")
    df = pd.read_csv(args.input)

    print("Reconstruindo prompts...")
    df["prompt"] = df.apply(build_structured_prompt, axis=1)

    # Estatísticas de tokens
    df["_token_estimate"] = df["prompt"].apply(estimate_tokens)
    print(f"\n── Estatísticas de tokens estimados ──")
    print(f"  Média  : {df['_token_estimate'].mean():.1f}")
    print(f"  Mediana: {df['_token_estimate'].median():.1f}")
    print(f"  Máx    : {df['_token_estimate'].max()}")
    print(f"  Min    : {df['_token_estimate'].min()}")
    print(f"  >256   : {(df['_token_estimate'] > 256).sum()} amostras ({(df['_token_estimate'] > 256).mean():.1%})")
    print(f"  >128   : {(df['_token_estimate'] > 128).sum()} amostras ({(df['_token_estimate'] > 128).mean():.1%})")

    df = df.drop(columns=["_token_estimate"])

    if args.preview > 0:
        print(f"\n── {args.preview} exemplos ──")
        for _, row in df.head(args.preview).iterrows():
            label = "MALICIOSO" if row["malicious"] == 1 else "BENIGNO"
            print(f"\n[{label}] {row['name']}")
            print(row["prompt"])
            print("-" * 60)

    df.to_csv(args.output, index=False)
    print(f"\n✓ Salvo em {args.output} ({len(df)} linhas)")


if __name__ == "__main__":
    main()