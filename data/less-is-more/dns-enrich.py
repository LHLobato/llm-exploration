import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse

import dns.resolver
import numpy as np
import pandas as pd
from tqdm import tqdm

DOMAIN_FEATURE_ORDER = [
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
]

_csv_lock = threading.Lock()


def clean_domain(url):
    """
    Extrai apenas o domínio puro de uma URL suja.
    Remove http://, caminhos, parâmetros e portas.
    """
    url = str(url).strip()
    
    # Se a string não tiver protocolo, o urlparse pode confundir o domínio com o 'path'.
    # Injetamos um // genérico para forçar o reconhecimento correto do 'netloc'.
    if not url.startswith(('http://', 'https://', '//')):
        url = '//' + url
        
    parsed = urlparse(url)
    
    # netloc pega o domínio. O split(':') arranca a porta caso exista (ex: google.com:8080)
    dominio_limpo = parsed.netloc.split(':')[0]
    
    return dominio_limpo


def extract_live_dns_features(domain):
    features = {k: 0.0 for k in DOMAIN_FEATURE_ORDER}

    # Cada thread precisa do seu próprio resolver (não é thread-safe compartilhar)
    resolver = dns.resolver.Resolver()
    resolver.timeout = 1
    resolver.lifetime = 1

    def calculate_ttl_stats(answers):
        if not answers:
            return 0, 0, 0
        try:
            ttl = float(answers.rrset.ttl)
        except AttributeError:
            ttl = float(answers.ttl) if hasattr(answers, "ttl") else 0.0
        return len(answers), ttl, 0.0

    try:
        a_records = resolver.resolve(domain, "A")
        features["A Count"] = len(a_records)
        features["IP Count"] = len(a_records)
        features["A Missing"] = 0
        count, mean, std = calculate_ttl_stats(a_records)
        features["A TTL Count"] = count
        features["A Medium TTL"] = mean
        features["A TTL Standard Deviation"] = std
        features["ASN Count"] = 1
    except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN, dns.exception.Timeout):
        features["A Missing"] = 1

    try:
        aaaa_records = resolver.resolve(domain, "AAAA")
        features["AAAA Count"] = len(aaaa_records)
        _, mean, _ = calculate_ttl_stats(aaaa_records)
        features["AAAA Medium TTL"] = mean
    except:
        pass

    try:
        mx_records = resolver.resolve(domain, "MX")
        features["MX Count"] = len(mx_records)
        features["MX Missing"] = 0
        _, mean, _ = calculate_ttl_stats(mx_records)
        features["MX Medium TTL"] = mean
    except:
        features["MX Missing"] = 1

    try:
        ns_records = resolver.resolve(domain, "NS")
        features["NS Count"] = len(ns_records)
    except:
        pass

    try:
        cname_records = resolver.resolve(domain, "CNAME")
        features["CNAME Count"] = len(cname_records)
    except:
        pass

    soa_found = False
    current_domain_search = domain
    for _ in range(3):
        try:
            soa_records = resolver.resolve(current_domain_search, "SOA")
            soa = soa_records[0]
            features["SOA Missing"] = 0
            features["SOA Retry"] = float(soa.retry)
            features["SOA Refresh"] = float(soa.refresh)
            features["SOA Minimum"] = float(soa.minimum)
            features["SOA Expire Length"] = float(soa.expire)
            features["SOA Serial Length"] = len(str(soa.serial))
            _, mean, std = calculate_ttl_stats(soa_records)
            features["SOA TTL Count"] = 1
            features["SOA Medium TTL"] = mean
            features["SOA TTL Standard Deviation"] = std
            soa_found = True
            break
        except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN):
            parts = current_domain_search.split(".")
            if len(parts) > 2:
                current_domain_search = ".".join(parts[1:])
            else:
                break
        except:
            break

    if not soa_found:
        features["SOA Missing"] = 1

    return features


def _process_one(domain, label):
    """Processa um único domínio (já limpo) — função executada por cada thread."""
    try:
        feats = extract_live_dns_features(domain)
    except Exception:
        feats = {k: 0.0 for k in DOMAIN_FEATURE_ORDER}
        feats["A Missing"] = 1
        feats["MX Missing"] = 1
        feats["SOA Missing"] = 1
    
    # Retorna o domínio limpo que será gravado no CSV de saída
    return {"name": domain, "label": label, **feats}


def collect_dns_features(
    input_csv: str, output_csv: str, label: int, max_workers: int = 50
):
    """
    Lê as URLs de input_csv, limpa para domínios puros, extrai features DNS 
    em paralelo e salva em output_csv.
    """
    import os # Importado localmente apenas para checar o arquivo
    
    # O on_bad_lines='skip' protege contra aquelas linhas zoadas da base do artigo
    df = pd.read_csv(input_csv, index_col=False, on_bad_lines='skip')
    
    # Se a primeira coluna não tiver nome, assume que é o 'name' (comum em CSVs acadêmicos)
    if 'name' not in df.columns:
        df.rename(columns={df.columns[0]: 'name'}, inplace=True)
    
    df['name'] = df['name'].astype(str).str.strip()
    
    # 1. Limpa todos os domínios
    print(f"[{input_csv}] Limpando URLs...")
    df['clean_domain'] = df['name'].apply(clean_domain)
    
    # 2. Remove duplicatas e domínios em branco
    unique_domains = df[df['clean_domain'] != '']['clean_domain'].unique()

    # 3. Retomada: pula domínios já processados
    already_done = set()
    if os.path.exists(output_csv):
        try:
            done_df = pd.read_csv(output_csv, usecols=["name"])
            already_done = set(done_df["name"].astype(str).values)
            print(f"[checkpoint] {len(already_done)} domínios já processados em '{output_csv}', retomando...")
        except ValueError:
            pass # Arquivo existe mas tá vazio ou sem a coluna 'name'

    pending = [d for d in unique_domains if d not in already_done]
    print(f"[{input_csv}] {len(pending)} domínios únicos para processar (total de URLs={len(df)}, workers={max_workers})")

    if not pending:
        print("Nada a fazer.")
        return

    # Controla se o header já foi escrito
    write_header = not os.path.exists(output_csv)
    header_written = threading.Event()
    if not write_header:
        header_written.set()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_process_one, domain, label): domain for domain in pending
        }

        with tqdm(total=len(pending), desc=f"Coletando {input_csv}") as pbar:
            for future in as_completed(futures):
                row = future.result()

                with _csv_lock:
                    needs_header = not header_written.is_set()
                    pd.DataFrame([row]).to_csv(
                        output_csv, mode="a", header=needs_header, index=False
                    )
                    if needs_header:
                        header_written.set()

                pbar.update(1)

    print(f"Salvo em '{output_csv}'\n")


if __name__ == "__main__":
    collect_dns_features(
        input_csv="benign_5345_features.csv",
        output_csv="benign_dns_live.csv",
        label=0,
        max_workers=50,
    )

    collect_dns_features(
        input_csv="malicious_1356_features.csv", 
        output_csv="mal_dns_live.csv",
        label=1,
        max_workers=50,
    )

    print("Concluído. Arquivos gerados: benign_dns_live.csv, mal_dns_live.csv")