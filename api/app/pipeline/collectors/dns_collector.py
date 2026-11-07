"""Coletor DNS ao vivo (`A`, `AAAA`, `MX`, `NS`, `CNAME`, `SOA`) baseado em `data/less-is-more/dns-enrich.py`."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from app.pipeline.constants import DNS_FEATURE_ORDER


@dataclass
class DnsCollectionResult:
    """Resultado da coleta DNS contendo os 23 atributos e as evidências brutas."""

    features: dict[str, float]
    evidence: dict[str, Any]
    resolved_ips: list[str] = field(default_factory=list)
    partial: bool = False
    notes: list[str] = field(default_factory=list)


def default_empty_dns_features() -> dict[str, float]:
    """Valores padrão quando um domínio não possui registros DNS resolvidos."""
    base = {key: 0.0 for key in DNS_FEATURE_ORDER}
    base["A Missing"] = 1.0
    base["MX Missing"] = 1.0
    base["SOA Missing"] = 1.0
    return base


class DnsCollector:
    """
    Executa consultas DNS para `A`, `AAAA`, `MX`, `NS`, `CNAME` e `SOA` (com caminhamento
    hierárquico em até 3 níveis para `SOA`) e calcula os 23 atributos de `DNS_FEATURE_ORDER`.
    """

    def __init__(self, timeout_s: float = 1.0, lifetime_s: float = 1.0) -> None:
        self._timeout_s = timeout_s
        self._lifetime_s = lifetime_s

    def _sync_collect(self, domain: str) -> DnsCollectionResult:
        features = default_empty_dns_features()
        evidence: dict[str, Any] = {
            "a_records": [],
            "aaaa_records": [],
            "mx_records": [],
            "ns_records": [],
            "cname_records": [],
            "soa": None,
        }
        resolved_ips: list[str] = []
        notes: list[str] = []
        partial = False

        try:
            import dns.exception  # type: ignore
            import dns.resolver  # type: ignore
        except ImportError:
            notes.append("DNSPYTHON_UNAVAILABLE")
            return DnsCollectionResult(
                features=features,
                evidence=evidence,
                resolved_ips=[],
                partial=True,
                notes=notes,
            )

        resolver = dns.resolver.Resolver()
        resolver.timeout = self._timeout_s
        resolver.lifetime = self._lifetime_s

        def _ttl_stats(answers: Any) -> tuple[int, float, float]:
            if not answers:
                return 0, 0.0, 0.0
            ttl_val = float(getattr(answers.rrset, "ttl", getattr(answers, "ttl", 0.0)))
            return 1, ttl_val, 0.0

        # 1. Registro A
        try:
            a_ans = resolver.resolve(domain, "A")
            ips = [str(r) for r in a_ans]
            resolved_ips.extend(ips)
            evidence["a_records"] = ips
            t_cnt, t_med, t_std = _ttl_stats(a_ans)
            features["A Count"] = float(len(ips))
            features["IP Count"] = float(len(ips))
            features["A Missing"] = 0.0
            features["ASN Count"] = 1.0 if ips else 0.0
            features["A TTL Count"] = float(t_cnt)
            features["A Medium TTL"] = float(t_med)
            features["A TTL Standard Deviation"] = float(t_std)
        except dns.exception.Timeout:
            partial = True
            notes.append("DNS_A_TIMEOUT")
        except Exception:
            features["A Missing"] = 1.0

        # 2. Registro AAAA
        try:
            aaaa_ans = resolver.resolve(domain, "AAAA")
            ipv6s = [str(r) for r in aaaa_ans]
            evidence["aaaa_records"] = ipv6s
            _, t_med, _ = _ttl_stats(aaaa_ans)
            features["AAAA Count"] = float(len(ipv6s))
            features["AAAA Medium TTL"] = float(t_med)
        except dns.exception.Timeout:
            partial = True
            notes.append("DNS_AAAA_TIMEOUT")
        except Exception:
            pass

        # 3. Registro MX
        try:
            mx_ans = resolver.resolve(domain, "MX")
            mxs = [str(getattr(r, "exchange", r)).rstrip(".") for r in mx_ans]
            evidence["mx_records"] = mxs
            _, t_med, _ = _ttl_stats(mx_ans)
            features["MX Count"] = float(len(mxs))
            features["MX Missing"] = 0.0
            features["MX Medium TTL"] = float(t_med)
        except dns.exception.Timeout:
            partial = True
            notes.append("DNS_MX_TIMEOUT")
        except Exception:
            features["MX Missing"] = 1.0

        # 4. Registro NS
        try:
            ns_ans = resolver.resolve(domain, "NS")
            nss = [str(r).rstrip(".") for r in ns_ans]
            evidence["ns_records"] = nss
            features["NS Count"] = float(len(nss))
        except dns.exception.Timeout:
            partial = True
            notes.append("DNS_NS_TIMEOUT")
        except Exception:
            pass

        # 5. Registro CNAME
        try:
            cname_ans = resolver.resolve(domain, "CNAME")
            cnames = [str(r).rstrip(".") for r in cname_ans]
            evidence["cname_records"] = cnames
            features["CNAME Count"] = float(len(cnames))
        except Exception:
            pass

        # 6. Registro SOA (caminhamento hierárquico em até 3 níveis — dns-enrich.py linhas 426-447)
        current_domain = domain
        for _ in range(3):
            try:
                soa_ans = resolver.resolve(current_domain, "SOA")
                if soa_ans:
                    soa = soa_ans[0]
                    _, soa_med_ttl, _ = _ttl_stats(soa_ans)
                    expire_int = int(getattr(soa, "expire", 0))
                    serial_int = int(getattr(soa, "serial", 0))
                    refresh_int = int(getattr(soa, "refresh", 0))
                    retry_int = int(getattr(soa, "retry", 0))
                    minimum_int = int(getattr(soa, "minimum", 0))

                    features["SOA Missing"] = 0.0
                    features["SOA Retry"] = float(retry_int)
                    features["SOA Refresh"] = float(refresh_int)
                    features["SOA Minimum"] = float(minimum_int)
                    features["SOA Expire Length"] = float(len(str(abs(expire_int)))) if expire_int else 0.0
                    features["SOA Serial Length"] = float(len(str(abs(serial_int)))) if serial_int else 0.0
                    features["SOA TTL Count"] = 1.0
                    features["SOA Medium TTL"] = float(soa_med_ttl)
                    # Paridade com dataset.csv (SOA TTL Standard Deviation == SOA Medium TTL em consulta única)
                    features["SOA TTL Standard Deviation"] = float(soa_med_ttl)

                    evidence["soa"] = {
                        "mname": str(getattr(soa, "mname", "")).rstrip("."),
                        "serial": serial_int,
                        "refresh": refresh_int,
                        "retry": retry_int,
                        "expire": expire_int,
                        "minimum": minimum_int,
                        "ttl": int(soa_med_ttl),
                    }
                    break
            except (dns.resolver.NoAnswer, dns.resolver.NXDOMAIN):
                parts = current_domain.split(".")
                if len(parts) > 2:
                    current_domain = ".".join(parts[1:])
                else:
                    break
            except dns.exception.Timeout:
                partial = True
                notes.append("DNS_SOA_TIMEOUT")
                break
            except Exception:
                break

        return DnsCollectionResult(
            features=features,
            evidence=evidence,
            resolved_ips=resolved_ips,
            partial=partial,
            notes=notes,
        )

    async def collect(self, domain: str) -> DnsCollectionResult:
        """Executa a coleta DNS em thread pool para não bloquear o event loop."""
        return await asyncio.to_thread(self._sync_collect, domain)
