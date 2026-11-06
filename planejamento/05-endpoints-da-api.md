# 05 — Contratos dos Endpoints da API

Base URL: `/v1`. Todos os payloads em JSON UTF-8. Todos os identificadores são `uuid`.

Formato padrão de erro:

```json
{
  "error": {
    "code": "invalid_domain",
    "message": "Informe apenas o nome do domínio (sem protocolo http/https, caminho ou porta).",
    "request_id": "550e8400-e29b-41d4-a716-446655440000"
  }
}
```

## Matriz de Autenticação

| Grupo de Rotas | Credencial Exigida | Observação |
|----------------|-------------------|------------|
| `/v1/auth/*` | Nenhuma (público com rate limit por IP/email) | Fluxos de cadastro, login 2FA e recuperação de senha |
| `/v1/me`, `/v1/keys/*`, `/v1/usage` | `Authorization: Bearer <supabase_jwt>` | Exclusivo para sessão autenticada na interface web |
| `/v1/domains/*` | **`X-API-Key: dmd_...`** (clientes externos / `curl`) **OU** `Authorization: Bearer <supabase_jwt>` (interface web) | Na web, o backend resolve e valida obrigatoriamente a chave ativa do usuário logado (ver doc 04) |
| `/healthz`, `/readyz` | Nenhuma | Healthcheck dos containers Docker (`start.sh`) |

## 1. Autenticação Web (`/v1/auth`)

Detalhado no doc 03:

- `POST /v1/auth/signup`
  - Body: `{ "username": "string", "first_name": "string", "last_name": "string", "birth_date": "YYYY-MM-DD", "email": "string", "password": "string" }`
  - Response `202`: `{ "otp_required": true, "email": "u***@exemplo.com", "expires_in_seconds": 600 }`
- `POST /v1/auth/signup/verify`
  - Body: `{ "email": "string", "token": "123456" }`
  - Response `200`: `{ "access_token": "...", "refresh_token": "...", "user": { "id": "uuid", "username": "...", "full_name": "...", "is_adult": true } }`
- `POST /v1/auth/login`
  - Body: `{ "email": "string", "password": "string" }`
  - Valida senha sem reter sessão e dispara OTP por email.
  - Response `200`: `{ "otp_required": true, "expires_in_seconds": 600 }`
- `POST /v1/auth/login/verify`
  - Body: `{ "email": "string", "token": "123456" }`
  - Response `200`: `{ "access_token": "...", "refresh_token": "...", "user": { ... } }`
- `POST /v1/auth/otp/resend` → `200 { "sent": true, "cooldown_seconds": 60 }`
- `POST /v1/auth/logout` → `204`
- `POST /v1/auth/password/reset` e `POST /v1/auth/password/reset/verify`

## 2. Perfil, Chaves de API e Uso (`/v1/me`, `/v1/keys`, `/v1/usage`)

| Método | Rota | Descrição |
|--------|------|-----------|
| `GET` | `/v1/me` | Retorna perfil (`id`, `username`, `first_name`, `last_name`, `full_name`, `birth_date`, `email`, `is_adult`) + status da chave (`has_active_key`, `key_id`, `key_prefix`, `key_status`, `can_generate_key`, `last_used_at`) |
| `POST` | `/v1/keys` | Gera chave (exige `is_adult == true`), retorna o segredo completo `dmd_...` **uma única vez** (`201`) |
| `GET` | `/v1/keys` | Lista histórico de chaves do usuário (apenas metadados: `id`, `name`, `key_prefix`, `status`, `created_at`, `last_used_at`, `revoked_at`) |
| `POST` | `/v1/keys/{id}/rotate` | Revoga a chave atual e emite uma nova (retornando a nova uma única vez) |
| `DELETE` | `/v1/keys/{id}` | Revoga imediatamente a chave especificada |
| `GET` | `/v1/usage` | Estatísticas de uso da chave ativa (requisições hoje/minuto, limite restante, latência média e histórico paginado de consultas do usuário) |

## 3. Detecção de Domínio Malicioso (`/v1/domains`)

### `POST /v1/domains/analyze`

Request — recebe **apenas o nome do domínio**:

```json
{
  "domain": "exemplo.com.br"
}
```

Exemplo via `curl`:

```bash
curl -X POST http://localhost:8000/v1/domains/analyze \
  -H "X-API-Key: dmd_a1B2c3D4_9fK..." \
  -H "Content-Type: application/json" \
  -d '{"domain": "exemplo.com.br"}'
```

#### Regras de Validação e Normalização de Entrada (`400 invalid_domain`)
Alinhadas com `clean_domain` (`data/less-is-more/dns-enrich.py` e `data/enriching.py`) e `remove_www_prefix` (`data/filterdata.py` e `data/enriching.py`), porém **rejeitando entradas que não sejam um domínio válido**:
1. Rejeitar strings vazias, URLs contendo esquema (`http://`, `https://`, `//`), caminho (`/`), query string (`?`), fragmento (`#`), credenciais (`@`), porta (`:8080`), espaços, endereços IPv4/IPv6 literais ou wildcards (`*.`).
2. Converter para minúsculas, remover ponto final (`.` de FQDN) e remover prefixo `www.` se presente (`removeprefix("www.")`, mantendo paridade exata com `data/filterdata.py` e `data/enriching.py`).
3. Converter IDN (domínios internacionalizados) para ASCII/Punycode via `idna.encode`.
4. Validar comprimento total (≤ 253 chars), formato de cada label (1–63 chars, `[a-z0-9-]`, sem iniciar/terminar com hífen) e existência de sufixo público válido via `tldextract` (extraindo `registered_domain`).
5. Bloquear domínios reservados/internos (`localhost`, `.local`, `.internal`, `.arpa`, `.test`, `.example`, `.invalid`).

#### Response `200 OK`

Retorna o veredito final do `Domain_Ensemble`, a probabilidade (`score`), as probabilidades individuais das 3 modalidades (`model_scores`), o prompt estruturado gerado, a imagem GADF 42×42 (em base64 PNG para exibição direta na interface web), os 42 atributos agrupados/ordenados e as evidências de coleta:

```json
{
  "id": "b3d79642-8d5c-414b-9a16-247b1e08f412",
  "domain": "fr7ehgd.duckdns.org",
  "registered_domain": "duckdns.org",
  "verdict": "malicious",
  "label_id": 1,
  "score": 0.9642,
  "threshold": 0.5,
  "model_version": "ensemble-v1-cnn42-mlp42-llama1b",
  "model_scores": {
    "vision_cnn": 0.9410,
    "tabular_mlp": 0.9285,
    "text_llm": 0.9712
  },
  "meta_weights": {
    "vision_cnn": 1.4215,
    "tabular_mlp": 1.1840,
    "text_llm": 2.6510,
    "intercept": -2.4102
  },
  "partial": false,
  "cached": false,
  "latency_ms": 284,
  "analyzed_at": "2026-11-06T16:45:00Z",
  "artifacts": {
    "structured_prompt": "[name]: fr7ehgd.duckdns.org\n[entropy]: 3.6819\n[whois]: present=no\n[risk]: ccr=0.863 cca=0.863 country=yes\n[dns]: A=1 AAAA=0 MX=0 NS=2 CNAME=0 IPs=1 ASNs=1 a_missing=no mx_missing=yes\n[ttl]: a_med=60.0 a_std=0.0 a_distinct=1 mx_med=0.0 aaaa_med=0.0\n[soa]: missing=no refresh=43200 retry=3600 min=3601 expire_len=6 serial_len=10 med=3601.0 std=0.0 distinct=1",
    "gadf_image_base64": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACoAAAAq..."
  },
  "features": {
    "lexical": {
      "lex_length": 19,
      "lex_digit_count": 2,
      "lex_digit_ratio": 0.1053,
      "lex_vowel_ratio": 0.1579,
      "lex_consonant_ratio": 0.6316,
      "lex_special_char_ratio": 0.1053,
      "lex_entropy": 3.6819,
      "lex_hyphen_count": 0,
      "lex_starts_with_digit": 0,
      "lex_has_repeated_chars": 0,
      "lex_max_consonant_seq": 4,
      "lex_max_digit_seq": 1,
      "lex_unique_char_ratio": 0.6842
    },
    "dns_counts_and_flags": {
      "A Count": 1,
      "IP Count": 1,
      "MX Count": 0,
      "NS Count": 2,
      "A Missing": 0,
      "ASN Count": 1,
      "AAAA Count": 0,
      "MX Missing": 1,
      "CNAME Count": 0,
      "SOA Missing": 0
    },
    "ttl_stats": {
      "A TTL Count": 1,
      "A Medium TTL": 60.0,
      "A TTL Standard Deviation": 0.0,
      "MX Medium TTL": 0.0,
      "AAAA Medium TTL": 0.0,
      "SOA TTL Count": 1,
      "SOA Medium TTL": 3601.0,
      "SOA TTL Standard Deviation": 0.0
    },
    "soa_params": {
      "SOA Refresh": 43200,
      "SOA Retry": 3600,
      "SOA Minimum": 3601,
      "SOA Expire Length": 6,
      "SOA Serial Length": 10
    },
    "whois_and_risk": {
      "has_whois": 0.0,
      "lifetime": 0.0,
      "active_time": 0.0,
      "has_country": 1.0,
      "CCR": 0.863,
      "CCA": 0.863
    }
  },
  "evidence": {
    "dns": {
      "a_records": ["203.0.113.45"],
      "aaaa_records": [],
      "mx_records": [],
      "ns_records": ["ns1.duckdns.org", "ns2.duckdns.org"],
      "cname_records": [],
      "soa": {
        "mname": "ns1.duckdns.org",
        "serial": 2026110601,
        "refresh": 43200,
        "retry": 3600,
        "expire": 604800,
        "minimum": 3601,
        "ttl": 3601
      }
    },
    "whois": {
      "present": false,
      "creation_date": null,
      "updated_date": null,
      "expiration_date": null,
      "registrar": null
    },
    "geo_asn": {
      "countries": ["US"],
      "asns": ["AS16509"],
      "ccr": 0.863,
      "cca": 0.863
    },
    "notes": []
  }
}
```

### `GET /v1/domains/analyses/{id}`

Recupera uma análise específica por `uuid` (desde que vinculada ao histórico de `api_requests` do usuário/chave autenticada).

## Cabeçalhos de Resposta Padrão

`X-Request-Id`, `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset`, `Retry-After` (em respostas `429`).
