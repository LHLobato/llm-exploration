# 02 — Modelo de Dados (Supabase / PostgreSQL)

Regra global: **toda chave primária (PK) e chave estrangeira (FK) é obrigatoriamente `uuid`**. Senha e códigos OTP ficam sob responsabilidade do Supabase Auth (`auth.users`) no container de autenticação, nunca em tabelas de aplicação em texto puro.

## Extensões Obrigatórias

```sql
create extension if not exists "pgcrypto";
create extension if not exists "citext";
```

## Tabelas

### 1. `public.profiles` (1:1 com `auth.users`)

```sql
create table public.profiles (
  id            uuid primary key references auth.users(id) on delete cascade,
  username      citext not null unique,
  first_name    text   not null,
  last_name     text   not null,
  full_name     text   generated always as (first_name || ' ' || last_name) stored,
  birth_date    date   not null,
  email         citext not null unique,
  created_at    timestamptz not null default now(),
  updated_at    timestamptz not null default now(),
  constraint username_format check (username ~ '^[a-z0-9_]{3,30}$'),
  constraint valid_birth_date check (birth_date >= date '1900-01-01' and birth_date <= current_date)
);

-- Função auxiliar para checagem de maioridade (≥ 18 anos)
-- Nota: current_date não é immutable no Postgres, portanto não pode ser coluna GENERATED STORED;
-- usamos função SQL + view para leitura e validação no backend/banco.
create or replace function public.is_adult(p_birth_date date)
returns boolean
language sql
stable
as $$
  select p_birth_date <= (current_date - interval '18 years')::date;
$$;
```

### 2. `public.api_keys`

```sql
create table public.api_keys (
  id           uuid primary key default gen_random_uuid(),
  user_id      uuid not null references public.profiles(id) on delete cascade,
  name         text not null default 'default',
  key_prefix   text not null,            -- ex.: "dmd_a1B2c3D4" (público p/ UI e logs)
  key_hash     bytea not null unique,    -- HMAC-SHA256(API_KEY_PEPPER, chave_completa)
  status       text not null default 'active'
               check (status in ('active','revoked','expired')),
  last_used_at timestamptz,
  expires_at   timestamptz,
  created_at   timestamptz not null default now(),
  revoked_at   timestamptz
);

create unique index one_active_key_per_user
  on public.api_keys(user_id) where status = 'active';
create index idx_api_keys_prefix on public.api_keys(key_prefix);
```

### 3. `public.domain_analyses` (Resultado da Inferência Multimodal + Cache)

Armazena o resultado completo do `Domain_Ensemble`, os **42 atributos extraídos** (`features`), os scores individuais de cada submodelo (`model_scores`) e as evidências brutas/prompt (`evidence`):

```sql
create table public.domain_analyses (
  id                uuid primary key default gen_random_uuid(),
  domain            text not null,                 -- domínio limpo (sem http/www, punycode, minúsculo)
  registered_domain text not null,                 -- eTLD+1 via tldextract
  label             text not null check (label in ('malicious','benign')),
  score             real not null,                 -- probabilidade final do meta-modelo LogisticRegression [0.0..1.0]
  threshold         real not null default 0.5,
  model_version     text not null,                 -- ex.: "ensemble-v1-cnn42-mlp42-llama1b"
  model_scores      jsonb not null,                -- {"vision_cnn": 0.91, "tabular_mlp": 0.88, "text_llm": 0.94}
  features          jsonb not null,                -- Dicionário com os 42 atributos (lex_*, DNS, TTL, SOA, WHOIS, CCR, CCA)
  structured_prompt text not null,                 -- Prompt de 7 seções ([name]..[soa]) enviado ao LLM
  evidence          jsonb not null,                -- Resumo DNS (IPs, NS, MX, SOA), WHOIS, país/ASN e GADF base64
  partial           boolean not null default false,-- true se WHOIS ou parte do DNS sofreu timeout
  latency_ms        int,                           -- tempo total do pipeline em ms
  created_at        timestamptz not null default now(),
  expires_at        timestamptz not null
);

create index idx_domain_analyses_lookup
  on public.domain_analyses(domain, model_version, expires_at desc);
```

**Estrutura esperada do JSONB `features` (42 chaves na ordem canônica do repositório):**
- **Lexicais (13)**: `lex_length`, `lex_digit_count`, `lex_digit_ratio`, `lex_vowel_ratio`, `lex_consonant_ratio`, `lex_special_char_ratio`, `lex_entropy`, `lex_hyphen_count`, `lex_starts_with_digit`, `lex_has_repeated_chars`, `lex_max_consonant_seq`, `lex_max_digit_seq`, `lex_unique_char_ratio`.
- **DNS, TTL e SOA (23)**: `A Count`, `IP Count`, `MX Count`, `NS Count`, `A Missing`, `ASN Count`, `SOA Retry`, `AAAA Count`, `MX Missing`, `A TTL Count`, `CNAME Count`, `SOA Minimum`, `SOA Missing`, `SOA Refresh`, `A Medium TTL`, `MX Medium TTL`, `SOA TTL Count`, `SOA Medium TTL`, `AAAA Medium TTL`, `SOA Expire Length`, `SOA Serial Length`, `A TTL Standard Deviation`, `SOA TTL Standard Deviation`.
- **WHOIS e Risco (6)**: `has_whois`, `lifetime`, `active_time`, `has_country`, `CCR`, `CCA`.

### 4. `public.api_requests` (Auditoria, Histórico do Usuário e Rate Limit)

```sql
create table public.api_requests (
  id          uuid primary key default gen_random_uuid(),
  api_key_id  uuid not null references public.api_keys(id) on delete cascade,
  user_id     uuid not null references public.profiles(id) on delete cascade,
  analysis_id uuid references public.domain_analyses(id) on delete set null,
  channel     text not null check (channel in ('web','api')),
  domain      text not null,
  verdict     text check (verdict in ('malicious','benign')),
  score       real,
  cached      boolean not null default false,
  status_code int  not null,
  latency_ms  int,
  created_at  timestamptz not null default now()
);

create index idx_api_requests_key_time on public.api_requests(api_key_id, created_at desc);
create index idx_api_requests_user_time on public.api_requests(user_id, created_at desc);
```

### 5. `public.otp_audit` (Auditoria de Segurança de OTP)

```sql
create table public.otp_audit (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid references auth.users(id) on delete set null,
  email_hash  bytea not null,
  event_type  text not null check (event_type in ('signup_sent','login_sent','reset_sent','verified','failed')),
  ip_address  inet,
  created_at  timestamptz not null default now()
);
create index idx_otp_audit_email_time on public.otp_audit(email_hash, created_at desc);
```

## Row Level Security (RLS)

- `profiles`: `select` e `update` (exceto `birth_date` e `email`) apenas onde `id = auth.uid()`.
- `api_keys`: acesso de leitura pelo próprio usuário (`user_id = auth.uid()`) apenas através da view segura `public.user_api_keys_safe` (que omite a coluna `key_hash`); escrita exclusiva pelo backend (`service_role`).
- `api_requests`: `select` permitido onde `user_id = auth.uid()`; `insert` exclusivo pelo backend.
- `domain_analyses` e `otp_audit`: acesso restrito exclusivamente ao backend (`service_role`).

## Inicialização Automática no Container

Ao executar `./start.sh`, o container do banco de dados executa automaticamente os scripts em `supabase/migrations/` na ordem numérica, garantindo que todas as tabelas, funções (`public.is_adult`), índices e políticas RLS estejam prontos antes de a API aceitar conexões.
