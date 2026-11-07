-- =============================================================================
-- 001_init_schema.sql — Modelo de Dados (Supabase / PostgreSQL)
-- Especificação: planejamento/02-modelo-de-dados-supabase.md
-- =============================================================================

create extension if not exists "pgcrypto";
create extension if not exists "citext";

-- Garante existência do schema auth quando executado em PostgreSQL standalone
create schema if not exists auth;

do $$
begin
  if not exists (
    select 1 from information_schema.tables
    where table_schema = 'auth' and table_name = 'users'
  ) then
    create table auth.users (
      id                 uuid primary key default gen_random_uuid(),
      email              citext unique not null,
      encrypted_password text,
      email_confirmed_at timestamptz,
      raw_user_meta_data jsonb default '{}'::jsonb,
      created_at         timestamptz not null default now(),
      updated_at         timestamptz not null default now()
    );
  end if;
end $$;

-- Função auxiliar de maioridade (>= 18 anos completos)
create or replace function public.is_adult(p_birth_date date)
returns boolean
language sql
stable
as $$
  select p_birth_date <= (current_date - interval '18 years')::date;
$$;

-- 1. public.profiles (1:1 com auth.users)
create table if not exists public.profiles (
  id            uuid primary key references auth.users(id) on delete cascade,
  username      citext not null unique,
  first_name    text   not null,
  last_name     text   not null,
  full_name     text   generated always as (first_name || ' ' || last_name) stored,
  birth_date    date   not null,
  email         citext not null unique,
  created_at    timestamptz not null default now(),
  updated_at    timestamptz not null default now(),
  constraint username_format check (username ~ '^[a-zA-Z0-9_]{3,30}$'),
  constraint valid_birth_date check (birth_date >= date '1900-01-01' and birth_date <= current_date)
);

-- 2. public.api_keys
create table if not exists public.api_keys (
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

create unique index if not exists one_active_key_per_user
  on public.api_keys(user_id) where status = 'active';
create index if not exists idx_api_keys_prefix
  on public.api_keys(key_prefix);

-- View segura sem key_hash para leitura autenticada
create or replace view public.user_api_keys_safe as
select
  id,
  user_id,
  name,
  key_prefix,
  status,
  last_used_at,
  expires_at,
  created_at,
  revoked_at
from public.api_keys;

-- 3. public.domain_analyses (Resultado da Inferência Multimodal + Cache)
create table if not exists public.domain_analyses (
  id                uuid primary key default gen_random_uuid(),
  domain            text not null,
  registered_domain text not null,
  label             text not null check (label in ('malicious','benign')),
  score             real not null,
  threshold         real not null default 0.5,
  model_version     text not null,
  model_scores      jsonb not null,
  features          jsonb not null,
  structured_prompt text not null,
  evidence          jsonb not null,
  partial           boolean not null default false,
  latency_ms        int,
  created_at        timestamptz not null default now(),
  expires_at        timestamptz not null
);

create index if not exists idx_domain_analyses_lookup
  on public.domain_analyses(domain, model_version, expires_at desc);

-- 4. public.api_requests (Auditoria, Histórico do Usuário e Rate Limit)
create table if not exists public.api_requests (
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

create index if not exists idx_api_requests_key_time
  on public.api_requests(api_key_id, created_at desc);
create index if not exists idx_api_requests_user_time
  on public.api_requests(user_id, created_at desc);

-- 5. public.otp_audit (Auditoria de Segurança de OTP)
create table if not exists public.otp_audit (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid references auth.users(id) on delete set null,
  email_hash  bytea not null,
  event_type  text not null check (event_type in ('signup_sent','login_sent','reset_sent','verified','failed')),
  ip_address  inet,
  created_at  timestamptz not null default now()
);

create index if not exists idx_otp_audit_email_time
  on public.otp_audit(email_hash, created_at desc);

-- Row Level Security (RLS)
alter table public.profiles enable row level security;
alter table public.api_keys enable row level security;
alter table public.domain_analyses enable row level security;
alter table public.api_requests enable row level security;
alter table public.otp_audit enable row level security;
