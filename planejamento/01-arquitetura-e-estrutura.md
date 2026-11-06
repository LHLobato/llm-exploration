# 01 — Arquitetura, Stack e Estrutura de Pastas

## Visão Geral dos Componentes (Arquitetura Containerizada)

Toda a aplicação roda em containers orquestrados por `docker-compose.yml` e iniciados por um **único script `./start.sh`**:

```
                           ┌──────────────────────────┐
                           │     ./start.sh (CLI)     │
                           └────────────┬─────────────┘
                                        │ docker compose up --build
        ┌───────────────────────────────┼───────────────────────────────┐
        ▼                               ▼                               ▼
┌──────────────────────┐     ┌─────────────────────────┐     ┌─────────────────────────┐
│ Container: web       │     │ Container: api          │     │ Stack: db / supabase    │
│ (React + Vite/Nginx) │     │ (FastAPI + PyTorch GPU) │     │ (Postgres 15 + GoTrue)  │
│ Porta: 3000 / 80     │────►│ Porta: 8000             │────►│ Porta: 5432 / 54321     │
└──────────────────────┘     └────────────┬────────────┘     └─────────────────────────┘
                                          │
                 ┌────────────────────────┼────────────────────────┐
                 ▼                        ▼                        ▼
      ┌────────────────────┐   ┌────────────────────┐   ┌─────────────────────────┐
      │ Coleta Assíncrona  │   │ Engenharia de      │   │ Motor de Inferência     │
      │ - DNS (dnspython)  │   │ Atributos (42d)    │   │ Domain_Ensemble         │
      │ - WHOIS / RDAP     │   │ - 13 Lexicais      │   │ - CNN (GADF 3x42x42)    │
      │ - GeoIP / ASN Risk │   │ - 23 DNS/TTL/SOA   │   │ - MLP ([42,256,128,64,2]│
      └────────────────────┘   │ - 6 WHOIS/CCR/CCA  │   │ - LLM (Llama-3.2-1B 4b) │
                               │ - GADF 42x42 RGB   │   │ - Meta: LogReg Stacking │
                               │ - Prompt 7 seções  │   └─────────────────────────┘
                               └────────────────────┘
```

- **Container `web` (SPA React)**: interface completa para cadastro, login em 2 etapas com OTP, gestão da chave de API (com modal de visualização única), console interativo de análise de domínio (com explicabilidade visual das 3 modalidades do ensemble e dos 42 atributos), histórico de uso e documentação integrada.
- **Container `api` (FastAPI + Motor de IA)**: valida credenciais (JWT Supabase ou header `X-API-Key`), aplica rate limit, executa a coleta ao vivo (DNS + WHOIS/RDAP + GeoIP/ASN), constrói os 42 atributos, gera em memória a imagem GADF 42×42 e o prompt estruturado, e executa o `Domain_Ensemble` na GPU (com fallback configurável para CPU).
- **Containers `db` / `supabase` (Banco de Dados + Auth)**: instância PostgreSQL 15+ (com extensões `citext` e `pgcrypto`) e serviço Supabase Auth (GoTrue) + Inbucket/Mailpit (para captura local de e-mails OTP em desenvolvimento) ou integração com Supabase Cloud/Self-Hosted, inicializados automaticamente com as migrations SQL.

## Stack Tecnológica Consolidada (Baseada no Repositório)

| Camada | Tecnologia / Biblioteca | Referência no Repositório / Observação |
|--------|-------------------------|----------------------------------------|
| **Orquestração** | Docker, Docker Compose, NVIDIA Container Toolkit, Bash (`start.sh`) | Baseado e expandido a partir de `Dockerfile` e `run_docker.sh` (CUDA 12.4.1 + Ubuntu 22.04 + Python 3.11) |
| **API Backend** | Python 3.11, FastAPI, Uvicorn, Pydantic v2 | Async nativo; inferência pesada isolada em `ThreadPoolExecutor` |
| **Banco & Auth** | PostgreSQL 15 + Supabase Auth (`supabase-py`, `asyncpg`) | Todos os IDs são `uuid` |
| **Frontend Web** | React 18+, Vite, TypeScript, Tailwind CSS, Lucide Icons, Recharts | Gráficos para scores do ensemble e visualização da imagem GADF |
| **Coleta DNS** | `dnspython` (`dns.resolver.Resolver` / `dns.asyncresolver`) | `data/less-is-more/dns-enrich.py` (`timeout=1s, lifetime=1s`, registros A, AAAA, MX, NS, CNAME, SOA) |
| **Coleta WHOIS/Geo** | `httpx` (RDAP JSON), `python-whois`, `geoip2` / tabela local CCR/CCA | Extrai `has_whois`, `lifetime`, `active_time`, `has_country`, `CCR`, `CCA` |
| **Atributos & Imagem** | `numpy`, `pandas`, `scikit-learn` (`MinMaxScaler`), `pyts` (`GramianAngularField`), `Pillow` | `esemble/multimodal.py`, `esemble/CNN.py` (GADF 42×42 RGB) |
| **Modelos Deep Learning** | `torch` (cu124), `torchvision`, `transformers`, `peft` (DoRA/LoRA), `bitsandbytes` (4-bit NF4), `accelerate` | `esemble/multimodal.py`, `esemble/CNN.py`, `esemble/MLP.py`, `training/bert_training.py` |
| **Meta-Modelo / ML** | `scikit-learn` (`LogisticRegression`), `xgboost` (`XGBClassifier`), `joblib` | `esemble/multimodal.py`, `esemble/machine_learning.py` |

## Estrutura de Pastas do Projeto

```
llm-exploration/
├── start.sh                        # Script ÚNICO que sobe toda a aplicação (DB + API + Frontend)
├── docker-compose.yml              # Orquestração dos containers (db/supabase, api, web)
├── .env.example                    # Template de variáveis de ambiente
├── api/
│   ├── Dockerfile                  # Imagem da API (base CUDA 12.4.1 + Python 3.11 + deps ML/FastAPI)
│   ├── pyproject.toml / requirements.txt
│   ├── app/
│   │   ├── main.py                 # Entrypoint FastAPI + lifespan (carrega Domain_Ensemble 1x)
│   │   ├── core/                   # config.py, security.py, logging.py, exceptions.py
│   │   ├── db/                     # pool asyncpg, repositórios (profiles, keys, analyses, requests)
│   │   ├── auth/                   # rotas /v1/auth/* (signup, login 2 etapas, otp, password reset)
│   │   ├── keys/                   # geração CSPRNG, HMAC-SHA256, validação, revogação
│   │   ├── deps/                   # get_current_user, require_api_key_or_web_jwt, rate_limiter
│   │   ├── domains/                # rotas /v1/domains/*, schemas Pydantic, clean_domain
│   │   └── pipeline/
│   │       ├── constants.py        # Ordem exata das 42 features, SYSTEM_INSTRUCTION, tabelas CCR/CCA
│   │       ├── collectors/
│   │       │   ├── dns_collector.py    # Coleta A, AAAA, MX, NS, CNAME, SOA (busca 3 níveis)
│   │       │   ├── whois_collector.py  # RDAP + fallback python-whois (lifetime, active_time)
│   │       │   └── geo_asn.py          # Resolução de país/ASN dos IPs e lookup CCR / CCA
│   │       ├── features/
│   │       │   ├── lexical.py          # 13 features lexicais (lex_length .. lex_unique_char_ratio)
│   │       │   └── builder.py          # Monta vetor ordenado de 42 features + MinMaxScaler
│   │       ├── renderers/
│   │       │   ├── gadf.py             # GramianAngularField (42x42 RGB tensor) em memória
│   │       │   └── prompt.py           # build_structured_prompt ([name]..[soa]) + chat template
│   │       └── inference/
│   │           ├── cnn_model.py        # ConvolutionalNeuralNetwork de esemble/CNN.py (mid_3b_b, 3x42x42)
│   │           ├── mlp_model.py        # MultiLayerPerceptron de esemble/MLP.py ([42, 256, 128, 64, 2])
│   │           ├── llm_model.py        # Llama-3.2-1B-Instruct + PeftModel de esemble/multimodal.py
│   │           └── ensemble.py         # Domain_Ensemble + LogisticRegression de esemble/multimodal.py
│   └── tests/
├── web/                            # Aplicação Frontend (React + Vite + TypeScript + Tailwind)
│   ├── Dockerfile                  # Build multi-stage (Node 20 build -> Nginx alpine)
│   ├── nginx.conf                  # Proxy reverso /v1 -> api:8000 e SPA fallback
│   ├── package.json
│   └── src/                        # Código-fonte da interface (detalhado no doc 07)
├── supabase/
│   ├── migrations/                 # 001_init_schema.sql (profiles, api_keys, api_requests, domain_analyses)
│   ├── seed.sql                    # Dados iniciais de desenvolvimento
│   └── templates/                  # Templates de e-mail OTP em PT-BR
├── models/                         # Checkpoints treinados montados via volume no container da API
│   ├── Llama/domain-enriched-tokenized-1B/best/
│   ├── cnn/CustomCNN0-GADF-42_30.pth
│   ├── mlp/best_mlp.pt
│   └── ensemble/                   # tabular_scaler.joblib, meta_logreg.joblib, ccr_cca_lookup.json
└── planejamento/                   # Esta documentação de arquitetura e especificação
```

## Princípios Arquiteturais

1. **Inicialização Comando Único (`./start.sh`)**: prepara variáveis `.env`, verifica disponibilidade de GPU (habilitando profile GPU ou CPU automaticamente), sobe os containers de banco/Auth, aplica as migrations SQL, sobe a API aguardando o carregamento dos modelos (`/readyz`) e inicia o frontend.
2. **Carregamento Único de Modelos (Singleton no Lifespan)**: `CustomCNN`, `MultiLayerPerceptron`, `Llama-3.2-1B-Instruct` (ou `ModernBERT`), `MinMaxScaler` e `LogisticRegression` são carregados na memória/VRAM **uma única vez** na inicialização do FastAPI, com *warm-up* sintético.
3. **Execução em Memória (Zero Disco na Inferência)**: diferente do script de avaliação offline que lia PNGs do disco (`ImagePathDataset`), na API a imagem GADF 42×42 é gerada em RAM como tensor PyTorch `(1, 3, 42, 42)` diretamente a partir do vetor escalonado de 42 atributos.
4. **Resiliência em I/O de Rede**: consultas DNS (`timeout=1.0s`, `lifetime=1.0s` por tipo de registro em paralelo) e WHOIS/RDAP (`timeout=3.0s`) nunca derrubam a requisição; ausências alimentam as flags nativas do modelo (`A Missing=1`, `MX Missing=1`, `SOA Missing=1`, `has_whois=0`, `has_country=0`).
5. **Segurança por Padrão**: todos os IDs são UUID; chaves de API salvas exclusivamente como `HMAC-SHA256`; nenhuma requisição HTTP é feita ao domínio investigado (apenas resolução DNS e consulta WHOIS/RDAP).

## Variáveis de Ambiente

| Variável | Exemplo / Padrão | Descrição |
|----------|------------------|-----------|
| `SUPABASE_URL` | `http://supabase-kong:8000` | URL interna/externa do Supabase |
| `SUPABASE_ANON_KEY` | `eyJ...` | Chave pública anônima (Auth) |
| `SUPABASE_SERVICE_ROLE_KEY` | `eyJ...` | Chave administrativa (restrita ao backend) |
| `SUPABASE_JWT_SECRET` | `super-secret-jwt-token...` | Segredo para validação local do JWT |
| `DATABASE_URL` | `postgresql://postgres:postgres@db:5432/postgres` | Conexão direta do `asyncpg` |
| `API_KEY_PEPPER` | `hex-secret-32-bytes` | Pepper secreto para `HMAC-SHA256` das chaves de API |
| `MODELS_ROOT` | `/workspace/models` | Diretório montado com os pesos do ensemble |
| `CNN_CHECKPOINT` | `/workspace/models/cnn/CustomCNN0-GADF-42_30.pth` | Pesos da `CustomCNN` (GADF 42×42) |
| `MLP_CHECKPOINT` | `/workspace/models/mlp/best_mlp.pt` | Pesos do `MultiLayerPerceptron` (`[42, 256, 128, 64, 2]`) |
| `LLM_CHECKPOINT` | `/workspace/models/Llama/domain-enriched-tokenized-1B/best` | Adaptador PEFT DoRA + tokenizer do LLM |
| `LLM_BASE_MODEL` | `meta-llama/Llama-3.2-1B-Instruct` | Modelo base para o adaptador PEFT |
| `TABULAR_SCALER_PATH` | `/workspace/models/ensemble/tabular_scaler.joblib` | `MinMaxScaler` ajustado nos 42 atributos de treino |
| `META_MODEL_PATH` | `/workspace/models/ensemble/meta_logreg.joblib` | `LogisticRegression` do stacking |
| `INFERENCE_DEVICE` | `cuda` (fallback `cpu`) | Dispositivo de execução PyTorch |
| `DECISION_THRESHOLD` | `0.5` | Limiar de probabilidade para classificar como `malicious` |
| `DNS_TIMEOUT_S` | `1.0` | Timeout por consulta DNS (`dnspython`) |
| `WHOIS_TIMEOUT_S` | `3.0` | Timeout para consulta RDAP/WHOIS |
| `HF_TOKEN` | `hf_...` | Token HuggingFace para download inicial do modelo base |
