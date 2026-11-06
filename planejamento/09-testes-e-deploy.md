# 09 — Inicialização Única (`start.sh`), Containers, Testes e Deploy

Toda a aplicação (**Banco de Dados/Supabase**, **API de Inferência `Domain_Ensemble`** e **Interface Web**) é orquestrada via Docker Compose e iniciada através de **um único script shell (`./start.sh`)** na raiz do projeto.

---

## 1. Arquitetura de Containers (`docker-compose.yml`)

A stack completa é composta pelos seguintes serviços interconectados na rede interna `domain-net`:

| Container / Serviço | Imagem / Dockerfile | Porta Host | Responsabilidade |
|---|---|---|---|
| `db` (PostgreSQL 15 / Supabase) | `supabase/postgres:15.1.1.78` | `54322:5432` | Banco de dados relacional com RLS, funções `public.is_adult`, tabelas `profiles`, `api_keys`, `domain_analyses`, `api_requests`, `otp_audit` |
| `auth` (Supabase GoTrue) | `supabase/gotrue:v2.151.0` | `9999:9999` | Emissão de JWT, gestão de usuários e envio de códigos OTP de 6 dígitos |
| `mailpit` (SMTP / Webmail Dev) | `axllent/mailpit:latest` | `8025:8025` (UI) / `1025` (SMTP) | Servidor SMTP local para captura e visualização imediata dos e-mails de OTP durante desenvolvimento e testes |
| `redis` | `redis:7-alpine` | `6379:6379` | Rate limiting por chave/IP, controle de tentativas de OTP e cache quente de análises |
| `api` (FastAPI + PyTorch GPU) | `api/Dockerfile` (baseado no `Dockerfile` da raiz com `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`) | `8000:8000` | Coleta DNS/WHOIS, extração dos 42 atributos, GADF 42×42, prompt de 7 seções e inferência no `Domain_Ensemble` (`esemble/`) |
| `web` (React + Vite + Nginx) | `web/Dockerfile` (`node:20-alpine` → `nginx:stable-alpine`) | `3000:80` | Interface web SPA + proxy reverso `/v1/*` para `http://api:8000` |

### Volumes Persistentes e Montagem de Modelos
- `db-data:/var/lib/postgresql/data`: persistência dos dados do banco.
- `./models:/app/models:ro`: monta em modo somente-leitura os pesos treinados utilizados pelo `esemble/`:
  - `CustomCNN0-GADF-42_30.pth` (`esemble/CNN.py`)
  - `best_mlp.pt` (`esemble/MLP.py`)
  - `models/Llama/domain-enriched-tokenized-1B/best` (`esemble/multimodal.py`)
  - `meta_logreg.joblib` e `minmax_scaler_42.joblib`
- `./data/country_risk_rates.json:/app/data/country_risk_rates.json:ro`: tabela de risco por país (`CCR`) e ASN (`CCA`).

---

## 2. Script Único de Inicialização (`./start.sh`)

O arquivo `./start.sh` na raiz do repositório automatiza todas as etapas de verificação, preparação de ambiente, subida dos containers, migrações de banco e verificação de saúde (*healthcheck*):

### Fluxo de Execução do `./start.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# 1. Verificação de pré-requisitos
#    - Valida se docker e docker compose estão instalados.
#    - Detecta suporte a GPU NVIDIA (nvidia-smi) para ativar --gpus all no serviço api
#      (com fallback automático para CPU caso não haja GPU disponível no host).

# 2. Preparação de variáveis de ambiente (.env)
#    - Se .env não existir, copia de .env.example e gera segredos aleatórios seguros
#      (API_KEY_HMAC_PEPPER, JWT_SECRET, POSTGRES_PASSWORD) via openssl rand -hex 32.

# 3. Validação dos artefatos do Ensemble (pasta esemble/ e models/)
#    - Verifica a presença dos checkpoints das 3 modalidades (CustomCNN, MLP, Llama-3.2-1B LoRA)
#      e do scaler/meta-classificador antes de iniciar a API.

# 4. Subida da camada de Banco de Dados, Auth e Redis
#    - Executa: docker compose up -d db redis mailpit
#    - Aguarda healthcheck do PostgreSQL (pg_isready).

# 5. Aplicação automática das Migrations SQL
#    - Executa em ordem os scripts de supabase/migrations/*.sql
#      (profiles, is_adult(), api_keys, domain_analyses, api_requests, otp_audit, políticas RLS).
#    - Inicia o serviço Supabase GoTrue (auth).

# 6. Build e subida da API (FastAPI + Domain_Ensemble) e do Frontend (React/Nginx)
#    - Executa: docker compose up -d --build api web
#    - Aguarda os endpoints GET http://localhost:8000/healthz e http://localhost:3000 ficarem prontos.

# 7. Resumo no terminal
#    - Imprime as URLs de acesso:
#      * Interface Web:       http://localhost:3000
#      * API & OpenAPI Docs:  http://localhost:8000/docs
#      * Caixa de E-mail OTP: http://localhost:8025
```

### Modos de Uso do `./start.sh`
- `./start.sh`: sobe toda a stack (Banco de Dados + Auth/SMTP + API + Frontend) e exibe o status dos containers.
- `./start.sh --logs`: sobe toda a stack e acompanha os logs integrados (`docker compose logs -f`).
- `./start.sh --down`: encerra e remove os containers com segurança (`docker compose down`).
- `./start.sh --reset-db`: recria os volumes do banco e reaplica todas as migrations do zero.

---

## 3. Estratégia de Testes

| Nível | Escopo e Casos de Teste | Ferramentas |
|---|---|---|
| **Unitário — Domínio & Léxico** | Normalização (`clean_domain` + `remove_www_prefix` de `data/enriching.py`), rejeição de URLs/IPs/caminhos e cálculo exato dos **13 atributos léxicos** (`lex_length` até `lex_unique_char_ratio`) comparados contra linhas reais de `data/dataset.csv`. | `pytest` |
| **Unitário — DNS, TTL, SOA & WHOIS** | Cálculo dos **23 atributos DNS/TTL/SOA** (`DOMAIN_FEATURE_ORDER` de `data/less-is-more/dns-enrich.py`), incluindo `SOA Expire Length` e `SOA Serial Length`, e dos **6 atributos WHOIS/Risco** (`has_whois`, `lifetime`, `active_time`, `has_country`, `CCR`, `CCA`), testando cenários com registros presentes e ausentes (`-1`). | `pytest` + mocks `dnspython` |
| **Unitário — GADF 42×42 & Prompt** | Validação da transformação `MinMaxScaler` + `GramianAngularField(method='difference')` garantindo tensor de saída `(1, 3, 42, 42)`, e validação da função `build_structured_prompt` (`data/test-prompt.py`) garantindo as 7 linhas (`[DOMAIN]` a `[RISK]`). | `pytest` |
| **Integração — `Domain_Ensemble` (`esemble/`)** | Carregamento de `ConvolutionalNeuralNetwork` (`esemble/CNN.py`), `MultiLayerPerceptron` (`esemble/MLP.py`), `Llama-3.2-1B-Instruct + LoRA` (`esemble/multimodal.py`) e `LogisticRegression` meta-model; verificação de paridade de probabilidades `(p_vision, p_llm, p_tabular)` em amostra fixa de domínios. | `pytest` + `torch` |
| **Integração — Auth, OTP & Chaves** | Fluxo completo: `signup` → captura de OTP no Mailpit → `verify-otp` → `login` (2 etapas) → bloqueio de chave para menor de 18 anos (`public.is_adult`) → criação de chave (`HMAC-SHA256`) → consulta via `X-API-Key` (`curl`) e via JWT (`web`). | `pytest` + `httpx` |
| **E2E (Ponta a Ponta)** | Cadastro na UI, preenchimento dos 6 dígitos do OTP, abertura do `KeyOnceModal` (verificando exigência do checkbox de cópia única), consulta de domínio na tela `/analyze` e renderização dos 42 atributos, imagem GADF e prompt. | `Playwright` |

---

## 4. Fases de Implementação

1. **Fase 0 — Infraestrutura e `./start.sh`**:
   - Criação do `docker-compose.yml`, `start.sh`, container PostgreSQL/Supabase Auth/Mailpit/Redis e aplicação das migrations (`02-modelo-de-dados-supabase.md`).
2. **Fase 1 — Autenticação, OTP e Perfil (18+)**:
   - Implementação dos endpoints `/v1/auth/*` e `/v1/me` na API e das telas `/signup`, `/verify-otp`, `/login` no frontend.
3. **Fase 2 — Gestão de Chaves de API (`HMAC-SHA256`)**:
   - Implementação de `/v1/api-keys/*`, middleware dual (`X-API-Key` para `curl` e `Bearer JWT` + verificação de chave ativa para `web`) e componente `KeyOnceModal`.
4. **Fase 3 — Pipeline de Enriquecimento e Extração dos 42 Atributos**:
   - Portabilidade de `data/less-is-more/dns-enrich.py`, `data/enriching.py` e `data/test-prompt.py` para o serviço `api/app/pipeline/`, gerando o vetor canônico de 42 atributos, a imagem GADF 42×42 e o prompt estruturado de 7 seções.
5. **Fase 4 — Integração do `Domain_Ensemble` (`esemble/`) e UI de Análise**:
   - Empacotamento de `esemble/CNN.py`, `esemble/MLP.py` e `esemble/multimodal.py` no worker de inferência da API e construção da tela `/analyze` com visualização dos scores, GADF 42×42, prompt e tabela dos 42 atributos.
6. **Fase 5 — Testes E2E, Cache, Rate Limiting e Observabilidade**:
   - Bateria de testes automatizados, ajuste de timeouts DNS/WHOIS, métricas Prometheus e validação completa subindo do zero via `./start.sh`.

---

## 5. Critérios de Aceite

- Executar `./start.sh` em uma máquina com Docker sobe o banco de dados, aplica as migrations, inicia a API com o ensemble carregado e serve o frontend sem passos manuais adicionais.
- Cadastro e login exigem validação por código OTP de 6 dígitos enviado por e-mail.
- Usuários com idade `< 18 anos` (calculada a partir de `birth_date`) são impedidos de gerar chaves de API tanto na UI quanto na API (`403 AGE_RESTRICTION`).
- A chave de API completa é exibida **uma única vez** na geração/rotação; o banco armazena exclusivamente `key_hash` (`HMAC-SHA256`) e `key_prefix`.
- Qualquer chamada a `POST /v1/domains/analyze` via `curl` sem `X-API-Key` válida retorna `401/403`; na interface web, o estado da chave é validado automaticamente via `GET /v1/me` ao logar.
- O endpoint aceita apenas um nome de domínio válido por requisição, extrai os **42 atributos** na ordem canônica, gera a imagem **GADF 42×42** e o **prompt de 7 seções**, e retorna o veredito do **`Domain_Ensemble` (`esemble/`)**.
