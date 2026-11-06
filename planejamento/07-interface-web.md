# 07 — Interface Web (Planejamento de UI/UX e Arquitetura Frontend)

Aplicação Single-Page Application (SPA) desenvolvida em **React 18 + Vite + TypeScript + Tailwind CSS**, empacotada em container Nginx e inicializada junto com a API e o banco de dados pelo script único `./start.sh`.

---

## 1. Stack e Bibliotecas do Frontend

| Camada | Tecnologia | Papel na Aplicação |
|---|---|---|
| Core / Bundler | React 18 + Vite 5 + TypeScript 5 | Renderização tipada e build otimizado para container estático Nginx |
| Estilização & UI | Tailwind CSS + Radix UI (Primitives) + Lucide Icons | Sistema de design acessível (WCAG AA), modais, abas, tooltips e ícones |
| Roteamento | React Router v6 | Rotas públicas (`/login`, `/signup`, `/verify-otp`) e rotas protegidas (`/dashboard`, `/analyze`, `/usage`, `/docs`) |
| Estado & Cache | TanStack Query (React Query v5) | Sincronização de estado do servidor (`/v1/me`, histórico, mutações de chave e análise de domínio) |
| Formulários & Validação | React Hook Form + Zod | Validação em tempo real de cadastro, maioridade (18+), OTP e normalização de domínio |
| Autenticação | `@supabase/supabase-js` + Contexto Auth | Gestão do token JWT em memória/sessão e interceptador HTTP (`Authorization: Bearer <jwt>`) |
| Visualização de Dados | Recharts (ou SVG nativo) | Medidor (*gauge*) de probabilidade final e gráfico de barras dos pesos/scores das 3 modalidades do ensemble (`esemble/`) |

---

## 2. Arquitetura de Layouts e Navegação

### 2.1. Layout Público (`AuthLayout`)
- Fundo escuro/neutro focado em cibersegurança (tema *Slate/Zinc* com acentos em *Emerald* para benigno e *Rose/Amber* para malicioso/alerta).
- Card centralizado com logotipo da plataforma, seletor de tema (Claro/Escuro) e rodapé com link para a documentação pública da API.
- Utilizado nas rotas: `/signup`, `/verify-otp`, `/login`, `/forgot-password`.

### 2.2. Layout Autenticado (`DashboardLayout`)
- **Top Header**:
  - Indicador global de status da Chave de API do usuário logado (obtido automaticamente via `GET /v1/me` no bootstrap da sessão):
    - Badge verde `Chave Ativa (dmd_a1B2c3D4...)`
    - Badge amarelo `Sem Chave Ativa` (com botão rápido "Gerar Chave")
    - Badge vermelho `Restrito (< 18 anos)` ou `Chave Revogada`
  - Consumo rápido de cota diária (`ex.: 42 / 1.000 reqs`)
  - Menu do usuário (`username`, email e botão `Sair`).
- **Sidebar de Navegação**:
  1. **Visão Geral & Chave de API** (`/dashboard`)
  2. **Analisador de Domínio (Ensemble)** (`/analyze`)
  3. **Histórico & Uso** (`/usage`)
  4. **Documentação & Exemplos (`curl`)** (`/docs`)

---

## 3. Detalhamento das Telas e Fluxos de Usuário

### 3.1. Cadastro (`/signup`)
- **Campos do formulário** (validados via Zod):
  - `username`: 3 a 30 caracteres (`^[a-zA-Z0-9_]+$`), verificação de disponibilidade com *debounce* (400 ms).
  - `first_name` (Nome) e `last_name` (Sobrenome).
  - `birth_date` (Data de nascimento): seletor de data com cálculo imediato de idade.
    - **Aviso de Maioridade**: exibe *banner* informativo fixo: *"A criação de conta é permitida, mas a geração de chaves de API e a execução de consultas exigem idade mínima de 18 anos."* Caso a data informada indique `< 18 anos`, um alerta visual destaca que a conta ficará em modo restrito (`age_restricted`).
  - `email`: validação de formato.
  - `password` e `confirm_password`: medidor visual de força de senha (mínimo 10 caracteres, letras maiúsculas, minúsculas, números e símbolos).
- **Ação**: `POST /v1/auth/signup` → redireciona para `/verify-otp?flow=signup&email=...`.

### 3.2. Verificação de OTP (`/verify-otp`)
- Componente customizado `OtpInput` com **6 caixas de dígitos individuais**:
  - Avanço automático de foco ao digitar, retrocesso com `Backspace`, suporte a colar (`Ctrl+V` de 6 dígitos) e submissão automática ao preencher o 6º dígito.
- Cronômetro regressivo de **60 segundos** para habilitar o botão *"Reenviar código OTP"*.
- Tratamento claro de estados: código inválido, código expirado ou limite de tentativas excedido (`429 Too Many Requests`).

### 3.3. Login em Duas Etapas (`/login`)
- **Etapa 1 (Credenciais)**: `email` + `password` → `POST /v1/auth/login`.
  - Mensagens de erro genéricas (*"Credenciais inválidas"*) para evitar enumeração de contas.
  - Não emite sessão final ainda; retorna `challenge_id` e dispara OTP por e-mail.
- **Etapa 2 (OTP)**: transição na mesma tela (ou `/verify-otp?flow=login`) solicitando o código de 6 dígitos enviado por e-mail → `POST /v1/auth/login/verify-otp`.
- **Pós-Login Imediato**: armazena o JWT da sessão e executa imediatamente `GET /v1/me` para carregar o perfil e validar o estado da chave de API antes de renderizar `/dashboard`.

### 3.4. Painel Principal e Gestão de Chave de API (`/dashboard`)
Carrega os dados de `GET /v1/me` e apresenta o **Card de Status da Chave de API** conforme a máquina de estados:

1. **Estado `none` (Nenhuma chave gerada, usuário ≥ 18 anos)**:
   - Exibe call-to-action primário: **"Gerar Chave de API"**.
2. **Estado `active` (Chave ativa)**:
   - Exibe: Prefixo identificável (`dmd_a1B2c3D4`), Data de criação (`created_at`), Último uso (`last_used_at`) e status `Ativa`.
   - Botões de ação:
     - **Rotacionar Chave** (`POST /v1/keys/{id}/rotate`): abre modal de confirmação explicando que a chave atual será revogada imediatamente e uma nova será gerada.
     - **Revogar Chave** (`DELETE /v1/keys/{id}`): abre modal de confirmação exigindo digitar `REVOGAR`.
3. **Estado `revoked` (Chave revogada)**:
   - Mostra data da revogação e botão **"Gerar Nova Chave"**.
4. **Estado `age_restricted` (Usuário menor de 18 anos)**:
   - Card bloqueado com ícone de cadeado e explicação: *"A disponibilização de chave de API é restrita a usuários maiores de 18 anos conforme os termos de uso."* Botões de geração/consulta ficam desabilitados.

#### Modal de Exibição Única da Chave (`KeyOnceModal`)
Acionado exclusivamente na resposta de `POST /v1/keys` ou `POST /v1/keys/{id}/rotate`:
- **Bloqueio de fechamento acidental**: não fecha ao clicar no *backdrop* nem com a tecla `Esc`; registra listener `beforeunload` na aba do navegador enquanto aberto.
- **Banner de Alerta Crítico**:
  > *"Guarde esta chave em local seguro agora. Por motivos de segurança, armazenamos apenas o hash criptográfico no banco de dados e **não será possível exibi-la novamente** após fechar esta janela."*
- **Campo com a chave completa** (`dmd_<prefixo_8_chars>_<segredo_32_bytes>`) em fonte monoespaçada + botão **"Copiar Chave"** (com feedback visual `"Copiado!"`).
- **Checkbox obrigatório**: `[ ] Confirmo que copiei e armazenei minha chave de API em local seguro.`
- **Botão "Concluir e Fechar"**: habilitado somente após marcar o checkbox. Ao fechar, limpa a variável da chave da memória do React (`setPlaintextKey(null)`).

---

### 3.5. Console Interativo de Análise de Domínio (`/analyze`)

Tela principal de operação do sistema de detecção de domínios maliciosos baseado no ensemble multimodal (`esemble/`).

#### A. Barra de Entrada e Validação Local
- **Guarda de Chave Ativa**: se `api_key.status !== 'active'`, o campo de busca exibe um *overlay* bloqueante com botão direto para gerar uma chave em `/dashboard`.
- **Input de Domínio Único**:
  - Aceita apenas um nome de domínio por vez (ex.: `exemplo-login-seguro.com.br`).
  - **Validação em tempo real no cliente** (espelhando `clean_domain` + `remove_www_prefix` + regex RFC 1035/IDNA do backend):
    - Rejeita imediatamente se o usuário colar `http://`, `https://`, caminhos (`/login`), portas (`:8080`), credenciais (`user@`), endereços IP (`192.168.1.1`) ou múltiplos domínios separados por espaço/vírgula, exibindo dica clara de correção abaixo do campo.
- **Indicador de Progresso em Etapas** (enquanto aguarda `POST /v1/domains/analyze`):
  1. *Coletando registros DNS (A, AAAA, MX, NS, CNAME, SOA) e WHOIS/GeoIP...*
  2. *Extraindo os 42 atributos, gerando imagem GADF (42×42) e prompt estruturado...*
  3. *Executando inferência no `Domain_Ensemble` (CustomCNN + MLP + Llama-3.2-1B)...*

#### B. Visualização Multimodal do Resultado (`DomainResultView`)
Quando a resposta de `POST /v1/domains/analyze` retorna, a tela é dividida em 4 painéis técnicos:

1. **Painel 1 — Veredito Final e Meta-Classificador (`esemble/multimodal.py`)**:
   - **Badge de Veredito**: `MALICIOUS` (Vermelho), `SUSPICIOUS` (Âmbar) ou `BENIGN` (Verde).
   - **Gauge de Probabilidade Final (`score`)**: probabilidade de `0.00%` a `100.00%` calculada pela `LogisticRegression` do `Domain_Ensemble`.
   - **Metadados da Execução**: domínio normalizado (`domain`), tempo de processamento (`latency_ms`), status de cache (`cached: true/false`), versão do ensemble (`model_version`) e `request_id`.
   - **Aviso de Resultado Parcial** (quando `partial: true`): banner amarelo informando quais fontes externas sofreram *timeout* (ex.: `WHOIS_TIMEOUT` → `has_whois=0`, `lifetime=-1`, `active_time=-1`).

2. **Painel 2 — Decomposição por Modalidade do Ensemble (`esemble/`)**:
   Três cards lado a lado mostrando a probabilidade individual (`P(malicious)`) e o peso/contribuição na `LogisticRegression`:
   - **Visão — `CustomCNN` (`esemble/CNN.py`)**:
     - Score `model_scores.vision_cnn` (barra de progresso 0–100%).
     - Arquitetura: `mid_3b_b` (`3 blocos conv: 32→64→128 + FC 512`), entrada GADF 42×42 RGB.
   - **Tabular — `MultiLayerPerceptron` (`esemble/MLP.py`)**:
     - Score `model_scores.tabular_mlp` (barra de progresso 0–100%).
     - Arquitetura: `[42, 256, 128, 64, 2]` (`Linear + ReLU + BatchNorm1d + MinMaxScaler`).
   - **Texto/LLM — `Llama-3.2-1B-Instruct + LoRA` (`esemble/multimodal.py`)**:
     - Score `model_scores.text_llm` (barra de progresso 0–100%).
     - Arquitetura: `AutoModelForSequenceClassification` + `PeftModel` (`max_len=160`).

3. **Painel 3 — Artefatos Gerados (Imagem GADF 42×42 & Prompt Estruturado)**:
   - **Card Imagem GADF (`42×42 RGB`)**:
     - Renderiza `artifacts.gadf_image_base64` (`<img src="data:image/png;base64,..." />`) com opção de zoom (`image-rendering: pixelated` para inspecionar a matriz 42×42 gerada por `pyts.image.GramianAngularField(method='difference')`).
   - **Card Prompt Estruturado (7 linhas)**:
     - Exibe em bloco de código com *syntax highlight* o texto exato gerado por `build_structured_prompt` (`[name]`, `[entropy]`, `[whois]`, `[risk]`, `[dns]`, `[ttl]`, `[soa]`) enviado ao `Llama-3.2-1B-Instruct`, com botão de cópia.

4. **Painel 4 — Explorador dos 42 Atributos e Evidências Brutas (Abas)**:
   - **Aba 1: Atributos Léxicos (13 atributos)** — tabela com `lex_length`, `lex_digit_count`, `lex_digit_ratio`, `lex_vowel_ratio`, `lex_consonant_ratio`, `lex_special_char_ratio`, `lex_entropy`, `lex_hyphen_count`, `lex_starts_with_digit`, `lex_has_repeated_chars`, `lex_max_consonant_seq`, `lex_max_digit_seq`, `lex_unique_char_ratio`.
   - **Aba 2: Registros DNS & Contagens (9 atributos)** — `A Count`, `AAAA Count`, `MX Count`, `NS Count`, `CNAME Count`, `IP Count`, `ASN Count`, `A Missing`, `MX Missing`.
   - **Aba 3: Estatísticas de TTL (6 atributos)** — `A TTL Count`, `A Medium TTL`, `A TTL Standard Deviation`, `AAAA Medium TTL`, `MX Medium TTL`, `SOA TTL Count`.
   - **Aba 4: Parâmetros SOA (8 atributos)** — `SOA Missing`, `SOA Refresh`, `SOA Retry`, `SOA Minimum`, `SOA Medium TTL`, `SOA TTL Standard Deviation`, `SOA Serial Length`, `SOA Expire Length`.
   - **Aba 5: WHOIS & Risco Geográfico (6 atributos)** — `has_whois`, `lifetime`, `active_time`, `has_country`, `CCR`, `CCA`.
   - **Aba 6: Evidências de Rede (`evidence`)** — IPs resolvidos, ASNs, Nameservers, país e datas WHOIS.

---

### 3.6. Histórico e Métricas de Uso (`/usage`)
- Cards de resumo: Total de consultas no período, Taxa de domínios maliciosos detectados, Latência média (`ms`) e Cota disponível.
- Tabela paginada (`GET /v1/domains/history`) com colunas: Data/Hora, Domínio, Veredito, Score Ensemble, Scores (`CNN` / `MLP` / `LLM`), Status (`200` / Parcial) e botão **"Ver Detalhes"** (abre *drawer* lateral com os 42 atributos, imagem GADF e prompt daquela consulta).

### 3.7. Documentação Integrada para Desenvolvedores (`/docs`)
- Guia rápido de uso programático da API com o cabeçalho `X-API-Key`.
- Gerador de snippets com abas para **cURL**, **Python (`requests`)** e **JavaScript (`fetch`)**, pré-preenchendo um comentário com o prefixo da chave ativa do usuário (`dom_live_xxxx...`) e o domínio de exemplo:
  ```bash
  curl -X POST http://localhost:8000/v1/domains/analyze \
    -H "Content-Type: application/json" \
    -H "X-API-Key: SUA_CHAVE_AQUI" \
    -d '{"domain": "example.com"}'
  ```
- Link direto para a especificação interativa OpenAPI/Swagger (`/docs` e `/openapi.json` do FastAPI).

---

## 4. Estrutura de Diretórios do Frontend (`web/`)

```text
web/
├── Dockerfile                        # Multi-stage: Node 20 (build Vite) -> Nginx Alpine (runtime)
├── nginx.conf                        # SPA fallback (try_files $uri /index.html) + proxy reverso /v1 -> api:8000
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js
└── src/
    ├── main.tsx
    ├── App.tsx                       # Configuração de rotas e QueryClientProvider
    ├── api/
    │   ├── client.ts                 # Instância fetch/axios com Bearer JWT e X-Request-ID
    │   ├── auth.api.ts               # signup, verifyOtp, login, forgotPassword
    │   ├── keys.api.ts               # getMe, createKey, rotateKey, revokeKey
    │   └── domains.api.ts            # analyzeDomain, getAnalysisById, getHistory
    ├── auth/
    │   ├── AuthContext.tsx           # Sessão Supabase + cache do estado GET /v1/me
    │   └── ProtectedRoute.tsx        # Redireciona não autenticados para /login
    ├── components/
    │   ├── layout/
    │   │   ├── AuthLayout.tsx
    │   │   ├── DashboardLayout.tsx
    │   │   └── ApiKeyStatusBadge.tsx
    │   ├── auth/
    │   │   ├── OtpInput.tsx          # 6 dígitos com suporte a paste e foco automático
    │   │   └── PasswordStrength.tsx
    │   ├── keys/
    │   │   ├── KeyStatusCard.tsx
    │   │   └── KeyOnceModal.tsx      # Modal de exibição única da chave em texto puro
    │   └── analyze/
    │       ├── DomainSearchForm.tsx  # Input com validação RFC/IDNA espelhada do backend
    │       ├── EnsembleScoreCard.tsx # Gauge LogisticRegression + barras CNN, MLP e Llama
    │       ├── GadfImageViewer.tsx   # Preview da imagem GADF 42x42 RGB com zoom
    │       ├── PromptViewer.tsx      # Exibição das 7 seções do prompt estruturado
    │       └── FeaturesTable42.tsx   # Tabela categorizada dos 42 atributos
    ├── pages/
    │   ├── SignupPage.tsx
    │   ├── VerifyOtpPage.tsx
    │   ├── LoginPage.tsx
    │   ├── ForgotPasswordPage.tsx
    │   ├── DashboardPage.tsx
    │   ├── AnalyzePage.tsx
    │   ├── UsagePage.tsx
    │   └── DocsPage.tsx
    └── lib/
        ├── domainValidator.ts        # Regras equivalentes a clean_domain + remove_www_prefix
        └── ageValidator.ts           # Cálculo de maioridade (>= 18 anos)
```

---

## 5. Requisitos de Segurança e Acessibilidade no Frontend

1. **Isolamento da Chave de API em Texto Puro**:
   - A chave retornada na criação/rotação nunca é salva em `localStorage`, `sessionStorage`, `IndexedDB`, cookies ou cache global do React Query; vive apenas no `useState` local do componente `KeyOnceModal` e é sobrescrita com `null` ao desmontar.
2. **Validação Automática no Login**:
   - Sempre que a sessão é iniciada ou restaurada, `AuthContext` chama `GET /v1/me` para sincronizar `is_adult` e `api_key.status`.
3. **Proteção contra XSS e CSP**:
   - Nenhuma saída de DNS/WHOIS ou prompt utiliza `dangerouslySetInnerHTML`.
   - Cabeçalho `Content-Security-Policy` configurado no `web/nginx.conf` permitindo imagens `data:image/png;base64` apenas para a renderização do GADF 42×42.
4. **Acessibilidade (WCAG 2.1 AA)**:
   - Navegação completa por teclado nos 6 campos de OTP, no modal de chave única (`focus trap`) e nas abas dos 42 atributos (`aria-selected`, `role="tablist"`).
   - Contraste mínimo de 4.5:1 e indicadores de veredito que combinam cor + ícone + rótulo textual (`BENIGN`, `SUSPICIOUS`, `MALICIOUS`).
