# Plano de Implementação — Plataforma e API de Detecção de Domínios Maliciosos

Este conjunto de documentos descreve o plano de implementação completo da plataforma web e da API de detecção de domínios maliciosos baseada em **Ensemble Multimodal (Visão + Texto/LLM + Tabular)**, atualizado com todos os detalhes técnicos extraídos dos códigos de treinamento, inferência, ensemble e bases de dados (`*.csv`) do repositório `llm-exploration`.

## Índice

| # | Arquivo | Conteúdo |
|---|---------|----------|
| 01 | `01-arquitetura-e-estrutura.md` | Visão geral, arquitetura em containers, script único `.sh`, stack e estrutura de pastas |
| 02 | `02-modelo-de-dados-supabase.md` | Tabelas, SQL, RLS, índices (todos os ids UUID) e armazenamento de atributos/evidências |
| 03 | `03-autenticacao-e-otp.md` | Cadastro, login, OTP por email, validação de idade (≥ 18 anos) |
| 04 | `04-chaves-de-api.md` | Geração, hash HMAC-SHA256, exibição única, validação e revogação |
| 05 | `05-endpoints-da-api.md` | Contratos dos endpoints (web e API pública `/v1/domains/analyze`) |
| 06 | `06-pipeline-de-deteccao.md` | Especificação técnica completa: 42 atributos, coleta DNS/WHOIS, imagem GADF 42×42, prompt estruturado e Ensemble Multimodal |
| 07 | `07-interface-web.md` | Planejamento completo da Interface Web (UX/UI, telas, componentes, visualização multimodal e fluxos) |
| 08 | `08-seguranca-rate-limit-observabilidade.md` | Segurança, limites de requisição, logs estruturados e métricas |
| 09 | `09-testes-e-deploy.md` | Estratégia de testes, containerização completa e inicialização por script único (`start.sh`) |
| 10 | `10-pendencias-e-decisoes.md` | Decisões de arquitetura, especificações consolidadas do repositório e correções de código mapeadas |

## Requisitos Consolidados

1. **Cadastro e Login Web**: autenticação com **email + senha + código OTP enviado por email**.
2. **Acesso Protegido por Chave de API**: acesso aos endpoints de detecção **sempre vinculado a uma chave de API válida** (via header `X-API-Key` no `curl`/SDK ou resolvida automaticamente via sessão JWT na interface web).
3. **Exibição Única e Hash**: chave gerada uma única vez (`dmd_<prefix>_<secret>`), **armazenada apenas como hash (`HMAC-SHA256`)**, com modal e aviso explícito de que só será exibida uma vez.
4. **Banco de Dados e Auth (Supabase / Postgres)**: todos os identificadores primários e estrangeiros são **UUID** (`gen_random_uuid()`).
5. **Perfil do Usuário e Restrição de Idade**: `username`, `first_name`, `last_name`, `full_name`, `birth_date`, `email`, senha; geração e uso de chave de API restritos a **maiores de 18 anos**.
6. **Entrada Simples de Consulta**: a requisição de análise recebe **apenas o nome do domínio** (tanto na web quanto na API).
7. **Pipeline Multimodal Real (extraído do repositório)**:
   - Extração e cálculo determinístico de **42 atributos** (13 lexicais + 23 DNS/TTL/SOA + 6 WHOIS/Risco Geográfico e ASN).
   - Normalização com `MinMaxScaler` + geração de imagem **GADF 42×42 RGB** (`GramianAngularField`) + construção de **prompt estruturado em 7 blocos** (`[name]`, `[entropy]`, `[whois]`, `[risk]`, `[dns]`, `[ttl]`, `[soa]`).
   - Inferência com **Ensemble Multimodal (`Domain_Ensemble` — pasta `esemble/`)** combinando:
     - **Visão (`esemble/CNN.py`)**: `ConvolutionalNeuralNetwork` (`arch="mid_3b_b"`, blocos `[(32,3,True), (64,3,True), (128,3,True)]`, `fc_dims=[512]`, `dropout=0.5`, `use_bn=True`, entrada `3×42×42`, checkpoint `CustomCNN0-GADF-42_30.pth`);
     - **Tabular (`esemble/MLP.py`)**: `MultiLayerPerceptron` (`mlp_shape=[42, 256, 128, 64, 2]`, `ReLU`, `BatchNorm1d`, `dropout=0.0`, entrada escalonada por `MinMaxScaler`, checkpoint `best_mlp.pt`);
     - **Texto / LLM (`esemble/multimodal.py`)**: `AutoModelForSequenceClassification` com base `meta-llama/Llama-3.2-1B-Instruct` + adaptador PEFT (`PeftModel` em `models/Llama/domain-enriched-tokenized-1B/best`), `max_len=160`;
     - **Meta-classificador (`esemble/multimodal.py` e `esemble/test.py`)**: `LogisticRegression(random_state=42, max_iter=1000)` fazendo stacking de `np.hstack((p_vision, p_llm, p_tabular))`.
8. **Execução Unificada via Script Único (`.sh`)**: toda a aplicação (containers de **Banco de Dados/Supabase**, **API FastAPI** e **Frontend React**) deve ser iniciada através de um único script `./start.sh` na raiz do repositório.

## Ponto de Atenção Crítico (Resolvido)

O requisito *"ao logar, a chave de API é recuperada automaticamente"* conflita diretamente com *"chave armazenada apenas em hash e exibida uma única vez"*, pois hashes criptográficos são irreversíveis. A solução arquitetural adotada está detalhada em `04-chaves-de-api.md` (seção "Acesso web") e registrada em `10-pendencias-e-decisoes.md`.
