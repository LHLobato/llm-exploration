# 10 — Decisões Técnicas, Ajustes de Código (`esemble/`) e Questões em Aberto

## 1. Decisões Arquiteturais e de Segurança (D1–D7)

| # | Decisão | Motivo |
|---|---|---|
| **D1** | Chave de API com `HMAC-SHA256` + *pepper* secreto (não bcrypt) | A chave gerada (`dmd_<prefixo>_<segredo>`) já possui alta entropia (256 bits); `HMAC-SHA256` permite validação em tempo constante (`O(1)`) em cada requisição sem custo de CPU elevado. |
| **D2** | Web usa JWT da sessão e o backend resolve/valida automaticamente a chave ativa do usuário | Como o hash da chave é irreversível e ela só é exibida uma única vez, o usuário logado na web não precisa redigitar a chave a cada consulta; o backend valida que o usuário possui chave `active` e registra o uso contra essa `api_key_id`. |
| **D3** | Login em duas etapas (`email + senha` → `OTP por e-mail`) sem emitir sessão final na etapa 1 | Garante que o código OTP seja obrigatório tanto na confirmação de cadastro quanto em cada novo login. |
| **D4** | Uma chave ativa por usuário no MVP (com suporte a rotação e revogação) | Simplifica a resolução automática de chave na interface web e o controle de cota por conta. |
| **D5** | Menor de 18 anos pode criar conta, mas recebe `403 AGE_RESTRICTION` ao tentar gerar chave de API | Atende estritamente à regra de negócio de que a disponibilização de chave de API é exclusiva para maiores de 18 anos (`public.is_adult(birth_date)`). |
| **D6** | Nenhuma conexão HTTP/HTTPS ao domínio consultado | A detecção utiliza exclusivamente consultas DNS passivas/ativas aos resolvedores recursivos e WHOIS/RDAP, eliminando riscos de SSRF ou execução de payload malicioso. |
| **D7** | Inicialização completa via script único `./start.sh` com Docker Compose | Garante reprodutibilidade imediata subindo Banco de Dados + Migrations + Auth/SMTP + API GPU (`Domain_Ensemble`) + Frontend Web com um único comando. |

---

## 2. Decisões Técnicas Extraídas do Repositório e da Pasta `esemble/` (D8–D14)

Todas as pendências técnicas anteriores sobre atributos, imagens, prompts e arquitetura do ensemble foram **resolvidas** a partir da inspeção dos códigos e dados do repositório:

| # | Componente | Especificação Técnica Definida | Fonte no Repositório |
|---|---|---|---|
| **D8** | **Conjunto e Ordem Canônica dos 42 Atributos** | Vetor numérico de exatas **42 features**: **13 Léxicas** (`lex_length` a `lex_unique_char_ratio`), **23 DNS/TTL/SOA** (`DOMAIN_FEATURE_ORDER`: `A Count` a `SOA TTL Standard Deviation`) e **6 WHOIS/Risco** (`has_whois`, `lifetime`, `active_time`, `has_country`, `CCR`, `CCA`). | `data/dataset.csv`, `data/less-is-more/dns-enrich.py`, `logs/xgb_seed0.log` |
| **D9** | **Normalização de Domínio e Tratamento de Ausentes** | Normalização via `clean_domain` + `remove_www_prefix`. Quando DNS ou WHOIS não retornam registros, as flags `A Missing`, `MX Missing`, `SOA Missing` valem `1` (ou `has_whois=0`, `has_country=0`) e estatísticas de TTL/SOA/WHOIS/Risco ausentes recebem `-1` (`-1.0`). | `data/enriching.py`, `data/less-is-more/dns-enrich.py`, `data/test-prompt.py` |
| **D10** | **Geração de Imagem de Visão (GADF 42×42 RGB)** | Os 42 atributos numéricos escalonados são convertidos em matriz `42×42` via **Gramian Angular Difference Field** (`pyts.image.GramianAngularField(method='difference')`), salva/renderizada como imagem RGB `42×42` e transformada com `transforms.Resize((42, 42))` + `transforms.ToTensor()`. | `esemble/CNN.py`, `esemble/multimodal.py`, `esemble/script.sh` (`CustomCNN0-GADF-42_30.pth`) |
| **D11** | **Template de Prompt Estruturado (7 Seções)** | Prompt determinístico de 7 linhas (`[DOMAIN]`, `[DNS]`, `[TTL]`, `[SOA]`, `[WHOIS]`, `[LEXICAL]`, `[RISK]`) gerado por `build_structured_prompt` e formatado via `tokenizer.apply_chat_template` com a `SYSTEM_INSTRUCTION` de especialista em cibersegurança (`max_len=160`). | `data/test-prompt.py`, `esemble/multimodal.py` |
| **D12** | **Arquitetura Exclusiva do Ensemble (`esemble/`)** | O ensemble de produção segue estritamente `esemble/multimodal.py` (`Domain_Ensemble`):<br>1. **Visão (`esemble/CNN.py`)**: `ConvolutionalNeuralNetwork` com `arch="mid_3b_b"` (`[(32,3,True), (64,3,True), (128,3,True)]`, `fc_dims=[512]`, `dropout=0.5`, `use_bn=True`).<br>2. **Tabular (`esemble/MLP.py`)**: `MultiLayerPerceptron` com `mlp_shape=[42, 256, 128, 64, 2]`, `ReLU`, `BatchNorm1d`, entrada escalonada com `MinMaxScaler`.<br>3. **Texto/LLM (`esemble/multimodal.py`)**: `meta-llama/Llama-3.2-1B-Instruct` + adaptador LoRA (`PeftModel` em `models/Llama/domain-enriched-tokenized-1B/best`).<br>4. **Meta-Classificador**: `LogisticRegression(random_state=42, max_iter=1000)` combinando as probabilidades das 3 modalidades. | `esemble/multimodal.py`, `esemble/CNN.py`, `esemble/MLP.py`, `esemble/test.py`, `esemble/script.sh` |
| **D13** | **Persistência do `MinMaxScaler` e do Meta-Classificador** | No script de pesquisa (`esemble/multimodal.py`), o `MinMaxScaler` e o `LogisticRegression` (`self.clf`) são ajustados em memória sobre `val_loader`. Para inferência online de um único domínio em tempo real na API, ambos serão pré-ajustados no conjunto de validação e salvos em disco (`minmax_scaler_42.joblib` e `meta_logreg.joblib`) para carregamento instantâneo no startup do FastAPI. | `esemble/multimodal.py` |
| **D14** | **Correções de Paridade ao Portar `esemble/` para a API** | Foram identificadas 3 pequenas inconsistências nos scripts de experimento da pasta `esemble/` que devem ser corrigidas no módulo de inferência da API (detalhadas na Seção 3 abaixo). | `esemble/multimodal.py`, `esemble/test.py`, `esemble/script.sh` |

---

## 3. Ajustes Técnicos Mapeados na Pasta `esemble/` para a Implementação da API

Durante a análise de `esemble/multimodal.py`, `esemble/test.py` e `esemble/script.sh`, foram identificados os seguintes detalhes que o módulo de inferência da API (`api/app/pipeline/ensemble.py`) deve tratar:

1. **Normalização de Probabilidade da `CustomCNN` (`esemble/multimodal.py` linha 237)**:
   - *Como está em `esemble/multimodal.py`*: enquanto `llm_inference` e `tabular_inference` aplicam `torch.softmax(outputs, dim=-1)[:, 1]` e retornam um vetor `(N, 1)`, o método `vision_inference` faz `outputs = self.vision_model(image_batch)` e acumula os logits brutos de 2 classes `(N, 2)`.
   - *Como deve ficar na API (e no treino do meta-modelo)*: aplicar `probs = torch.softmax(outputs, dim=-1)[:, 1]` também em `vision_inference`, garantindo que `X_meta = np.hstack((p_vision, p_llm, p_tabular))` tenha shape `(N, 3)` consistente com `esemble/test.py` (linha 72), que espera exatamente 3 coeficientes no meta-modelo (`["CNN", "LLM", "MLP"]`).
2. **Nome do Atributo do Meta-Modelo (`self.clf` vs `self.meta_model`)**:
   - *Como está*: `esemble/multimodal.py` instancia `self.clf = LogisticRegression(...)`, mas `esemble/test.py` (linha 67) acessa `ensemble.meta_model.coef_[0]`.
   - *Como deve ficar na API*: padronizar o atributo como `self.clf` e expor a propriedade `meta_model` como alias (`@property def meta_model(self): return self.clf`) para manter compatibilidade total com `esemble/test.py`.
3. **Ordem dos Coeficientes no Log/Resposta**:
   - Em `esemble/multimodal.py` (linha 275), a matriz do meta-classificador é concatenada na ordem `(vision_predictions, llm_predictions, tabular_predictions)`. Portanto, os índices de `self.clf.coef_[0]` correspondem a `[0] = vision_cnn`, `[1] = text_llm`, `[2] = tabular_mlp`.
4. **Cálculo de `SOA Expire Length`**:
   - Em `data/dataset.csv`, `SOA Expire Length` representa a quantidade de dígitos inteiros do campo `expire` do SOA (`len(str(int(soa.expire)))`, tipicamente `6` ou `7`, ou `-1` quando `SOA Missing == 1`). No coletor `data/less-is-more/dns-enrich.py` (linha 440), havia um cálculo alternativo por média; na API de produção, deve ser usada a contagem de dígitos (`len(str(int(expire)))`) exatamente igual a `SOA Serial Length` para manter 100% de fidelidade com a distribuição do `dataset.csv` em que o `MLP` e a `CustomCNN` foram treinados.

---

## 4. Perguntas de Produto Opcionais (Não Bloqueantes para o MVP)

As definições padrão já estão estabelecidas nos documentos `00` a `09` para permitir a implementação completa do MVP. Caso o produto deseje customizar futuramente:

- **Limites de Rate Limit**: definidos inicialmente em `60 req/min` e `1.000 req/dia` por chave de API (configuráveis via variáveis de ambiente no `.env`).
- **Retenção de Histórico**: consultas ficam salvas em `public.domain_analyses` e `public.api_requests` com opção de limpeza periódica.
- **Servidor de E-mail (SMTP)**: no ambiente local iniciado por `./start.sh`, o container **Mailpit** (`http://localhost:8025`) captura todos os e-mails de OTP sem necessidade de credenciais externas; em produção, basta configurar `SMTP_HOST`, `SMTP_USER` e `SMTP_PASS` no `.env`.
