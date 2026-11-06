# 06 — Pipeline de Detecção e Especificação do Ensemble Multimodal

Este documento especifica integralmente a coleta em tempo real, o cálculo dos **42 atributos**, a geração da imagem **GADF 42×42 RGB**, a construção do **prompt estruturado** e a inferência do **Ensemble Multimodal (`Domain_Ensemble`)**, consolidados a partir dos dados (`*.csv`), dos scripts de enriquecimento (`data/`) e estritamente da implementação da pasta `esemble/` (`esemble/multimodal.py`, `esemble/CNN.py`, `esemble/MLP.py`, `esemble/test.py` e `esemble/script.sh`).

---

## Visão Geral do Fluxo

```
Domínio de Entrada (ex.: "fr7ehgd.duckdns.org")
   │
   ├─► 1. Limpeza e Normalização (clean_domain + remove_www_prefix + idna)
   │
   ├─► 2. Coleta Paralela:
   │      ├─► 2.1. Extrator Lexicográfico (13 atributos puros sobre o nome)
   │      ├─► 2.2. Coletor DNS ao vivo (dnspython: A, AAAA, MX, NS, CNAME, SOA -> 23 atributos)
   │      └─► 2.3. Coletor WHOIS/RDAP + GeoIP/ASN (6 atributos: has_whois, lifetime, active_time, has_country, CCR, CCA)
   │
   ├─► 3. Vetor de 42 Atributos na Ordem Canônica
   │      │
   │      ├─────────────────────────┬──────────────────────────────┐
   │      ▼                         ▼                              ▼
   │   [Escalonamento Tabular]   [Renderizador Visual GADF]    [Construtor de Prompt]
   │   MinMaxScaler (42d)        GramianAngularField           build_structured_prompt (7 tags)
   │   Tensor (1, 42) float32    Imagem RGB 42×42 -> (1,3,42,42) + Chat Template Llama-3.2 (max_len=160)
   │      │                         │                              │
   │      ▼                         ▼                              ▼
   │   Modelo Tabular (MLP)      Modelo Visual (CustomCNN)     Modelo Textual (Llama-3.2-1B + PeftModel)
   │   [42, 256, 128, 64, 2]     arch="mid_3b_b" (3 blocos)    AutoModelForSequenceClassification
   │   p_tabular ∈ [0, 1]        p_vision ∈ [0, 1]             p_llm ∈ [0, 1]
   │      │                         │                              │
   │      └─────────────────────────┼──────────────┬───────────────┘
   │                                ▼              │
   │               X_meta = np.hstack((p_vision, p_llm, p_tabular))
   │                                │
   │                                ▼
   └────────────► 4. Meta-Modelo (LogisticRegression Stacking)
                                    │
                                    ▼
                     score = clf.predict_proba(X_meta)[:, 1]
                     verdict = "malicious" if score >= 0.5 else "benign"
```

---

## 1. Coleta de Dados (`collectors/`)

### 1.1. Limpeza e Normalização (`clean_domain` / `remove_www_prefix`)
Baseado em `data/less-is-more/dns-enrich.py` e `data/filterdata.py`:
- A API exige apenas o nome do domínio (rejeita URLs completas com `http://`, caminhos ou portas com HTTP `400`).
- Remove espaços, converte para minúsculas, remove ponto final e remove o prefixo `www.` (`re.sub(r'^www\.', '', domain)`).
- Extrai `registered_domain` via `tldextract` para a consulta WHOIS/RDAP.

### 1.2. Coletor DNS ao Vivo (`collectors/dns.py` — ref.: `data/less-is-more/dns-enrich.py`)
Utiliza `dns.resolver.Resolver` (ou `dns.asyncresolver.Resolver`) com `timeout = 1.0` s e `lifetime = 1.0` s por consulta:

- **Função de estatísticas de TTL (`calculate_ttl_stats(answers)`)**:
  - Se não houver resposta: retorna `(0, 0.0, 0.0)`.
  - Caso contrário: extrai `ttl = float(answers.rrset.ttl)` (ou `float(answers.ttl)`), retornando `(len(answers), ttl, 0.0)` — ou, caso múltiplos RRsets com TTLs distintos sejam observados, `(count_distinct, median_ttl, std_ttl)`.
- **Consultas executadas**:
  1. **`A`**:
     - Em sucesso: `A Count = len(a_records)`, `IP Count = len(a_records)`, `A Missing = 0`, `A TTL Count = count`, `A Medium TTL = mean`, `A TTL Standard Deviation = std`, e extrai a lista de IPs IPv4 para enriquecimento de ASN/País (`ASN Count = len(distinct_asns)` ou `1` como fallback quando há IP resolvido).
     - Em `NoAnswer`, `NXDOMAIN` ou `Timeout`: `A Missing = 1` (demais campos `A` permanecem `0.0`).
  2. **`AAAA`**:
     - Em sucesso: `AAAA Count = len(aaaa_records)`, `AAAA Medium TTL = mean`.
     - Em falha/ausência: `AAAA Count = 0.0`, `AAAA Medium TTL = 0.0`.
  3. **`MX`**:
     - Em sucesso: `MX Count = len(mx_records)`, `MX Missing = 0`, `MX Medium TTL = mean`.
     - Em falha/ausência: `MX Missing = 1`, `MX Count = 0.0`, `MX Medium TTL = 0.0`.
  4. **`NS`**:
     - Em sucesso: `NS Count = len(ns_records)`. Em falha: `0.0`.
  5. **`CNAME`**:
     - Em sucesso: `CNAME Count = len(cname_records)`. Em falha: `0.0`.
  6. **`SOA` (com caminhamento hierárquico em até 3 níveis)**:
     - Tenta resolver `SOA` para `current_domain = domain`. Se `NoAnswer` ou `NXDOMAIN` e `len(current_domain.split(".")) > 2`, remove o primeiro rótulo (`".".join(parts[1:])`) e tenta novamente por até **3 iterações**.
     - Se encontrado (`soa = soa_records[0]`):
       - `SOA Missing = 0`
       - `SOA Retry = float(soa.retry)`
       - `SOA Refresh = float(soa.refresh)`
       - `SOA Minimum = float(soa.minimum)`
       - `SOA Expire Length = len(str(int(soa.expire)))` *(conforme os valores inteiros de número de dígitos `5, 6, 7` presentes em `data/dataset.csv` e no prompt `expire_len=6` em `data/test-prompt.py`)*
       - `SOA Serial Length = len(str(soa.serial))` (ex.: `10`)
       - `SOA TTL Count = 1`
       - `SOA Medium TTL = mean`
       - `SOA TTL Standard Deviation = std` (no dataset `acme`, quando calculado sobre 1 registro SOA, assume o próprio valor de `SOA Medium TTL` ou `0.0`; manter paridade com o dataset).
     - Se não encontrado após 3 tentativas: `SOA Missing = 1` e demais atributos `SOA` em `0.0`.

### 1.3. Coletor WHOIS/RDAP e Risco Geográfico/ASN (`collectors/whois.py` e `geo_asn.py`)
Baseado em `data/test-prompt.py`, `data/acme/explore_acme.py` e `data/acme/advanced_analysis.py`:
- **WHOIS / RDAP** (consultado sobre `registered_domain`):
  - Se o registro WHOIS/RDAP retornar data de criação (`creation_date`) válida:
    - `has_whois = 1.0`
    - `lifetime`: tempo de vida em dias (`(expiration_date - creation_date).total_seconds() / 86400.0`, ou `(now - creation_date).days` caso `expiration_date` ausente; `>= 0.0`).
    - `active_time`: tempo ativo em dias (`(updated_date - creation_date).total_seconds() / 86400.0`, ou igual a `lifetime` se `updated_date` ausente; `>= 0.0`).
  - Se WHOIS falhar, sofrer timeout ou tiver datas redigidas/ausentes:
    - `has_whois = 0.0`, `lifetime = 0.0`, `active_time = 0.0`.
- **País, ASN, `CCR` e `CCA`**:
  - A partir dos IPs resolvidos nos registros `A`/`AAAA` (e/ou país do registrante no WHOIS), consulta base local GeoIP2/MaxMind (`GeoLite2-Country.mmdb` e `GeoLite2-ASN.mmdb`):
    - `ASN Count`: número de ASNs distintos dos IPs resolvidos (se `A Count > 0` e sem base ASN, fallback `1`).
    - `has_country`: `1.0` se pelo menos um código de país ISO foi identificado, `0.0` caso contrário.
    - `CCR` (*Country Code Risk*) e `CCA` (*Country/ASN Risk*): probabilidades empíricas de risco no intervalo `[0.0, 1.0]` consultadas em tabela estática exportada do conjunto de treino (`models/ensemble/ccr_cca_lookup.json`, mapeando `country_code -> CCR` e `asn -> CCA`, com fallback `0.0` quando `has_country == 0.0` ou não tabelado).

---

## 2. Especificação e Ordem Exata dos 42 Atributos (`features/`)

Conforme verificado nos logs de treinamento (`logs/xgb_seed0.log`, `logs/rf_seed0.log`), no treino do modelo tabular (`mlp-test/training.py`) e em `esemble/multimodal.py` (`architecture: [42, 256, 128, 64, 2]`), o vetor de entrada tabular e de geração da imagem GADF possui **exatamente 42 atributos numéricos (`float32`) na seguinte ordem canônica**:

### Grupo A — Atributos Lexicais do Nome do Domínio (Índices 0 a 12 — 13 atributos)
Calculados sobre a string `name` normalizada (ex.: `fr7ehgd.duckdns.org`), onde $L = \text{len}(\text{name})$, $V = \text{set}(\text{'aeiou'})$, $C = \text{set}(\text{'bcdfghjklmnpqrstvwxyz'})$, $D = \text{set}(\text{'0123456789'})$:

| Índice | Nome da Coluna | Tipo | Fórmula / Regra de Construção |
|--------|----------------|------|-------------------------------|
| `0` | `lex_length` | `float` | Comprimento total da string do domínio: $L = \text{len}(d)$ |
| `1` | `lex_digit_count` | `float` | Quantidade de dígitos numéricos: $\sum_{c \in d} \mathbb{I}(c \in D)$ |
| `2` | `lex_digit_ratio` | `float` | Razão de dígitos: $\text{lex\_digit\_count} / \max(L, 1)$ |
| `3` | `lex_vowel_ratio` | `float` | Razão de vogais: $\left(\sum_{c \in d} \mathbb{I}(c \in V)\right) / \max(L, 1)$ *(ou sobre total de letras; manter função única compartilhada)* |
| `4` | `lex_consonant_ratio` | `float` | Razão de consoantes: $\left(\sum_{c \in d} \mathbb{I}(c \in C)\right) / \max(L, 1)$ |
| `5` | `lex_special_char_ratio` | `float` | Razão de caracteres não alfanuméricos (`.` , `-`, `_`): $\left(\sum_{c \in d} \mathbb{I}(\neg c.\text{isalnum}())\right) / \max(L, 1)$ |
| `6` | `lex_entropy` | `float` | Entropia de Shannon dos caracteres: $-\sum_{c \in \text{unique}(d)} \frac{\text{count}(c)}{L} \log_2\left(\frac{\text{count}(c)}{L}\right)$ |
| `7` | `lex_hyphen_count` | `float` | Quantidade de hífens (`'-'`): $d.\text{count}(\text{'-'})$ |
| `8` | `lex_starts_with_digit` | `float` | `1.0` se $d[0] \in D$, senão `0.0` |
| `9` | `lex_has_repeated_chars` | `float` | `1.0` se existe caractere repetido adjacente ($\exists i: d[i] == d[i+1]$), senão `0.0` |
| `10` | `lex_max_consonant_seq` | `float` | Comprimento da maior sequência contígua de consoantes: $\max(\text{len}(m) \text{ for } m \text{ in re.findall}(r\text{'[bcdfghjklmnpqrstvwxyz]+'}, d))$ |
| `11` | `lex_max_digit_seq` | `float` | Comprimento da maior sequência contígua de dígitos: $\max(\text{len}(m) \text{ for } m \text{ in re.findall}(r\text{'\d+'}, d))$ |
| `12` | `lex_unique_char_ratio` | `float` | Razão de caracteres únicos: $\text{len}(\text{set}(d)) / \max(L, 1)$ |

### Grupo B — Atributos de DNS, TTL e SOA (Índices 13 a 35 — 23 atributos)
Na ordem exata de `DOMAIN_FEATURE_ORDER` (`data/less-is-more/dns-enrich.py` e `data/dataset.csv`):

| Índice | Nome da Coluna | Valor quando ausente | Regra de Construção |
|--------|----------------|----------------------|---------------------|
| `13` | `A Count` | `0.0` | Quantidade de registros `A` resolvidos |
| `14` | `IP Count` | `0.0` | Quantidade de IPs resolvidos |
| `15` | `MX Count` | `0.0` | Quantidade de registros `MX` |
| `16` | `NS Count` | `0.0` | Quantidade de registros `NS` |
| `17` | `A Missing` | `1.0` | `0.0` se `A Count > 0`, `1.0` se `NoAnswer`/`NXDOMAIN`/`Timeout` |
| `18` | `ASN Count` | `0.0` | Quantidade de ASNs distintos dos IPs (`1.0` se `A Count > 0` e ASN único) |
| `19` | `SOA Retry` | `0.0` | Campo `retry` do registro `SOA` (segundos) |
| `20` | `AAAA Count` | `0.0` | Quantidade de registros `AAAA` (IPv6) |
| `21` | `MX Missing` | `1.0` | `0.0` se `MX Count > 0`, `1.0` caso contrário |
| `22` | `A TTL Count` | `0.0` | Contagem de TTLs da resposta `A` |
| `23` | `CNAME Count` | `0.0` | Quantidade de registros `CNAME` |
| `24` | `SOA Minimum` | `0.0` | Campo `minimum` do registro `SOA` (segundos) |
| `25` | `SOA Missing` | `1.0` | `0.0` se `SOA` encontrado (até 3 níveis), `1.0` caso contrário |
| `26` | `SOA Refresh` | `0.0` | Campo `refresh` do registro `SOA` (segundos) |
| `27` | `A Medium TTL` | `0.0` | TTL mediano/médio dos registros `A` |
| `28` | `MX Medium TTL` | `0.0` | TTL mediano/médio dos registros `MX` |
| `29` | `SOA TTL Count` | `0.0` | `1.0` se `SOA` encontrado, `0.0` se `SOA Missing == 1` |
| `30` | `SOA Medium TTL` | `0.0` | TTL do registro `SOA` |
| `31` | `AAAA Medium TTL` | `0.0` | TTL mediano/médio dos registros `AAAA` |
| `32` | `SOA Expire Length` | `0.0` | Número de dígitos de `soa.expire`: `len(str(int(soa.expire)))` |
| `33` | `SOA Serial Length` | `0.0` | Número de dígitos de `soa.serial`: `len(str(soa.serial))` |
| `34` | `A TTL Standard Deviation` | `0.0` | Desvio padrão dos TTLs `A` |
| `35` | `SOA TTL Standard Deviation` | `0.0` | Desvio padrão / TTL do registro `SOA` |

### Grupo C — Atributos WHOIS, Tempo e Risco (Índices 36 a 41 — 6 atributos)

| Índice | Nome da Coluna | Valor quando ausente | Regra de Construção |
|--------|----------------|----------------------|---------------------|
| `36` | `has_whois` | `0.0` | `1.0` se WHOIS/RDAP retornou dados válidos de ciclo de vida, `0.0` caso contrário |
| `37` | `lifetime` | `0.0` | Tempo de vida do domínio em dias (`float`) |
| `38` | `active_time` | `0.0` | Tempo ativo do domínio em dias (`float`) |
| `39` | `has_country` | `0.0` | `1.0` se país identificado para o domínio/IP, `0.0` caso contrário |
| `40` | `CCR` | `0.0` | *Country Code Risk* (`float` em `[0, 1]`) |
| `41` | `CCA` | `0.0` | *Country / ASN Risk* (`float` em `[0, 1]`) |

### Escalonamento Tabular (`MinMaxScaler`)
Conforme `esemble/test.py` (linhas 25–30) e `mlp-test/training.py` (linhas 54–58):
- Os 42 atributos numéricos são normalizados para o intervalo `[0, 1]` utilizando `sklearn.preprocessing.MinMaxScaler`.
- **Artefato obrigatório**: durante o empacotamento para produção, o `MinMaxScaler` ajustado sobre `data/acme/fixed_train.csv` (ou `val.csv`) deve ser salvo em `models/ensemble/tabular_scaler.joblib` e carregado no startup da API (`scaler.transform(X_42)`).

---

## 3. Geração de Imagens GADF (`renderers/images.py`)

Baseado em `esemble/multimodal.py` (`from pyts.image import GramianAngularField`, `ImagePathDataset`), `esemble/CNN.py` (`IMG_SIZE = 42, IN_CHANNELS = 3`) e `esemble/script.sh` (`CustomCNN0-GADF-42_30.pth`, `images/GADF/`):

1. **Entrada**: vetor 1D com os **42 atributos numéricos** do domínio, shape `(1, 42)`.
2. **Transformação GADF (*Gramian Angular Difference Field*)**:
   - Usa `GramianAngularField(image_size=42, method='difference', sample_range=(-1, 1))` da biblioteca `pyts.image`.
   - Gera uma matriz 2D de dimensão `(42, 42)` com valores em `[-1, 1]`.
3. **Conversão para RGB 3 Canais (`3×42×42`)**:
   - Na pipeline original (`ImagePathDataset` em `esemble/multimodal.py`, linhas 91–105), as imagens GADF salvas com colormap (ex.: `viridis`/`rainbow` do matplotlib ou convertidas via `Image.open(...).convert('RGB')`) passam por:
     ```python
     transforms.Compose([
         transforms.Resize((42, 42)),
         transforms.ToTensor(), # float32 em [0.0, 1.0], shape (3, 42, 42)
     ])
     ```
   - Na API, essa conversão (`GADF 42x42 -> colormap/RGB PIL Image -> ToTensor()`) é executada **inteiramente em memória (RAM)**, produzindo:
     - O tensor `torch.FloatTensor` de shape `(1, 3, 42, 42)` para a `CustomCNN`;
     - Uma string `data:image/png;base64,...` (buffer `io.BytesIO`) para retorno no JSON da API e visualização na interface web.

---

## 4. Construção do Prompt de Texto (`renderers/prompts.py`)

### 4.1. Estrutura de 7 Linhas (`build_structured_prompt` — `data/test-prompt.py`)
A função `build_structured_prompt(row)` converte os atributos brutos (antes do `MinMaxScaler`) em um texto compacto (~80–120 tokens) com 7 seções tagueadas:

```python
def fmt_bool(val) -> str:
    if isinstance(val, str):
        val = val.strip().lower() == "true"
    return "yes" if bool(float(val)) else "no"

def build_structured_prompt(row: dict) -> str:
    lines = []
    # 1. [name]
    lines.append(f"[name]: {row['name']}")
    # 2. [entropy] (4 casas decimais)
    lines.append(f"[entropy]: {round(float(row['lex_entropy']), 4)}")
    # 3. [whois]
    is_whois = bool(float(row.get('has_whois', 0)))
    if is_whois:
        lifetime = round(float(row.get('lifetime', 0)), 2)
        active   = round(float(row.get('active_time', 0)), 2)
        lines.append(f"[whois]: present=yes | lifetime={lifetime}d active={active}d")
    else:
        lines.append("[whois]: present=no")
    # 4. [risk]
    has_country = fmt_bool(row.get('has_country', 0))
    ccr = round(float(row.get('CCR', 0)), 4)
    cca = round(float(row.get('CCA', 0)), 4)
    lines.append(f"[risk]: ccr={ccr} cca={cca} country={has_country}")
    # 5. [dns]
    lines.append(
        f"[dns]: A={int(row['A Count'])} AAAA={int(row['AAAA Count'])} "
        f"MX={int(row['MX Count'])} NS={int(row['NS Count'])} "
        f"CNAME={int(row['CNAME Count'])} IPs={int(row['IP Count'])} "
        f"ASNs={int(row['ASN Count'])} "
        f"a_missing={fmt_bool(row['A Missing'])} mx_missing={fmt_bool(row['MX Missing'])}"
    )
    # 6. [ttl]
    lines.append(
        f"[ttl]: a_med={round(float(row['A Medium TTL']), 1)} "
        f"a_std={round(float(row['A TTL Standard Deviation']), 1)} "
        f"a_distinct={int(row['A TTL Count'])} "
        f"mx_med={round(float(row['MX Medium TTL']), 1)} "
        f"aaaa_med={round(float(row['AAAA Medium TTL']), 1)}"
    )
    # 7. [soa]
    lines.append(
        f"[soa]: missing={fmt_bool(row['SOA Missing'])} "
        f"refresh={int(row['SOA Refresh'])} retry={int(row['SOA Retry'])} "
        f"min={int(row['SOA Minimum'])} expire_len={int(row['SOA Expire Length'])} "
        f"serial_len={int(row['SOA Serial Length'])} "
        f"med={round(float(row['SOA Medium TTL']), 1)} "
        f"std={round(float(row['SOA TTL Standard Deviation']), 1)} "
        f"distinct={int(row['SOA TTL Count'])}"
    )
    return "\n".join(lines)
```

### 4.2. Formatação Chat Template e Tokenização (`TextInferenceDataset` — `esemble/multimodal.py`)
Conforme `esemble/multimodal.py` (linhas 36–89):
- **System Instruction**:
  ```python
  SYSTEM_INSTRUCTION = (
      "You are a cybersecurity classifier. "
      "Analyze the domain record below and classify it as benign or malicious."
  )
  ```
- **Aplicação do Chat Template**:
  ```python
  messages = [{"role": "user", "content": f"{SYSTEM_INSTRUCTION}\n\n{structured_prompt}"}]
  formatted_text = llm_tokenizer.apply_chat_template(
      messages, tokenize=False, add_generation_prompt=True
  )
  ```
- **Tokenização**:
  - `truncation=True`, `padding='max_length'`, `max_length=160`, `return_tensors='pt'`.
  - `llm_tokenizer.pad_token = llm_tokenizer.eos_token` e `llm_model.config.pad_token_id = llm_tokenizer.eos_token_id`.

---

## 5. Especificação do Ensemble Multimodal (`esemble/`)

Toda a lógica de inferência segue estritamente a arquitetura definida na pasta `esemble/` (`esemble/multimodal.py`, `esemble/CNN.py`, `esemble/MLP.py`, `esemble/test.py` e `esemble/script.sh`).

### 5.1. Modelo 1 — Visão (`ConvolutionalNeuralNetwork` em `esemble/CNN.py`)
- **Construção**:
  ```python
  arch = {"name": "mid_3b_b", "blocks": [(32, 3, True), (64, 3, True), (128, 3, True)]}
  fc_dims = [512]
  vision_model = build_model(arch=arch, fc_dims=fc_dims, dropout=0.5, use_bn=True)
  ```
- **Camadas internas (`esemble/CNN.py`, linhas 12–79)**:
  - `IMG_SIZE = 42`, `IN_CHANNELS = 3`, `NUM_CLASSES = 2`.
  - `conv_block`: 3 blocos compostos por `Conv2d(ch_in, ch_out, kernel_size=3, padding=1) -> BatchNorm2d(ch_out) -> ReLU(inplace=True) -> MaxPool2d(kernel_size=2, stride=2)`.
    - Dimensões espaciais: `42×42` → `21×21` (32 ch) → `10×10` (64 ch) → `5×5` (128 ch) → `flat_size = 128 * 5 * 5 = 3200`.
  - `fc_block`: `Linear(3200, 512) -> ReLU(inplace=True) -> Dropout(0.5) -> Linear(512, 2)`.
- **Checkpoint**: `CustomCNN0-GADF-42_30.pth` (carregado via `torch.load(cnn_path, map_location=device)` lendo `checkpoint['model_state_dict']` ou `checkpoint` direto).
- **Saída de Probabilidade (`p_vision`)**:
  - `logits = vision_model(image_tensor)`
  - `p_vision = torch.softmax(logits, dim=-1)[:, 1].reshape(-1, 1)` (probabilidade da classe `1 = malicious`).
  - *(Nota de correção em relação ao protótipo `esemble/multimodal.py` linha 237: no protótipo `vision_model(image_batch)` era concatenado sem `softmax(...)[:, 1]`, gerando 2 colunas em vez de 1 e conflitando com `feature_names = ["CNN", "MLP", "LLM"]` em `esemble/test.py` linha 72; na API padroniza-se `torch.softmax(logits, dim=-1)[:, 1].reshape(-1, 1)` para que cada modalidade contribua com sua probabilidade escalar da classe maliciosa, ou mantém-se compatibilidade com o `meta_model` salvo).*

### 5.2. Modelo 2 — Texto / LLM (`Llama-3.2-1B-Instruct` + `PeftModel` em `esemble/multimodal.py`)
- **Construção (`esemble/multimodal.py`, linhas 129–150)**:
  - Modelo base: `AutoModelForSequenceClassification.from_pretrained("meta-llama/Llama-3.2-1B-Instruct", num_labels=2)` (suportando quantização 4-bit `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)` para reduzir uso de VRAM em produção).
  - Adaptador PEFT (DoRA/LoRA): `PeftModel.from_pretrained(llm_model, llm_path)` apontando para `models/Llama/domain-enriched-tokenized-1B/best`.
  - Tokenizer: `AutoTokenizer.from_pretrained(llm_path, local_files_only=True)`.
- **Saída de Probabilidade (`p_llm`)**:
  - Executado com `torch.cuda.amp.autocast(enabled=device.type == 'cuda')`.
  - `probs = torch.softmax(outputs.logits, dim=-1).detach().cpu().numpy()`
  - `p_llm = probs[:, 1].reshape(-1, 1)`.

### 5.3. Modelo 3 — Tabular (`MultiLayerPerceptron` em `esemble/MLP.py`)
- **Construção (`esemble/multimodal.py`, linhas 152–177 e `esemble/MLP.py`, linhas 49–74)**:
  ```python
  runs = {
      'model_type':     'MLP',
      'architecture':   [42, 256, 128, 64, 2],
      'activation':     nn.ReLU,
      'regularization': 0.0,
      'model_kwargs':   {},
  }
  tab_model = MultiLayerPerceptron(
      runs['architecture'],
      runs['regularization'],
      runs['activation'],
  )
  ```
- **Camadas internas (`esemble/MLP.py`)**:
  - `make_layer(42, 256, 0.0, nn.ReLU)`: `Linear(42, 256) -> ReLU() -> BatchNorm1d(256) -> Dropout(0.0)`
  - `make_layer(256, 128, 0.0, nn.ReLU)`: `Linear(256, 128) -> ReLU() -> BatchNorm1d(128) -> Dropout(0.0)`
  - `make_layer(128, 64, 0.0, nn.ReLU)`: `Linear(128, 64) -> ReLU() -> BatchNorm1d(64) -> Dropout(0.0)`
  - `make_output_layer(64, 2)`: `Linear(64, 2)`
- **Checkpoint**: `mlp-test/training_results/models/best_mlp.pt` (`val_accuracy=0.9056`, `test_accuracy=0.9062`, `test_auc=0.9651`).
- **Saída de Probabilidade (`p_tabular`)**:
  - `outputs = tab_model(X_tabular_scaled_tensor)`
  - `probs = torch.softmax(outputs, dim=-1).detach().cpu().numpy()`
  - `p_tabular = probs[:, 1].reshape(-1, 1)`.

### 5.4. Combinador Meta-Modelo (`LogisticRegression` em `esemble/multimodal.py` e `esemble/test.py`)
- **Arquitetura**: `LogisticRegression(random_state=42, max_iter=1000)` (`self.clf` / `self.meta_model`).
- **Vetor de Meta-Features**:
  ```python
  X_meta = np.hstack((p_vision, p_llm, p_tabular))
  ```
- **Treinamento Offline e Persistência para a API**:
  - Em `esemble/test.py`, o meta-modelo é ajustado (`ensemble.fit(...)`) sobre o conjunto de validação (`data/acme/val.csv` + `images/GADF/val`).
  - Para a API em produção, um script de preparação (`scripts/export_ensemble_artifacts.py` ou etapa de build) executa o `.fit()` uma única vez no conjunto de validação e salva:
    1. `models/ensemble/tabular_scaler.joblib` (`MinMaxScaler` ajustado nos 42 atributos);
    2. `models/ensemble/meta_logreg.joblib` (`LogisticRegression` ajustado sobre `X_meta_val`).
  - No startup da API, o `Domain_Ensemble` carrega `meta_logreg.joblib` diretamente em `self.clf`, ficando pronto para chamar `self.clf.predict_proba(X_meta)` em milissegundos por requisição.
- **Decisão Final**:
  - `score = float(self.clf.predict_proba(X_meta)[0, 1])`
  - `verdict = "malicious" if score >= threshold else "benign"` (onde `threshold = 0.5` por padrão, equivalente a `np.argmax(probabilities, axis=1)` em `esemble/multimodal.py` linha 313).

---

## 6. Política de Dados Ausentes e Degradação Graciosa

1. **Ausência de registros DNS ou WHOIS não é falha de sistema, mas sinal preditivo**:
   - Conforme `data/acme/explore_acme.py`, domínios maliciosos frequentemente não possuem WHOIS (`has_whois=0`) ou registros `MX`/`SOA`. O próprio conjunto de 42 atributos possui flags dedicadas (`A Missing`, `MX Missing`, `SOA Missing`, `has_whois`, `has_country`) e valores padrão `0.0` que foram usados durante o treinamento da `CustomCNN`, do `MLP` e do `Llama-3.2-1B`.
   - Quando um timeout de rede (ex.: servidor WHOIS lento além de `3.0s`) impedir a coleta de um enriquecimento, os atributos correspondentes recebem o valor padrão de ausência (`has_whois=0.0, lifetime=0.0, active_time=0.0`) e a resposta da API sinaliza `"partial": true` com uma nota em `evidence.notes`.
2. **Orçamento de Tempo (Latency Budget)**:
   - Coleta DNS (6 queries em paralelo via `asyncio.gather`): máx. `1.2s`.
   - Coleta RDAP/WHOIS (em paralelo ao DNS): máx. `3.0s`.
   - Cálculo de atributos + GADF 42×42 + Prompt: `< 15ms`.
   - Inferência `Domain_Ensemble` (CNN + MLP + Llama-3.2-1B + LogReg): `~50–250ms` em GPU (`cuda`).
   - Tempo total esperado (cache miss): `~0.5s – 3.3s`; (cache hit): `< 15ms`.
