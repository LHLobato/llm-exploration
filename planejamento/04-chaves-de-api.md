# 04 — Chaves de API

## Formato

```
dmd_<prefixo 8 chars>_<segredo 32 bytes em base62/base64url>
ex.: dmd_a1B2c3D4_9fK...(43 chars)
```

- Gerada com CSPRNG (`secrets.token_urlsafe(32)`).
- `key_prefix` (`dmd_a1B2c3D4`) é público e usado para localizar a linha e identificar a chave na UI/logs.

## Armazenamento do hash

- Como o segredo tem alta entropia, **não é necessário bcrypt/argon2**; usar `HMAC-SHA256(API_KEY_PEPPER, chave_completa)` e guardar em `key_hash` (bytea).
- O pepper fica em variável de ambiente (nunca no BD). Rotação de pepper: suportar `pepper_id` se necessário no futuro.
- Comparação em tempo constante (`hmac.compare_digest`).
- A chave em texto puro **nunca** é gravada em BD, log ou APM.

## Geração (somente maior de 18 anos)

`POST /keys` (JWT obrigatório):
1. Verifica sessão e `is_adult(birth_date)`; senão `403 age_restricted`.
2. Verifica que não há chave ativa (limite de 1; ou N configurável).
3. Gera chave, calcula hash, insere em `api_keys`.
4. Responde `201` com a chave **em texto puro, uma única vez**:

```json
{ "id": "uuid", "name": "default", "key": "dmd_...", "key_prefix": "dmd_a1B2c3D4",
  "created_at": "...", "shown_once": true }
```

Cabeçalhos `Cache-Control: no-store`.

### Exibição única — requisito explícito na tela

A UI deve mostrar, em destaque (banner de alerta, não apenas texto pequeno):

> **Copie sua chave agora. Ela será exibida apenas uma vez e não poderá ser recuperada.** Se você perdê-la, será necessário revogá-la e gerar uma nova.

- Botão "Copiar" + checkbox obrigatório "Já copiei minha chave" antes de fechar.
- Aviso ao tentar sair da tela (`beforeunload`).
- Não guardar a chave em `localStorage`, estado persistente ou URL.

## Validação em cada requisição

Dependência `require_api_key`:
1. Lê `X-API-Key` (ou `Authorization: Bearer dmd_...`).
2. Valida formato e extrai prefixo; busca candidatos por `key_prefix`.
3. Calcula HMAC e compara com `key_hash`.
4. Exige `status='active'`, não expirada, e dono ainda `is_adult` e com conta ativa.
5. Atualiza `last_used_at` (em lote/assíncrono para não penalizar latência).
6. Retorna `401` (ausente/inválida) ou `403` (revogada, expirada, restrita por idade); mensagens genéricas, sem revelar se o prefixo existe.
7. Cache em memória curto (ex.: 30–60 s) do resultado de validação por `key_hash`, com invalidação na revogação.

## Acesso web (conflito resolvido)

Requisito: "ao logar, a chave é recuperada e validada automaticamente". Como o hash é irreversível, **o texto puro não é recuperável**. Solução adotada:

- Na web, a requisição de consulta carrega o **JWT do usuário**; o backend resolve a chave ativa do usuário (`api_keys` onde `user_id = sub` e `status='active'`), **valida-a** (status, validade, idade) e registra o uso nessa chave (`channel='web'`).
- Assim a chave é "sempre informada" no sentido lógico (toda consulta está vinculada a uma chave válida) sem nunca precisar do segredo em texto puro no navegador.
- Se o usuário não tem chave ativa: a tela de consulta orienta a gerar uma (ou mostra a restrição de idade).
- `GET /me` retorna `{ has_active_key, key_prefix, key_status, can_generate_key }` — nunca o segredo.

Alternativa **não recomendada**: guardar a chave cifrada de forma reversível para reexibir/injetar. Contradiz "armazenada com hash" e amplia o impacto de vazamentos.

## Revogação e rotação

- `DELETE /keys/{id}` → `status='revoked'`, `revoked_at=now()`, invalida cache.
- Rotação: `POST /keys/{id}/rotate` revoga a antiga e cria nova (mostrada uma vez); opcional período de graça curto.
- Expiração opcional (`expires_at`), com aviso por email antes de vencer.
- Detecção de vazamento: registrar o prefixo no formato reconhecível (`dmd_`) para permitir *secret scanning* em repositórios públicos.

## Testes obrigatórios

Hash nunca igual à chave; comparação constante; chave revogada rejeitada imediatamente; menor de 18 não gera; segunda geração com chave ativa falha; nenhum log contém a chave.
